"""Execution planner: decide where each stage of a solve runs, by measuring.

The planner does not assume "GPU is always faster". For every backend that is
available and whose memory budget fits, it builds the real solver for the real
case and times a few real time steps (after JIT/plan warm-up). The fastest
measured configuration wins, and the plan records *why* for every stage, so
the UI can show the reasoning.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


from .backend import BackendUnavailable, available_backends, get_backend
from .memory import estimate_structured, estimate_unstructured, fmt_bytes
from .probe import SystemProfile, cached_profile


@dataclass
class Placement:
    stage: str
    device: str
    method: str
    reason: str


@dataclass
class ExecutionPlan:
    backend: str
    device: str
    precision: str
    pressure: str
    diffusion: str
    fmg: List[int]
    placements: List[Placement] = field(default_factory=list)
    memory: Dict[str, dict] = field(default_factory=dict)
    measured_ms_per_step: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    precision_reason: str = ""  # why this precision (accuracy test, user choice, device)
    precision_by_backend: Dict[str, str] = field(default_factory=dict)  # the decision for every candidate

    def to_dict(self) -> dict:
        return asdict(self)


def _candidates(case_backend: str) -> List[str]:
    avail = available_backends()
    if case_backend and case_backend != "auto":
        if case_backend not in avail:
            raise BackendUnavailable(f"backend {case_backend!r} is not available here (have {avail})")
        return [case_backend]
    return avail


def _time_solver(solver, steps: int, warmup: int = 5) -> float:
    """Best of two timed blocks after a warm-up (GPU kernels are compiled on first use
    and clocks ramp up; measured on an RTX 4060 Laptop GPU, 2 warm-up steps overstated
    the CuPy time per step by up to 3x)."""
    be = solver.be
    for _ in range(warmup if be.is_gpu else 2):
        solver.step()
    best = float("inf")
    for _ in range(2):
        t0 = be.timer()
        for _ in range(steps):
            solver.step()
        best = min(best, (be.timer() - t0) / steps)
    return best


def plan_structured(case, autotune: bool = True, steps: Optional[int] = None,
                    profile: Optional[SystemProfile] = None, keep_solver: bool = True):
    """Return (plan, solver). ``solver`` is the already-built winning solver
    (it has advanced ``2 + steps`` time steps) or None when not autotuning."""
    import numpy as np

    from ..structured.sequencing import fmg_factors
    from ..structured.solver import MACSolver, fp32_admissible, fp32_round_off_bound, resolve_dtype

    profile = profile or cached_profile()
    cands = _candidates(case.backend)
    has_obs = bool(case.obstacles)
    cells = case.nx * case.ny
    if steps is None:
        steps = 6 if cells <= 256 ** 2 else (3 if cells <= 1024 ** 2 else 1)

    memory, timings, solvers, notes = {}, {}, {}, []
    precision_why: Dict[str, str] = {}
    for name in cands:
        be = get_backend(name)
        dtype = resolve_dtype(case, be)
        free, _total = be.mem_info()
        est = estimate_structured(case.nx, case.ny, case.energy, dtype.itemsize,
                                  "mg" if has_obs else "fft", True, has_obs, be.device, free)
        memory[name] = est.to_dict()
        if not est.fits:
            notes.append(f"{name}: skipped, " + " ".join(est.advice))
            continue
        if not autotune:
            continue
        try:
            s = MACSolver(case, backend=be, dtype=dtype)
            timings[name] = _time_solver(s, steps) * 1e3
            solvers[name] = s
        except Exception as exc:  # a backend that fails at runtime is simply not chosen
            notes.append(f"{name}: failed during autotune ({exc})")
            continue
        if case.precision == "fp32":
            bound = fp32_round_off_bound(case, s.dt)
            precision_why[name] = "user choice (fp32)"
            if case.steady_tol is not None and bound > case.steady_tol:
                notes.append(f"precision fp32: its round-off bound {bound:.1e} exceeds steady_tol {case.steady_tol:g}; "
                             "the residual will likely stall above it and the run end at max_steps")
        elif case.precision == "fp64":
            precision_why[name] = "user choice (fp64)"
        elif be.is_gpu and not (profile.best_gpu is not None and profile.best_gpu.is_datacenter_fp64):
            # consumer GPU: FP64 runs at a fraction of the FP32 rate, so FP32 is worth a look,
            # but only for a case whose own convergence criterion FP32 can meet
            ok, why = fp32_admissible(case, s.dt)
            precision_why[name] = why
            if ok:
                try:
                    s32 = MACSolver(case, backend=be, dtype=np.float32)
                    t32 = _time_solver(s32, steps) * 1e3
                    notes.append(f"{name}: FP64 {timings[name]:.2f} ms/step, FP32 {t32:.2f} ms/step")
                    if t32 < timings[name]:
                        timings[name], solvers[name] = t32, s32
                        precision_why[name] += "; FP32 measured faster"
                    else:
                        precision_why[name] += "; FP64 measured as fast, kept"
                except Exception as exc:
                    notes.append(f"{name}: FP32 candidate failed ({exc})")
        else:
            precision_why[name] = "FP64 (CPU, or a GPU with full-rate FP64)"
    if autotune and timings:
        best = min(timings, key=timings.get)
    else:
        fitting = [n for n in cands if memory.get(n, {}).get("fits", True)]
        if not fitting:
            raise MemoryError("case does not fit in memory on any backend: " + "; ".join(notes))
        best = "cupy" if "cupy" in fitting and cells >= 128 ** 2 else ("numba" if "numba" in fitting else fitting[0])
    be = get_backend(best)
    solver = solvers.get(best) if keep_solver else None
    if solver is None:
        kept = solvers.get(best)
        solver = MACSolver(case, backend=be, dtype=kept.dtype if kept is not None else None)
    for n in list(solvers):
        if n != best:
            del solvers[n]
    desc = solver.describe()
    gpu = profile.best_gpu
    plan = ExecutionPlan(
        backend=best, device=be.device, precision=desc["precision"],
        pressure=desc["pressure_detail"], diffusion=desc["diffusion"],
        fmg=fmg_factors(case), memory=memory,
        measured_ms_per_step={k: round(v, 3) for k, v in timings.items()}, notes=notes,
        precision_reason=precision_why.get(best, "FP64 (default)"), precision_by_backend=dict(precision_why),
    )
    dev_label = f"GPU ({gpu.name})" if be.is_gpu and gpu else ("CPU" if not be.is_gpu else "GPU")
    stencil_impl = {"numpy": "vectorised NumPy", "numba": "Numba fused multi-core kernels",
                    "cupy": "CuPy device kernels"}[best]
    why = (f"fastest measured: {timings[best]:.2f} ms/step vs "
           + ", ".join(f"{k} {v:.2f}" for k, v in timings.items() if k != best)) if len(timings) > 1 else \
        "only available backend" if len(cands) == 1 else "selected by cost model"
    pt = solver.poisson_timings
    p_reason = ("autotuned: " + ", ".join(f"{k} {v * 1e3:.2f} ms" for k, v in pt.items())) if pt else (
        "immersed obstacles make the operator non-separable" if has_obs else "user choice")
    plan.placements = [
        Placement("Numerical precision", dev_label, desc["precision"], plan.precision_reason),
        Placement("Case setup, masks, eigen-decompositions", "CPU", "NumPy/LAPACK (once)",
                  "one-off O(N) / O(n^3) work; results moved to the compute device"),
        Placement("Convection-diffusion stencils", dev_label, f"{stencil_impl}, {case.scheme}",
                  why),
        Placement("Pressure projection", dev_label, desc["pressure_detail"], p_reason),
        Placement("Viscous / thermal diffusion", dev_label, desc["diffusion"],
                  "Crank-Nicolson only where explicit diffusion would limit the time step"),
        Placement("CFL & residual reductions", dev_label, "on-device reductions",
                  "only scalars cross to the host"),
        Placement("Live visualisation", "CPU", "downsampled snapshots, <=4 fps",
                  "keeps PCIe traffic negligible"),
        Placement("Checkpoints", "CPU thread -> disk", "asynchronous .npz",
                  "I/O overlaps with computation"),
    ]
    if plan.fmg:
        plan.placements.append(Placement("Steady-state acceleration", dev_label,
                                         "FMG grid sequencing " + " -> ".join(f"1/{f}" for f in plan.fmg) + " -> 1",
                                         "coarse grids converge 16x/4x cheaper and seed the fine grid"))
    if be.is_gpu and gpu is not None:
        dmma = gpu.is_datacenter_fp64 and "eig" in desc["pressure_detail"]
        tc = ("FP64 Tensor-Core GEMMs (cuBLAS DMMA) in the dense transforms" if dmma
              else "not used: FFT transforms beat GEMM here, and TF32 would degrade the pressure solve")
        plan.placements.append(Placement("Tensor Cores", dev_label, tc, "accuracy first, then speed"))
    plan.placements.append(Placement("RT cores / OptiX", "-", "not used",
                                     "no radiation or particle-geometry model in this case"))
    if profile.on_battery:
        plan.notes.append("On battery power: expect throttled clocks.")
    est = memory.get(best)
    if est:
        plan.notes.append(f"Memory: ~{fmt_bytes(est['required'])} of {fmt_bytes(est['available'])} free on {be.device}.")
    return plan, solver


# --------------------------------------------------------------------------
# unstructured (SIMPLE/SIMPLEC) solver
# --------------------------------------------------------------------------
#
# A SIMPLEC iteration on CuPy launches thousands of small kernels and syncs the
# host for every Krylov dot product, so its cost has a floor that does not
# shrink with the mesh (measured on an RTX 4060 Laptop GPU, FP64: 15-30 ms per
# iteration from 1k to 262k cells, where the CPU needs 1.6 ms to 490 ms;
# benchmarks/simplec_iteration_cost.py). The planner therefore
#
# 1. times the real solver on the CPU;
# 2. skips the GPU when the CPU is already faster than the GPU's *floor* (one
#    iteration on a tiny mesh, measured once per machine and cached): a larger
#    case cannot make a GPU iteration cheaper than that;
# 3. otherwise times the real solver on the GPU as well and keeps the faster.
#
# Each timed window is one AMG rebuild period, so the host-side hierarchy setup
# (pyamg on the CPU, FlowTrace SA-AMG for the GPU) is amortised as in the run.
# The kept solver continues from the iterations spent on timing.

PLANNER_CACHE = os.path.join(os.path.expanduser("~"), ".flowtrace", "planner.json")
GPU_FLOOR_MARGIN = 0.7  # skip the GPU when t_cpu < margin * floor (the floor is a lower bound)


@dataclass
class UnstructuredPlan:
    backend: str
    device: str
    reason: str
    measured_ms_per_iteration: Dict[str, float] = field(default_factory=dict)
    timed_iterations: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    memory: Dict[str, dict] = field(default_factory=dict)
    calibration: Optional[dict] = None
    predicted_ms_per_iteration: Dict[str, float] = field(default_factory=dict)
    autotune_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _machine_key() -> str:
    """Identity of the CPU and libraries a calibration was measured with. Cheap on
    purpose (the full hardware probe costs ~1 s, more than planning a small case);
    the GPU identity is stored inside the entry and checked whenever CUDA is up."""
    import platform
    from importlib import metadata

    libs = []
    for dist in ("numpy", "scipy", "pyamg", "cupy-cuda12x", "cupy-cuda11x", "cupy"):
        try:
            libs.append(f"{dist}={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            pass
    return f"{platform.processor() or platform.machine()}|{os.cpu_count()}|{','.join(libs)}"


def _gpu_identity(gpu: str) -> str:
    be = get_backend(gpu)
    try:
        rt = be.xp.cuda.runtime
        name = rt.getDeviceProperties(be.device_index)["name"]
        return f"{name.decode() if isinstance(name, bytes) else name}/{rt.driverGetVersion()}"
    except Exception:
        return gpu


def _cupy_installed() -> bool:
    """Whether CuPy can be imported, without importing it or creating a CUDA context."""
    import importlib.util
    import sys

    if "cupy" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("cupy") is not None
    except (ImportError, ValueError):
        return False


def _load_cache() -> dict:
    try:
        with open(PLANNER_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def cached_calibration(precision: str = "fp64") -> Optional[dict]:
    """The cached SIMPLEC calibration for this machine, without initialising any device."""
    return _load_cache().get("unstructured", {}).get(f"{_machine_key()}|{precision}")


def _cavity_solver(n: int, backend, precision: str):
    from ..mesh.unstructured import cartesian_mesh_2d
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, max_iter=10 ** 6,
                         tol=0.0, backend=backend.name, precision=precision)
    return SimpleSolver(cartesian_mesh_2d(n, n), c, backend=backend)


def time_iterations(solver, max_steps: int, warmup: int = 1, budget_s: float = 5.0, min_steps: int = 3) -> List[float]:
    """Per-iteration wall times (s, device-synchronised) after ``warmup`` iterations.

    Stops after ``max_steps`` timed iterations, when the solver finishes, or once
    ``budget_s`` is spent (but not before ``min_steps``)."""
    be = solver.be
    for _ in range(warmup):
        if solver.finished:
            return []
        solver.step()
    ts: List[float] = []
    spent = 0.0
    while len(ts) < max_steps and not solver.finished:
        t0 = be.timer()
        solver.step()
        ts.append(be.timer() - t0)
        spent += ts[-1]
        if spent > budget_s and len(ts) >= min_steps:
            break
    return ts


def calibrate_unstructured(gpu: Optional[str] = "cupy", cpu: str = "numpy", precision: str = "fp64",
                           use_cache: bool = True) -> dict:
    """Machine constants of the SIMPLEC cost model, measured once and cached.

    ``gpu_floor_ms``: median iteration on a 16x16 mesh (launch/sync bound, a
    lower bound for any mesh); ``gpu_setup_ms``: building a second 16x16 solver once
    the device is warm (context, compiled kernels and caches already exist), a lower
    bound for setting up any case; ``cpu_us_per_cell``: mean CPU cost per cell and
    iteration on 64x64. A cached entry is reused only if it was measured with the
    same GPU (name and driver) as the one present now.
    """
    have_gpu = bool(gpu) and gpu in available_backends()
    gpu_id = _gpu_identity(gpu) if have_gpu else None
    key = f"{_machine_key()}|{precision}"
    cache = _load_cache() if use_cache else {}
    hit = cache.get("unstructured", {}).get(key)
    if hit is not None and hit.get("gpu_identity") == gpu_id:
        return hit
    out: dict = {"precision": precision, "timestamp": time.time(), "gpu_identity": gpu_id}
    s = _cavity_solver(64, get_backend(cpu), precision)
    ts = time_iterations(s, 10, warmup=1)
    out["cpu_backend"], out["cpu_us_per_cell"] = cpu, 1e6 * statistics.mean(ts) / s.nc
    if have_gpu and gpu:
        gbe = get_backend(gpu)
        try:
            s = _cavity_solver(16, gbe, precision)
            ts = time_iterations(s, 10, warmup=3)
            out["gpu_backend"], out["gpu_floor_ms"] = gpu, 1e3 * statistics.median(ts)
            del s
            t0 = time.perf_counter()
            s = _cavity_solver(16, gbe, precision)
            gbe.synchronize()
            out["gpu_setup_ms"] = 1e3 * (time.perf_counter() - t0)
            del s
        except Exception as exc:  # recorded, not cached as a valid floor
            out["gpu_error"] = str(exc)
            use_cache = False
        gbe.free_pools()
    if use_cache:
        cache.setdefault("unstructured", {})[key] = out
        try:
            os.makedirs(os.path.dirname(PLANNER_CACHE), exist_ok=True)
            with open(PLANNER_CACHE, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=1)
        except OSError:
            pass
    return out


def plan_unstructured(mesh, case, autotune: bool = True, max_steps: Optional[int] = None,
                      budget_s: float = 5.0, calibration: Optional[dict] = None, log=None):
    """Choose the backend for an unstructured case; return (UnstructuredPlan, solver).

    A ``case.backend`` other than ``"auto"`` is honoured (with a CPU fallback
    when the case does not fit in GPU memory). With ``autotune=False`` the
    decision comes from the calibrated cost model alone, without timing the case.
    With ``"auto"`` the GPU is initialised only when it could win: if the CPU
    already beats the cached GPU floor, no CUDA context is created at all.
    """
    from ..unstructured.conjugate import make_solver

    log = log or (lambda msg: None)
    t_start = time.perf_counter()
    itemsize = 4 if case.precision == "fp32" else 8
    wanted = case.backend or "auto"
    plan = UnstructuredPlan(backend="", device="", reason="")
    memory, skipped = plan.memory, plan.skipped

    def cpu_name() -> str:
        try:
            get_backend("numba")
            return "numba"  # same SIMPLEC code path as NumPy; Numba JITs the fallback AMG
        except BackendUnavailable:
            return "numpy"

    def fits(name: str) -> bool:
        be = get_backend(name)
        free, _ = be.mem_info()
        est = estimate_unstructured(mesh.n_cells, mesh.n_faces, mesh.dim, itemsize, case.energy, be.device, free)
        memory[name] = est.to_dict()
        if not est.fits:
            skipped[name] = "does not fit in memory: " + " ".join(est.advice)
        return est.fits

    def finish(name: str, reason: str, solver=None):
        be = get_backend(name)
        plan.backend, plan.device, plan.reason = name, be.device, reason
        if solver is None:
            solver = make_solver(mesh, case, backend=be)
        plan.autotune_seconds = round(time.perf_counter() - t_start, 3)
        return plan, solver

    def time_on(name: str):
        """(solver, ms per iteration); (solver, None) if it converged while timed; None if it failed."""
        be = get_backend(name)
        try:
            s = make_solver(mesh, case, backend=be)
            period = getattr(getattr(s, "flow", s), "amg_rebuild", 10)
            ts = time_iterations(s, max_steps or period, warmup=2 if be.is_gpu else 1, budget_s=budget_s)
        except Exception as exc:  # a backend that fails at runtime is simply not chosen
            skipped[name] = f"failed during autotune ({exc})"
            return None
        if not ts or s.finished:
            return s, None
        ms = 1e3 * statistics.mean(ts)
        plan.timed_iterations[name] = len(ts)
        plan.measured_ms_per_iteration[name] = round(ms, 3)
        log(f"Planner: {name} {ms:.2f} ms/iteration ({len(ts)} iterations timed)")
        return s, ms

    if wanted != "auto":  # an explicit choice: initialise only that device (raises BackendUnavailable)
        be = get_backend(wanted)
        if fits(wanted):
            return finish(wanted, "user choice")
        if not be.is_gpu:
            raise MemoryError("case does not fit in memory: " + "; ".join(skipped.values()))
        return finish(cpu_name(), f"{wanted} does not fit in memory: CPU fallback")

    cpu = cpu_name()
    gpu_installed = _cupy_installed()
    cpu_fits = fits(cpu)
    cal = calibration or cached_calibration(case.precision)
    if calibration is None and cal is not None and cal.get("gpu_identity") is None and gpu_installed:
        cal = None  # calibrated while no GPU was usable here: measure again once CUDA is up

    def gpu_ready() -> Optional[str]:
        if not gpu_installed or "cupy" not in available_backends():
            return None
        return "cupy" if fits("cupy") else None

    if not autotune:
        if cal is None:
            cal = calibrate_unstructured(gpu=gpu_ready(), cpu="numpy", precision=case.precision)
        plan.calibration = cal
        floor = cal.get("gpu_floor_ms")
        pred_cpu = cal["cpu_us_per_cell"] * 1e-3 * mesh.n_cells
        plan.predicted_ms_per_iteration = {cpu: round(pred_cpu, 3)}
        if floor is None:
            return finish(cpu, "no usable GPU")
        plan.predicted_ms_per_iteration["cupy"] = round(floor, 3)
        model = (f"cost model: CPU ~{pred_cpu:.1f} ms/it ({cal['cpu_us_per_cell']:.2f} us/cell x "
                 f"{mesh.n_cells:,} cells) vs GPU floor {floor:.1f} ms/it")
        if cpu_fits and pred_cpu < floor:
            return finish(cpu, model)
        gpu = gpu_ready()
        return finish(gpu, model) if gpu else finish(cpu, "GPU unavailable or too small")

    solvers: Dict[str, object] = {}
    timings: Dict[str, float] = {}
    if cpu_fits:
        r = time_on(cpu)
        if r is not None:
            if r[1] is None:
                return finish(cpu, "converged while being timed", r[0])
            solvers[cpu], timings[cpu] = r

    def gpu_hopeless(floor) -> bool:
        return cpu in timings and floor is not None and timings[cpu] < GPU_FLOOR_MARGIN * floor

    def gpu_cannot_pay_back(c: Optional[dict]) -> Optional[str]:
        """Why timing the GPU cannot pay off, or None. Even at its floor speed, the GPU saves at most
        (t_cpu - floor) per iteration; over the iterations this case may still run (at most max_iter
        minus those done) that must exceed its measured warm set-up time, a lower bound for the
        real cost of trying it."""
        c = c or {}
        floor, setup = c.get("gpu_floor_ms"), c.get("gpu_setup_ms")
        if cpu not in timings or floor is None or setup is None:
            return None
        remaining = max(int(case.max_iter) - int(getattr(solvers[cpu], "iteration", 0)), 0)
        best_saving = remaining * max(timings[cpu] - floor, 0.0)
        if best_saving >= setup:
            return None
        return (f"not timed: at its floor of {floor:.1f} ms/it the GPU would save at most {best_saving / 1e3:.2f} s over "
                f"the {remaining:,} iterations left, less than its measured set-up time of {setup / 1e3:.2f} s")

    floor = (cal or {}).get("gpu_floor_ms")
    no_payback = gpu_cannot_pay_back(cal)
    if gpu_hopeless(floor) or no_payback:  # decided from the cached calibration: CUDA is never initialised
        plan.calibration = cal
        skipped["cupy"] = no_payback or (f"not timed: the CPU's {timings[cpu]:.1f} ms/it is below {GPU_FLOOR_MARGIN:g} x "
                                         f"the GPU floor of {floor:.1f} ms/it measured on this machine")
        return finish(cpu, f"fastest measured: {cpu} {timings[cpu]:.2f} ms/it; GPU {skipped['cupy']}", solvers[cpu])
    gpu = gpu_ready()
    if gpu is not None:
        if calibration is None and (cal is None or cal.get("gpu_identity") != _gpu_identity(gpu)):
            cal = calibrate_unstructured(gpu=gpu, cpu="numpy", precision=case.precision)
        plan.calibration = cal
        floor = cal.get("gpu_floor_ms")
        no_payback = gpu_cannot_pay_back(cal)
        if gpu_hopeless(floor) or no_payback:
            skipped[gpu] = no_payback or (f"not timed: the CPU's {timings[cpu]:.1f} ms/it is below {GPU_FLOOR_MARGIN:g} x "
                                          f"the GPU floor of {floor:.1f} ms/it measured on this machine")
        else:
            r = time_on(gpu)
            if r is not None:
                if r[1] is None:
                    return finish(gpu, "converged while being timed", r[0])
                solvers[gpu], timings[gpu] = r
    if not timings:
        if not cpu_fits and gpu is None:
            raise MemoryError("case does not fit in memory: " + "; ".join(skipped.values()))
        raise RuntimeError("no backend could run this case: " + "; ".join(skipped.values()))
    best = min(timings, key=lambda k: timings[k])
    for name in list(solvers):
        if name != best:
            del solvers[name]
            get_backend(name).free_pools()
    if len(timings) == 1 and not skipped:
        reason = "only available backend"
    else:
        others = ", ".join(f"{k} {v:.2f}" for k, v in timings.items() if k != best)
        reason = f"fastest measured: {best} {timings[best]:.2f} ms/it" + (f" vs {others}" if others else "")
        if "cupy" in skipped and best != "cupy":
            reason += f"; GPU {skipped['cupy']}"
    return finish(best, reason, solvers[best])
