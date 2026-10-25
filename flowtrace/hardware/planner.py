"""Execution planner: decide where each stage of a solve runs, by measuring.

The planner does not assume "GPU is always faster". For every backend that is
available and whose memory budget fits, it builds the real solver for the real
case and times a few real time steps (after JIT/plan warm-up). The fastest
measured configuration wins, and the plan records *why* for every stage, so
the UI can show the reasoning.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .backend import BackendUnavailable, available_backends, get_backend
from .memory import estimate_structured, fmt_bytes
from .probe import SystemProfile, probe_system


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

    def to_dict(self) -> dict:
        return asdict(self)


def _candidates(case_backend: str) -> List[str]:
    avail = available_backends()
    if case_backend and case_backend != "auto":
        if case_backend not in avail:
            raise BackendUnavailable(f"backend {case_backend!r} is not available here (have {avail})")
        return [case_backend]
    return avail


def _time_solver(solver, steps: int) -> float:
    be = solver.be
    for _ in range(2):
        solver.step()
    t0 = be.timer()
    for _ in range(steps):
        solver.step()
    return (be.timer() - t0) / steps


def plan_structured(case, autotune: bool = True, steps: Optional[int] = None,
                    profile: Optional[SystemProfile] = None, keep_solver: bool = True):
    """Return (plan, solver). ``solver`` is the already-built winning solver
    (it has advanced ``2 + steps`` time steps) or None when not autotuning."""
    from ..structured.sequencing import fmg_factors
    from ..structured.solver import MACSolver, resolve_dtype

    profile = profile or probe_system()
    cands = _candidates(case.backend)
    has_obs = bool(case.obstacles)
    cells = case.nx * case.ny
    if steps is None:
        steps = 6 if cells <= 256 ** 2 else (3 if cells <= 1024 ** 2 else 1)

    memory, timings, solvers, notes = {}, {}, {}, []
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
        solver = MACSolver(case, backend=be)
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
        tc = ("FP64 Tensor-Core GEMMs (cuBLAS DMMA) in the dense transforms" if gpu.is_datacenter_fp64 and "eig" in desc["pressure_detail"]
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
