"""Warm-start study: how many solver iterations does a FlowTrace initial field save?

Two measurements, kept strictly apart:

1. **Proxy (always runs, measured here).** FlowTrace's structured MAC solver
   computes the flow; the solution goes through the *same* ``.ip`` file Fluent
   would read and is interpolated onto a *different* unstructured mesh, where
   FlowTrace's pressure-based SIMPLEC solver (the algorithm family of
   Fluent's default solver) is run twice: from rest (cold) and from the
   interpolated field (warm). This measures the benefit of the warm start for
   a Fluent-like solver, not for Fluent itself.

2. **Ansys Fluent (only with a licensed installation).** For every case a
   folder with the mesh, the ``.ip`` file, ``cold.jou``/``warm.jou`` journals
   and ``case.json`` is written. ``run_fluent_study`` drives Fluent through
   PyFluent (cold vs warm, same settings) and counts iterations; without
   PyFluent/Fluent the result table keeps clearly labelled placeholders.
   Users without PyFluent can run the journals and feed the transcripts to
   :func:`parse_transcript`.

Fair convergence criterion: Fluent (and FlowTrace) normalise the continuity
residual by its largest value in the first five iterations. A warm start
begins with a small imbalance, so the *normalised* residual looks worse than
it is, and absolute residual levels differ by round-off between two runs.
The study therefore measures the answer itself: both runs are counted until
the velocity (and temperature) field is within ``FIELD_TOL`` (relative, max
norm) of the converged solution, taken from the cold run converged to 1e-7.
The usual residual-tolerance counts are reported as well.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

CASES: Dict[str, dict] = {
    "cavity": {"title": "Lid-driven cavity, Re = 400", "preset": "cavity", "params": {"re": 400, "n": 64},
               "target": {"kind": "tri", "n": 40}},
    "channel": {"title": "Developing channel flow, Re = 100", "preset": "channel",
                "params": {"re": 100, "nx": 128, "ny": 16, "length": 8.0},
                "target": {"kind": "structured", "scale": 0.75}},
    "step": {"title": "Backward-facing step, Re = 200 (laminar)", "preset": "step",
             "params": {"re": 200, "n_per_h": 16}, "target": {"kind": "structured", "scale": 0.75}},
    "cylinder": {"title": "Cylinder in a channel, Re = 40 (steady)", "preset": "cylinder",
                 "params": {"re": 40, "n_per_d": 12},
                 "overrides": {"steady_tol": 1e-6, "t_end": None, "perturb": 0.0, "max_steps": 200000},
                 "target": {"kind": "structured", "scale": 0.75}},
    "heated": {"title": "Natural convection, Ra = 1e5 (heated cavity)", "preset": "natural_convection",
               "params": {"ra": 1e5, "n": 64}, "target": {"kind": "tri", "n": 40}},
}


@dataclass
class ProxyResult:
    case: str
    title: str
    cells_source: int
    cells_target: int
    flowtrace_seconds: float  # MAC solve + export
    cold_iterations: int  # until the field is within FIELD_TOL of the converged solution
    warm_iterations: int  # same criterion, starting from the interpolated .ip field
    cold_iterations_tol: Optional[int]  # until the scaled residuals reach the tolerance (what a user sees)
    warm_iterations_normalised: Optional[int]  # same, warm run (continuity normalised by its first iterations)
    cold_seconds: float  # whole cold loop, to the tight tolerance, monitoring included
    warm_seconds: float  # whole warm loop, monitoring included
    cold_converged: bool
    warm_converged: bool
    max_velocity_difference: float  # |u_warm - u_ref|_inf / |u_ref|_inf at the end of the warm run
    stride: int = 1  # field snapshots of the cold run every `stride` iterations (count resolution)
    package: Dict[str, str] = field(default_factory=dict)
    # wall time on equal terms: solver steps only (field monitoring excluded on both sides), until the
    # field is within FIELD_TOL of the converged one; the interpolation of the .ip file separately
    cold_solver_seconds: float = float("nan")
    warm_solver_seconds: float = float("nan")
    interpolation_seconds: float = float("nan")

    @property
    def speedup(self) -> float:
        return self.cold_iterations / max(self.warm_iterations, 1)

    @property
    def wallclock_ratio(self) -> float:
        """Cold solver time / (pre-solve + interpolation + warm solver time) to the same field;
        > 1 only if the whole warm-start path is faster in elapsed time."""
        total = self.flowtrace_seconds + self.interpolation_seconds + self.warm_solver_seconds
        return self.cold_solver_seconds / total if total > 0 else float("nan")


FIELD_TOL = 1e-4  # relative max-norm distance to the converged field that counts as "converged"


def _structured_case(spec):
    from ..structured.case import make_preset

    c = make_preset(spec["preset"], **spec["params"])
    for k, v in (spec.get("overrides") or {}).items():
        setattr(c, k, v)
    return c


def _target_mesh(case, spec):
    from ..fluent.convert import structured_mesh
    from ..mesh.unstructured import cartesian_mesh_2d
    from ..structured.solver import MACSolver
    from ..hardware.backend import get_backend

    t = spec["target"]
    if t["kind"] == "tri":
        types = tuple({"wall": "wall", "inlet": "velocity-inlet", "outlet": "pressure-outlet",
                       "symmetry": "symmetry"}[getattr(case, s).kind] for s in ("left", "right", "bottom", "top"))
        n = t["n"]
        ny = max(4, int(round(n * case.ly / case.lx)))
        return cartesian_mesh_2d(n, ny, case.lx, case.ly, types=types, triangulate=True)
    d = case.to_dict()
    d["nx"] = int(round(case.nx * t["scale"]))
    d["ny"] = int(round(case.ny * t["scale"]))
    from ..structured.case import StructuredCase

    c2 = StructuredCase.from_dict(d)
    solid = MACSolver(c2, backend=get_backend("numpy")).solid_h
    return structured_mesh(c2, solid)


def _ucase(case, max_iter, tol):
    from ..fluent.convert import structured_boundary_specs
    from ..unstructured.case import UnstructuredCase

    return UnstructuredCase(name=case.name, boundaries=structured_boundary_specs(case), density=1.0,
                            viscosity=case.nu, energy=case.energy, conductivity=case.kappa, cp=1.0,
                            buoyancy=list(case.buoyancy) + [0.0], t_ref=case.t_ref, init_T=case.init_T,
                            max_iter=max_iter, tol=tol)


def run_proxy_case(name: str, out_dir: str, backend: str = "auto", tol: float = 1e-5,
                   max_iter: int = 20000) -> ProxyResult:
    from ..fluent.interpolate import interpolate_ip_file
    from ..hardware.backend import get_backend
    from ..runtime.engine import EventHub, Job
    from ..unstructured.simple import SimpleSolver

    spec = CASES[name]
    case = _structured_case(spec)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    job_spec = {"kind": "structured", "preset": spec["preset"], "params": spec["params"],
                "overrides": dict(spec.get("overrides") or {}, backend=backend), "fmg": True}
    job = Job(job_spec, EventHub(), os.path.join(out_dir, "_runs"))
    job.start()
    job.join()
    if job.error:
        raise RuntimeError(f"{name}: FlowTrace run failed: {job.error}")
    pkg = job.export_package(out_dir)
    ft_seconds = time.perf_counter() - t0

    mesh = _target_mesh(case, spec)
    be = get_backend(backend if backend != "auto" else "numpy")

    def state(solver):
        f = solver.fields()
        parts = [f["velocity-magnitude"]] + ([f["temperature"]] if "temperature" in f else [])
        return [p.astype(np.float32) for p in parts]

    def distance(a, ref):
        return max(float(np.abs(x - r).max() / max(float(np.abs(r).max()), 1e-30)) for x, r in zip(a, ref))

    # cold: from rest to a tight tolerance; that run's final field is the reference answer, and its
    # snapshots tell when it first came within FIELD_TOL of it
    stride = max(1, mesh.n_cells // 2000)
    cold = SimpleSolver(mesh, _ucase(case, max_iter, min(tol, 1e-7)), backend=be)
    snaps, n_cold_tol = [], None
    cold_solver = 0.0  # time inside cold.step() only
    t1 = time.perf_counter()
    while not cold.finished:
        ts = time.perf_counter()
        cold.step()
        cold_solver += time.perf_counter() - ts
        if n_cold_tol is None and cold.iteration > 5 and max(cold.residuals()["scaled"].values()) < tol:
            n_cold_tol = cold.iteration
        if cold.iteration % stride == 0:
            snaps.append((cold.iteration, cold_solver, state(cold)))
    cold_s = time.perf_counter() - t1
    ref = state(cold)
    n_cold, cold_field_s = next(((it, sec) for it, sec, s in snaps if distance(s, ref) <= FIELD_TOL),
                                (cold.iteration, cold_solver))
    del snaps
    # warm: interpolate the .ip file onto the target cells, iterate until within FIELD_TOL
    warm = SimpleSolver(mesh, _ucase(case, max_iter, min(tol, 1e-7)), backend=be)
    ti = time.perf_counter()
    warm.initialize(interpolate_ip_file(pkg["ip"], warm.cell_centres()))
    interp_s = time.perf_counter() - ti
    t2 = time.perf_counter()
    warm_solver = 0.0
    n_norm, d = None, float("inf")
    while warm.iteration < max_iter:
        ts = time.perf_counter()
        warm.step()
        warm_solver += time.perf_counter() - ts
        if n_norm is None and warm.iteration > 5 and max(warm.residuals()["scaled"].values()) < tol:
            n_norm = warm.iteration
        d = distance(state(warm), ref)
        if d <= FIELD_TOL:
            break
    warm_s = time.perf_counter() - t2
    res = ProxyResult(name, spec["title"], case.nx * case.ny, mesh.n_cells, ft_seconds, n_cold, warm.iteration,
                      n_cold_tol, n_norm, cold_s, warm_s, cold.converged, d <= FIELD_TOL, d, stride,
                      {k: os.path.relpath(v, out_dir) for k, v in pkg.items()},
                      cold_solver_seconds=cold_field_s, warm_solver_seconds=warm_solver,
                      interpolation_seconds=interp_s)
    write_fluent_case_files(out_dir, case, mesh_file=os.path.basename(pkg["msh"]),
                            ip_file=os.path.basename(pkg["ip"]), tol=tol)
    return res


# --------------------------------------------------------------------------
# Fluent side: journals, case description, PyFluent runner, transcript parser
# --------------------------------------------------------------------------

def write_fluent_case_files(out_dir: str, case, mesh_file: str, ip_file: str, tol: float = 1e-5,
                            iterations: int = 5000) -> Dict[str, str]:
    """cold.jou / warm.jou (identical except for the interpolation) + case.json."""
    from ..fluent.convert import structured_boundary_specs

    bnd = structured_boundary_specs(case)
    desc = {"name": case.name, "dimension": 2, "mesh": mesh_file, "ip": ip_file, "density": 1.0,
            "viscosity": case.nu, "energy": bool(case.energy), "conductivity": case.kappa, "cp": 1.0,
            "boussinesq": {"rho_beta_g": list(case.buoyancy), "t_ref": case.t_ref} if case.energy else None,
            "boundaries": bnd, "tolerance": tol, "max_iterations": iterations}
    files = {}
    p = os.path.join(out_dir, "case.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(desc, fh, indent=1)
    files["case"] = p
    n_eq = 3 + (1 if case.energy else 0)
    crit = " ".join([f"{tol:g}"] * n_eq)
    for mode in ("cold", "warm"):
        lines = [
            f"; FlowTrace warm-start study ({mode} start) - {case.name}",
            "; Fluent and Ansys are trademarks of ANSYS, Inc.; FlowTrace is not affiliated with ANSYS, Inc.",
            f'/file/start-transcript "{mode}.trn"',
            f'/file/read-case "{mesh_file}"',
            "/define/models/viscous/laminar yes",
        ]
        if case.energy:
            lines += ["/define/models/energy yes", "", "", "", ""]
        lines += [
            "; >>> set materials and boundary conditions as in case.json (or use study_pyfluent.py) <<<",
            f"/solve/monitors/residual/convergence-criteria {crit}",
            "/solve/initialize/hybrid-initialize",
        ]
        if mode == "warm":
            lines += [f'/file/interpolate/read-data "{ip_file}"', "", ""]
        lines += [f"/solve/iterate {iterations}", "/file/stop-transcript",
                  f'/file/write-case-data "{mode}_result.cas.h5"', "/exit yes"]
        p = os.path.join(out_dir, f"{mode}.jou")
        with open(p, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")
        files[mode] = p
    return files


_ROW = re.compile(r"^\s*(\d+)\s+((?:[-+]?\d\.\d+e[-+]\d+\s+){2,})")


def parse_transcript(text: str) -> dict:
    """Iterations and convergence from a Fluent transcript (.trn).

    Recognises the residual table rows (``iter continuity x-velocity ...``)
    and the ``solution is converged`` message.
    """
    iters: List[int] = []
    header: List[str] = []
    rows: List[List[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("iter") and "continuity" in s:
            header = s.split()
            continue
        m = _ROW.match(line)
        if m and header:
            iters.append(int(m.group(1)))
            rows.append([float(x) for x in m.group(2).split()])
    converged = bool(re.search(r"solution is converged", text))
    out: dict = {"iterations": iters[-1] if iters else None, "converged": converged, "rows": len(rows)}
    if header and rows:
        names = header[1:1 + len(rows[-1])]
        out["final_residuals"] = dict(zip(names, rows[-1]))
        out["first_iteration"] = iters[0]
    return out


@dataclass
class FluentResult:
    case: str
    status: str  # measured | not-run
    cold_iterations: Optional[int] = None
    warm_iterations: Optional[int] = None
    message: str = ""

    @property
    def speedup(self) -> Optional[float]:
        if self.cold_iterations and self.warm_iterations:
            return self.cold_iterations / self.warm_iterations
        return None


def run_fluent_study(case_dirs: Dict[str, str], processors: int = 4) -> List[FluentResult]:
    """Run cold and warm starts in Ansys Fluent through PyFluent (needs a licence).

    Never raises: every case that cannot be run is reported as ``not-run``
    with the reason, so a results table never contains invented numbers.
    """
    from .bridge import pyfluent_status

    st = pyfluent_status()
    if not st.available:
        return [FluentResult(c, "not-run", message=st.message) for c in case_dirs]
    out = []
    for name, d in case_dirs.items():
        try:
            with open(os.path.join(d, "case.json"), encoding="utf-8") as fh:
                desc = json.load(fh)
            counts = {}
            for mode in ("cold", "warm"):
                counts[mode] = _pyfluent_run(d, desc, mode, processors)
            out.append(FluentResult(name, "measured", counts["cold"], counts["warm"]))
        except Exception as exc:  # licence, launch or API problems
            out.append(FluentResult(name, "not-run", message=f"{type(exc).__name__}: {exc}"))
    return out


def _pyfluent_run(d: str, desc: dict, mode: str, processors: int) -> int:
    import ansys.fluent.core as pyfluent  # type: ignore

    solver = pyfluent.launch_fluent(mode="solver", dimension=desc["dimension"], precision="double",
                                    processor_count=processors, cwd=d)
    trn = os.path.join(d, f"{mode}_pyfluent.trn")
    try:
        s = solver.settings
        try:
            s.file.start_transcript(file_name=trn)
        except Exception:
            solver.tui.file.start_transcript(trn)
        s.file.read_case(file_name=os.path.join(d, desc["mesh"]))
        s.setup.models.viscous.model = "laminar"
        if desc.get("energy"):
            s.setup.models.energy.enabled = True
        air = s.setup.materials.fluid["air"]
        air.density.value = desc["density"]
        air.viscosity.value = desc["viscosity"]
        bc = s.setup.boundary_conditions
        for zname, spec in desc["boundaries"].items():
            kind = spec.get("kind")
            if kind == "inlet":
                bc.velocity_inlet[zname].momentum.velocity_magnitude.value = float(spec.get("normal_speed") or 0.0)
            elif kind == "outlet":
                bc.pressure_outlet[zname].momentum.gauge_pressure.value = float(spec.get("pressure", 0.0))
            elif kind == "wall" and spec.get("velocity") and any(spec["velocity"]):
                w = bc.wall[zname].momentum
                w.wall_motion = "Moving Wall"
                v = spec["velocity"]
                spd = float(np.linalg.norm(v))
                w.speed = spd
                w.direction.value = [float(x) / spd for x in v] + [0.0] * (3 - len(v))
        for eq in s.solution.monitor.residual.equations.keys():
            s.solution.monitor.residual.equations[eq].absolute_criteria = desc["tolerance"]
        s.solution.initialization.hybrid_initialize()
        if mode == "warm":
            s.file.interpolate.read_data(file_name=os.path.join(d, desc["ip"]))
        s.solution.run_calculation.iterate(iter_count=int(desc["max_iterations"]))
        n = None
        try:
            iters, _data = solver.monitors.get_monitor_set_data(monitor_set_name="residual")
            n = int(iters[-1])
        except Exception:
            pass
        try:
            s.file.stop_transcript()
        except Exception:
            pass
        if n is None:
            with open(trn, encoding="utf-8", errors="ignore") as fh:
                n = parse_transcript(fh.read())["iterations"]
        if n is None:
            raise RuntimeError("could not determine the iteration count")
        return n
    finally:
        solver.exit()


# --------------------------------------------------------------------------
# full study
# --------------------------------------------------------------------------

def run_study(out_dir: str = "benchmarks/fluent_warmstart", cases: Optional[List[str]] = None,
              backend: str = "auto", fluent: bool = True, tol: float = 1e-5, verbose: bool = True) -> dict:
    cases = cases or list(CASES)
    proxy, dirs = [], {}
    for name in cases:
        d = os.path.join(out_dir, name)
        if verbose:
            print(f"[warm-start] {name}: FlowTrace solve, export, SIMPLEC cold vs warm ...", flush=True)
        r = run_proxy_case(name, d, backend=backend, tol=tol)
        proxy.append(r)
        dirs[name] = d
        if verbose:
            print(f"[warm-start] {name}: cold {r.cold_iterations} it, warm {r.warm_iterations} it "
                  f"(x{r.speedup:.1f}); FlowTrace {r.flowtrace_seconds:.1f} s", flush=True)
    fl = run_fluent_study(dirs) if fluent else [FluentResult(c, "not-run", message="skipped (--no-fluent)")
                                                 for c in cases]
    data = {"proxy": [dict(asdict(r), speedup=r.speedup, wallclock_ratio=r.wallclock_ratio) for r in proxy],
            "fluent": [dict(asdict(r), speedup=r.speedup) for r in fl],
            "tolerance": tol}
    from ..validation.benchmarks import _clean, machine_info

    data["machine"] = machine_info()
    data = _clean(data)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(data))
    return data


def render_markdown(data: dict) -> str:
    L = ["### Proxy: FlowTrace SIMPLEC, cold start vs warm start from the FlowTrace `.ip` file", "",
         f"Measured. Both runs are counted until the velocity (and temperature) field is within {FIELD_TOL:g} "
         "(relative, max norm) of the converged solution. The last two columns count iterations until the "
         "scaled residuals reach the tolerance, which is what a solver's convergence monitor shows; the warm "
         "count there is distorted by the continuity normalisation (see the text).", "",
         "| Case | Target cells | Cold | Warm | Saved | FlowTrace pre-solve | Cold to tol. | Warm to tol. |",
         "|---|---|---|---|---|---|---|---|"]
    timing = ["", "Wall time on equal terms (solver steps only, until the same field tolerance; "
              "the warm path also pays the FlowTrace pre-solve and the interpolation). A ratio above 1 "
              "would mean the whole warm-start path is faster. The proxy's target solver is cheap, so this "
              "says nothing about a large Fluent case; it keeps iteration savings and elapsed-time savings apart.", "",
              "| Case | Cold solver time | Pre-solve + interpolation + warm solver | Wall-time ratio |",
              "|---|---|---|---|"]
    for r in data["proxy"]:
        def n(v):
            return f"{v:,}" if v else "-"
        warm = f"{r['warm_iterations']:,}" + ("" if r.get("warm_converged", True) else " (not reached)")
        L.append(f"| {r['title']} | {r['cells_target']:,} | {r['cold_iterations']:,} | {warm} | "
                 f"x{r['speedup']:.1f} | {r['flowtrace_seconds']:.1f} s | {n(r.get('cold_iterations_tol'))} | "
                 f"{n(r.get('warm_iterations_normalised'))} |")
        if r.get("cold_solver_seconds") is not None and r.get("wallclock_ratio") is not None:
            path = r["flowtrace_seconds"] + r["interpolation_seconds"] + r["warm_solver_seconds"]
            timing.append(f"| {r['title']} | {r['cold_solver_seconds']:.2f} s | {path:.2f} s "
                          f"({r['flowtrace_seconds']:.2f} + {r['interpolation_seconds']:.2f} + "
                          f"{r['warm_solver_seconds']:.2f}) | x{r['wallclock_ratio']:.2f} |")
    if len(timing) > 5:  # at least one measured row below the header
        L += timing
    L += ["", "### Ansys Fluent: cold start vs `.ip` warm start", ""]
    if all(r["status"] != "measured" for r in data["fluent"]):
        reason = data["fluent"][0]["message"] if data["fluent"] else ""
        L.append(f"**Not measured** - no licensed Fluent available when this table was generated ({reason}). "
                 "Run `python -m flowtrace warmstart-study` on a machine with Fluent + PyFluent, or run "
                 "`cold.jou`/`warm.jou` in each case folder and `python -m flowtrace parse-transcript cold.trn warm.trn`.")
        L.append("")
    L += ["| Case | Status | Cold iterations | Warm iterations | Saved |", "|---|---|---|---|---|"]
    for r in data["fluent"]:
        if r["status"] == "measured":
            L.append(f"| {r['case']} | measured | {r['cold_iterations']} | {r['warm_iterations']} | x{r['speedup']:.1f} |")
        else:
            L.append(f"| {r['case']} | not run | _placeholder_ | _placeholder_ | - |")
    m = data.get("machine", {})
    L += ["", f"_Generated {m.get('date', '')} on {m.get('cpu', '?')}, residual tolerance {data.get('tolerance')}._", ""]
    return "\n".join(L)
