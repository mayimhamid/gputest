"""Reproducible benchmark suite: every number FlowTrace publishes comes from here.

Each benchmark is a function returning a :class:`BenchResult` with named
metrics (value, reference, tolerance, pass/fail) and optional plot series.
``run_benchmarks`` executes a selection, records the machine it ran on and
writes ``results.json``; :mod:`flowtrace.validation.report` turns that into
HTML/PDF/Markdown.

Tiers: ``quick`` (minutes, CI-sized grids) and ``full`` (the grids quoted in
the README; Re=1000 and the turbulence cases take longer).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass
class Metric:
    value: float
    reference: Optional[float] = None
    unit: str = ""
    tolerance: Optional[str] = None  # human-readable acceptance rule
    passed: Optional[bool] = None
    source: str = ""  # reference source (paper, exact, analytic)


@dataclass
class Series:
    x: list
    y: list
    label: str = ""
    style: str = "line"  # line | marker | ref


@dataclass
class Plot:
    title: str
    xlabel: str
    ylabel: str
    series: List[Series] = field(default_factory=list)
    logx: bool = False
    logy: bool = False


@dataclass
class BenchResult:
    id: str
    title: str
    category: str
    metrics: Dict[str, Metric] = field(default_factory=dict)
    plots: List[Plot] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(m.passed is not False for m in self.metrics.values())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return _clean(d)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _m(value, reference=None, unit="", tol=None, ok=None, source="") -> Metric:
    return Metric(float(value) if value is not None else float("nan"),
                  None if reference is None else float(reference), unit, tol,
                  None if ok is None else bool(ok), source)


@dataclass
class Benchmark:
    id: str
    title: str
    category: str
    tiers: tuple
    fn: Callable


REGISTRY: Dict[str, Benchmark] = {}


def benchmark(id: str, title: str, category: str, tiers=("quick", "full")):
    def deco(fn):
        REGISTRY[id] = Benchmark(id, title, category, tuple(tiers), fn)
        return fn
    return deco


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

GHIA = "Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982)"
DVD_SRC = "de Vahl Davis, Int. J. Numer. Meth. Fluids 3 (1983)"


def _be(ctx):
    """Benchmarks run in FP64 on the CPU unless a backend is named: the published numbers
    must not depend on whether a consumer GPU (FP32 by default) is present. GPU results
    come from ``flowtrace gpu-report``."""
    from ..hardware.backend import get_backend

    name = ctx.get("backend", "auto")
    if name in (None, "auto", "cpu"):
        name = "numba" if _has_numba() else "numpy"
    return get_backend(name)


def _run_mac(solver, k=10):
    while not solver.finished:
        solver.step()
        if solver.step_count % k == 0:
            solver.residuals()
    return solver


def _run_simple(solver):
    while not solver.finished:
        solver.step()
    return solver


def _ghia_series(re, y, u, x, v):
    from .reference import ghia

    (gy, gu), (gx, gv) = ghia(re)
    return [
        Plot(f"Cavity Re={re:g}: u(x=0.5, y)", "y", "u", [
            Series(list(y), list(u), "FlowTrace"), Series(list(gy), list(gu), "Ghia et al. 1982", "ref")]),
        Plot(f"Cavity Re={re:g}: v(x, y=0.5)", "x", "v", [
            Series(list(x), list(v), "FlowTrace"), Series(list(gx), list(gv), "Ghia et al. 1982", "ref")]),
    ]


def _unstructured_centerlines(s, n=99):
    from scipy.interpolate import griddata

    xc, f = s.cell_centres()[:, :2], s.fields()
    t = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    u = griddata(xc, f["x-velocity"], np.column_stack([np.full_like(t, 0.5), t]))
    v = griddata(xc, f["y-velocity"], np.column_stack([t, np.full_like(t, 0.5)]))
    return t, u, t, v


def _cavity_simplec(n, re, triangulate=False, jitter=0.0, ctx=None, max_iter=20000, tol=1e-6):
    from ..mesh.unstructured import cartesian_mesh_2d
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    m = cartesian_mesh_2d(n, n, triangulate=triangulate, jitter=jitter)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=1.0 / re,
                         max_iter=max_iter, tol=tol)
    return _run_simple(SimpleSolver(m, c, backend=_be(ctx or {})))


# --------------------------------------------------------------------------
# verification: exact solutions and manufactured solutions
# --------------------------------------------------------------------------


@benchmark("taylor_green_order", "Taylor-Green vortex: error vs exact solution", "verification")
def b_taylor_green(ctx):
    from ..structured.case import taylor_green
    from ..structured.solver import MACSolver
    from .checks import validate_structured

    grids = (16, 32, 64, 128)
    errs = []
    for n in grids:
        s = _run_mac(MACSolver(taylor_green(n=n, re=10.0, t_end=0.5, cfl=0.25), backend=_be(ctx)))
        errs.append(validate_structured(s)["checks"][0]["value"])
    rates = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    r = BenchResult("taylor_green_order", "", "verification")
    for n, e in zip(grids, errs):
        r.metrics[f"Linf error n={n}"] = _m(e, None, "|u-u_exact|")
    r.metrics["observed order (finest pair)"] = _m(rates[-1], 2.0, "", "> 1.8", rates[-1] > 1.8, "exact solution")
    r.plots.append(Plot("Taylor-Green: L-inf error", "grid spacing h", "error", [
        Series([2 * math.pi / n for n in grids], errs, "FlowTrace", "marker"),
        Series([2 * math.pi / n for n in grids], [errs[0] * (grids[0] / n) ** 2 for n in grids], "2nd order", "ref")],
        logx=True, logy=True))
    r.details = {"grids": grids, "errors": errs, "orders": rates}
    return r


def _mms_result(id_, res, min_order, min_order_p=None):
    r = BenchResult(id_, "", "verification")
    for q, orders in res.orders.items():
        lim = min_order_p if (q == "p" and min_order_p is not None) else min_order
        r.metrics[f"observed order {q} (L2)"] = _m(orders[-1], 2.0, "", f"> {lim}", orders[-1] > lim,
                                                   "manufactured solution")
    for q, e in res.errors.items():
        r.metrics[f"L2 error {q}, finest grid"] = _m(e[-1], None, "")
    r.plots.append(Plot(res.solver + ": L2 error", "h", "L2 error", [
        Series(res.h, e, q, "marker") for q, e in res.errors.items()] + [
        Series(res.h, [res.errors["u"][0] * (h / res.h[0]) ** 2 for h in res.h], "2nd order", "ref")],
        logx=True, logy=True))
    r.details = res.to_dict()
    return r


@benchmark("mms_mac", "MMS: MAC solver observed order (walls, energy)", "verification")
def b_mms_mac(ctx):
    from .mms import run_mac

    grids = (16, 32, 64) if ctx["tier"] == "quick" else (16, 32, 64, 128)
    res = run_mac(grids, backend=ctx.get("cpu_backend", "numpy"))
    return _mms_result("mms_mac", res, 1.8)


@benchmark("mms_simplec_quads", "MMS: SIMPLEC observed order on warped (non-orthogonal) quads", "verification")
def b_mms_simplec_quads(ctx):
    from .mms import run_simplec

    grids = (12, 24, 48) if ctx["tier"] == "quick" else (16, 32, 64, 128)
    res = run_simplec(grids, warp=0.05, backend="numpy")
    return _mms_result("mms_simplec_quads", res, 1.8)


@benchmark("mms_simplec_tri", "MMS: SIMPLEC observed order on warped triangles", "verification")
def b_mms_simplec_tri(ctx):
    from .mms import run_simplec

    grids = (12, 24, 48) if ctx["tier"] == "quick" else (12, 24, 48, 96)
    res = run_simplec(grids, triangulate=True, warp=0.05, backend="numpy")
    return _mms_result("mms_simplec_tri", res, 1.8, min_order_p=1.0)


@benchmark("projection_divergence", "Discrete continuity of the projection", "verification")
def b_projection(ctx):
    from ..structured.case import cylinder_wake, lid_driven_cavity
    from ..structured.solver import MACSolver

    be = _be(ctx)
    s1 = MACSolver(lid_driven_cavity(re=100, n=64, max_steps=200, precision="fp64"), backend=be)
    _run_mac(s1)
    s2 = MACSolver(cylinder_wake(n_per_d=8, t_end=2.0, precision="fp64"), backend=be)
    _run_mac(s2)
    r = BenchResult("projection_divergence", "", "verification")
    r.metrics["max|div u| fast separable solver"] = _m(s1.max_divergence(), 0.0, "1/s", "< 1e-10",
                                                       s1.max_divergence() < 1e-10)
    r.metrics["max|div u| multigrid-PCG (obstacle)"] = _m(s2.max_divergence(), 0.0, "1/s", "< 1e-8",
                                                          s2.max_divergence() < 1e-8)
    r.details = {"fast": s1.poisson_detail, "mg": s2.poisson_detail}
    return r


# --------------------------------------------------------------------------
# validation: laminar reference cases
# --------------------------------------------------------------------------


@benchmark("cavity_re100_mac", "Lid-driven cavity Re=100, MAC 64x64", "laminar")
def b_cavity_mac(ctx):
    from ..structured.case import lid_driven_cavity
    from ..structured.solver import MACSolver
    from .reference import compare_ghia

    s = _run_mac(MACSolver(lid_driven_cavity(re=100, n=64), backend=_be(ctx)))
    (y, u), (x, v) = s.centerline()
    e = compare_ghia(100, y, u, x, v)
    r = BenchResult("cavity_re100_mac", "", "laminar")
    r.metrics["max |du| centreline"] = _m(e["u_max_err"], 0.0, "", "< 0.01", e["u_max_err"] < 0.01, GHIA)
    r.metrics["max |dv| centreline"] = _m(e["v_max_err"], 0.0, "", "< 0.015", e["v_max_err"] < 0.015, GHIA)
    r.plots += _ghia_series(100, y, u, x, v)
    r.details = {"steps": s.step_count, "converged": s.converged, "backend": s.be.name}
    return r


@benchmark("cavity_re100_simplec_quads", "Lid-driven cavity Re=100, SIMPLEC 48x48 quads", "laminar")
def b_cavity_simplec_quads(ctx):
    from .reference import compare_ghia

    s = _cavity_simplec(48, 100, ctx=ctx)
    y, u, x, v = _unstructured_centerlines(s)
    e = compare_ghia(100, y, u, x, v)
    r = BenchResult("cavity_re100_simplec_quads", "", "laminar")
    r.metrics["max |du| centreline"] = _m(e["u_max_err"], 0.0, "", "< 0.01", e["u_max_err"] < 0.01, GHIA)
    r.metrics["max |dv| centreline"] = _m(e["v_max_err"], 0.0, "", "< 0.015", e["v_max_err"] < 0.015, GHIA)
    r.plots += _ghia_series(100, y, u, x, v)
    r.details = {"iterations": s.iteration, "converged": s.converged}
    return r


@benchmark("cavity_re100_simplec_tri", "Lid-driven cavity Re=100, SIMPLEC 48x48 skewed triangles", "laminar")
def b_cavity_simplec_tri(ctx):
    from .reference import compare_ghia

    s = _cavity_simplec(48, 100, triangulate=True, jitter=0.2, ctx=ctx)
    y, u, x, v = _unstructured_centerlines(s)
    e = compare_ghia(100, y, u, x, v)
    r = BenchResult("cavity_re100_simplec_tri", "", "laminar")
    r.metrics["max |du| centreline"] = _m(e["u_max_err"], 0.0, "", "< 0.015", e["u_max_err"] < 0.015, GHIA)
    r.metrics["max |dv| centreline"] = _m(e["v_max_err"], 0.0, "", "< 0.02", e["v_max_err"] < 0.02, GHIA)
    r.metrics["max non-orthogonality"] = _m(s.fv.non_orthogonality_deg, None, "deg")
    r.plots += _ghia_series(100, y, u, x, v)
    r.details = {"iterations": s.iteration, "converged": s.converged}
    return r


@benchmark("cavity_re1000_simplec", "Lid-driven cavity Re=1000, SIMPLEC 128x128 quads", "laminar", tiers=("full",))
def b_cavity_re1000(ctx):
    from .reference import compare_ghia

    t0 = time.perf_counter()
    s = _cavity_simplec(128, 1000, ctx=ctx, max_iter=20000, tol=1e-6)
    y, u, x, v = _unstructured_centerlines(s, 127)
    e = compare_ghia(1000, y, u, x, v)
    r = BenchResult("cavity_re1000_simplec", "", "laminar")
    r.metrics["max |du| centreline"] = _m(e["u_max_err"], 0.0, "", "< 0.02", e["u_max_err"] < 0.02, GHIA)
    r.metrics["max |dv| centreline"] = _m(e["v_max_err"], 0.0, "", "< 0.02", e["v_max_err"] < 0.02, GHIA)
    r.metrics["iterations to 1e-6"] = _m(s.iteration, None, "")
    r.metrics["wall time"] = _m(time.perf_counter() - t0, None, "s")
    r.plots += _ghia_series(1000, y, u, x, v)
    r.details = {"iterations": s.iteration, "converged": s.converged, "backend": s.be.name}
    return r


def _midline_maxima(solver):
    """de Vahl Davis velocity maxima: max u on x=0.5, max v on y=0.5 (MAC solver)."""
    from ..structured.solver import G

    nx, ny = solver.nx, solver.ny
    U = solver.be.to_numpy(solver.U)[G:G + nx + 1, G:G + ny]
    V = solver.be.to_numpy(solver.V)[G:G + nx, G:G + ny + 1]
    u_mid = U[nx // 2] if nx % 2 == 0 else 0.5 * (U[nx // 2] + U[nx // 2 + 1])
    v_mid = V[:, ny // 2] if ny % 2 == 0 else 0.5 * (V[:, ny // 2] + V[:, ny // 2 + 1])
    return float(np.max(u_mid)), float(np.max(v_mid))


@benchmark("natural_convection_ra1e4_mac", "Natural convection Ra=1e4, MAC 64x64", "laminar")
def b_nc_1e4(ctx):
    from ..structured.case import natural_convection
    from ..structured.solver import MACSolver
    from .reference import DVD

    s = _run_mac(MACSolver(natural_convection(ra=1e4, n=64), backend=_be(ctx)))
    nu = s.nusselt("left")
    umax, vmax = _midline_maxima(s)
    ref = DVD[1e4]
    r = BenchResult("natural_convection_ra1e4_mac", "", "laminar")
    r.metrics["Nu (hot wall)"] = _m(nu, ref["nu"], "", "within 1.5 %", abs(nu - ref["nu"]) / ref["nu"] < 0.015, DVD_SRC)
    r.metrics["u_max on x=0.5"] = _m(umax, ref["umax"], "", "within 2 %", abs(umax - ref["umax"]) / ref["umax"] < 0.02,
                                     DVD_SRC)
    r.metrics["v_max on y=0.5"] = _m(vmax, ref["vmax"], "", "within 2 %", abs(vmax - ref["vmax"]) / ref["vmax"] < 0.02,
                                     DVD_SRC)
    r.details = {"steps": s.step_count, "converged": s.converged}
    return r


@benchmark("natural_convection_ra1e5_fmg", "Natural convection Ra=1e5, MAC 64x64 with FMG", "laminar")
def b_nc_1e5(ctx):
    from ..structured.case import natural_convection
    from ..structured.sequencing import coarse_case, fmg_factors, prolong
    from ..structured.solver import MACSolver
    from .reference import DVD

    be = _be(ctx)
    case = natural_convection(ra=1e5, n=64)
    prev = None
    for f in fmg_factors(case):
        cs = MACSolver(coarse_case(case, f), backend=be)
        if prev is not None:
            prolong(prev, cs)
        _run_mac(cs)
        prev = cs
    s = MACSolver(case, backend=be)
    if prev is not None:
        prolong(prev, s)
    _run_mac(s)
    nu = s.nusselt("left")
    ref = DVD[1e5]["nu"]
    r = BenchResult("natural_convection_ra1e5_fmg", "", "laminar")
    r.metrics["Nu (hot wall)"] = _m(nu, ref, "", "within 2 %", abs(nu - ref) / ref < 0.02, DVD_SRC)
    r.details = {"steps_fine": s.step_count, "fmg": fmg_factors(case), "converged": s.converged}
    return r


@benchmark("natural_convection_ra1e3_simplec", "Natural convection Ra=1e3, SIMPLEC + Boussinesq 48x48", "laminar")
def b_nc_simplec(ctx):
    from ..fluent.convert import structured_boundary_specs, structured_mesh
    from ..structured.case import natural_convection
    from ..unstructured.case import UnstructuredCase
    from ..unstructured.simple import SimpleSolver
    from .reference import DVD

    case = natural_convection(ra=1e3, n=48)
    mesh = structured_mesh(case)
    uc = UnstructuredCase(boundaries=structured_boundary_specs(case), viscosity=case.nu, energy=True,
                          conductivity=case.kappa, cp=1.0, buoyancy=list(case.buoyancy) + [0.0], t_ref=case.t_ref,
                          init_T=case.init_T, max_iter=5000, tol=1e-7)
    s = _run_simple(SimpleSolver(mesh, uc, backend=_be(ctx)))
    q = s.heat_flow()
    nu = q["left"]  # hot wall heat flow; k = 1, dT = 1, H = 1 -> Nu
    ref = DVD[1e3]["nu"]
    r = BenchResult("natural_convection_ra1e3_simplec", "", "laminar")
    r.metrics["Nu (hot wall)"] = _m(nu, ref, "", "within 1.5 %", abs(nu - ref) / ref < 0.015, DVD_SRC)
    r.metrics["energy imbalance |sum Q| / Q_hot"] = _m(abs(sum(q.values())) / abs(nu), 0.0, "", "< 1e-4",
                                                       abs(sum(q.values())) / abs(nu) < 1e-4)
    r.details = {"iterations": s.iteration, "converged": s.converged, "heat_flow": q}
    return r


@benchmark("channel_re50_mac", "Channel flow Re=50: Poiseuille profile and pressure gradient", "laminar")
def b_channel(ctx):
    from ..structured.case import channel_flow
    from ..structured.solver import MACSolver

    s = _run_mac(MACSolver(channel_flow(re=50, nx=128, ny=16, length=8.0), backend=_be(ctx)))
    c = s.case
    u, _ = s.cell_velocity()
    u = s.be.to_numpy(u)
    yc = (np.arange(c.ny) + 0.5) / c.ny
    ue = 6.0 * yc * (1 - yc)
    err = float(np.abs(u[-2] - ue).max())
    P = s.be.to_numpy(s.P)
    xc = (np.arange(c.nx) + 0.5) * c.dx
    sel = (xc > 0.5 * c.lx) & (xc < 0.9 * c.lx)
    dpdx = float(np.polyfit(xc[sel], P[sel].mean(axis=1), 1)[0])
    ref = -12.0 / 50.0
    r = BenchResult("channel_re50_mac", "", "laminar")
    r.metrics["outlet profile max |u - u_Poiseuille|"] = _m(err, 0.0, "", "< 0.01", err < 0.01, "analytic")
    r.metrics["dp/dx (developed region)"] = _m(dpdx, ref, "", "within 2 %", abs(dpdx - ref) / abs(ref) < 0.02,
                                               "analytic -12/Re")
    r.plots.append(Plot("Channel Re=50: outlet profile", "y", "u", [
        Series(list(yc), list(u[-2]), "FlowTrace"), Series(list(yc), list(ue), "Poiseuille", "ref")]))
    r.details = {"steps": s.step_count, "converged": s.converged}
    return r


@benchmark("simplec_mass_balance", "SIMPLEC global conservation (channel, exported step mesh)", "verification")
def b_mass(ctx):
    from ..fluent.convert import structured_boundary_specs, structured_mesh
    from ..mesh.unstructured import cartesian_mesh_2d
    from ..structured.case import backward_facing_step
    from ..structured.solver import MACSolver
    from ..unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
    from ..unstructured.simple import SimpleSolver

    be = _be(ctx)
    m = cartesian_mesh_2d(60, 12, 5.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"))
    bnd = default_boundaries(m)
    bnd["inlet"] = BoundarySpec("inlet", normal_speed=1.0)
    s1 = _run_simple(SimpleSolver(m, UnstructuredCase(boundaries=bnd, viscosity=0.02, max_iter=1500, tol=1e-7), backend=be))
    c = backward_facing_step(re=100, n_per_h=6)
    probe = MACSolver(c, backend=_be({"backend": "numpy"}))
    mesh = structured_mesh(c, probe.solid_h)
    s2 = _run_simple(SimpleSolver(mesh, UnstructuredCase(boundaries=structured_boundary_specs(c), viscosity=c.nu,
                                                         max_iter=3000, tol=1e-6), backend=be))
    r = BenchResult("simplec_mass_balance", "", "verification")
    for name, s in (("channel", s1), ("step mesh", s2)):
        mb = s.mass_balance()
        inflow = sum(-v for v in mb.values() if v < 0)
        imb = abs(sum(mb.values())) / inflow
        r.metrics[f"net mass imbalance / inflow, {name}"] = _m(imb, 0.0, "", "< 1e-8", imb < 1e-8)
    return r


# --------------------------------------------------------------------------
# grid convergence (GCI)
# --------------------------------------------------------------------------


@benchmark("gci_natural_convection", "Grid convergence index: Nu, natural convection Ra=1e4", "grid-convergence")
def b_gci_nc(ctx):
    from ..structured.case import natural_convection
    from ..structured.solver import MACSolver
    from .gci import gci
    from .reference import DVD

    grids = (32, 64, 128) if ctx["tier"] == "full" else (24, 48, 96)
    vals = []
    for n in grids:
        s = _run_mac(MACSolver(natural_convection(ra=1e4, n=n), backend=_be(ctx)))
        vals.append(s.nusselt("left"))
    h = [1.0 / n for n in grids]
    g = gci(vals[2], vals[1], vals[0], h[2], h[1], h[0])
    r = BenchResult("gci_natural_convection", "", "grid-convergence")
    for n, v in zip(grids, vals):
        r.metrics[f"Nu n={n}"] = _m(v, DVD[1e4]["nu"], "", source=DVD_SRC)
    r.metrics["observed order p"] = _m(g.observed_order, 2.0, "", "1.5 < p < 2.5", 1.5 < g.observed_order < 2.5)
    r.metrics["Richardson-extrapolated Nu"] = _m(g.extrapolated, DVD[1e4]["nu"], "", "within 1 %",
                                                 abs(g.extrapolated - DVD[1e4]["nu"]) / DVD[1e4]["nu"] < 0.01, DVD_SRC)
    r.metrics["GCI fine grid"] = _m(100 * g.gci_fine, None, "%")
    r.metrics["asymptotic-range ratio"] = _m(g.asymptotic_ratio, 1.0, "", "0.95-1.05",
                                             abs(g.asymptotic_ratio - 1) < 0.05)
    r.plots.append(Plot("Nu vs grid spacing", "h", "Nu", [
        Series(h, vals, "FlowTrace", "marker"), Series([0.0] + h, [g.extrapolated] * 4, "Richardson", "ref"),
        Series([0.0, h[0]], [DVD[1e4]["nu"]] * 2, "de Vahl Davis 1983", "ref")]))
    r.details = g.to_dict()
    return r


@benchmark("gci_cavity", "Grid convergence index: u_min on the cavity centreline, Re=100", "grid-convergence")
def b_gci_cavity(ctx):
    from ..structured.case import lid_driven_cavity
    from ..structured.solver import MACSolver
    from .gci import gci

    grids = (32, 64, 128) if ctx["tier"] == "full" else (24, 48, 96)
    vals = []
    for n in grids:
        s = _run_mac(MACSolver(lid_driven_cavity(re=100, n=n), backend=_be(ctx)))
        (y, u), _ = s.centerline()
        # parabolic refinement of the discrete minimum
        k = int(np.argmin(u))
        a, b, c = u[k - 1], u[k], u[k + 1]
        den = a - 2 * b + c
        vals.append(float(b - 0.125 * (a - c) ** 2 / den) if den != 0 else float(b))
    h = [1.0 / n for n in grids]
    g = gci(vals[2], vals[1], vals[0], h[2], h[1], h[0])
    r = BenchResult("gci_cavity", "", "grid-convergence")
    for n, v in zip(grids, vals):
        r.metrics[f"u_min n={n}"] = _m(v, -0.21090, "", source=GHIA + " (grid 129^2)")
    r.metrics["observed order p"] = _m(g.observed_order, 2.0, "", "1.5 < p < 2.5", 1.5 < g.observed_order < 2.5)
    r.metrics["Richardson-extrapolated u_min"] = _m(g.extrapolated, None, "")
    r.metrics["GCI fine grid"] = _m(100 * g.gci_fine, None, "%")
    r.details = g.to_dict()
    return r


# --------------------------------------------------------------------------
# turbulence (RANS)
# --------------------------------------------------------------------------


def _channel_result(id_, runs, tol_by_model):
    from .turbulent import DNS_UPLUS, DNS_YPLUS, MKM_SRC

    r = BenchResult(id_, "", "turbulence")
    series = [Series(list(DNS_YPLUS), list(DNS_UPLUS), "DNS (Moser, Kim & Mansour 1999)", "ref")]
    for res in runs:
        m = res["model"]
        tol = tol_by_model[m]
        r.metrics[f"{m}: max |U+ - DNS| (y+ >= {res['compared_from_yplus']:.0f})"] = _m(
            res["max_abs_err_pct"], 0.0, "%", f"< {tol} %", res["max_abs_err_pct"] < tol, MKM_SRC)
        r.metrics[f"{m}: centreline U+ error"] = _m(res["centre_err_pct"], 0.0, "%")
        r.metrics[f"{m}: wall shear (exact 1 at equilibrium)"] = _m(res["tau_w"], 1.0, "", "within 1 %",
                                                                     abs(res["tau_w"] - 1) < 0.01, "momentum balance")
        r.metrics[f"{m}: iterations"] = _m(res["iterations"], None, "")
        yp = [y * 392.24 for y in res["profile_y"]]
        series.append(Series(yp, res["profile_u"], f"FlowTrace {m}", "line"))
    r.plots.append(Plot("Channel Re_tau = 392: U+(y+)", "y+", "U+", series, logx=True))
    r.details = {"runs": runs}
    return r


@benchmark("channel_retau395_sst", "Turbulent channel Re_tau=392, k-omega SST (y+ ~ 0.25) vs DNS", "turbulence")
def b_channel_sst(ctx):
    from .turbulent import run_channel

    return _channel_result("channel_retau395_sst", [run_channel("k-omega-sst")], {"k-omega-sst": 5.0})


@benchmark("channel_retau395_kepsilon", "Turbulent channel Re_tau=392, k-epsilon + wall functions vs DNS",
           "turbulence")
def b_channel_ke(ctx):
    from .turbulent import run_channel

    runs = [run_channel("k-epsilon-standard"), run_channel("k-epsilon-realizable")]
    return _channel_result("channel_retau395_kepsilon", runs,
                           {"k-epsilon-standard": 5.0, "k-epsilon-realizable": 10.0})


def _schultz_grunow(re_x):
    return 0.370 * np.log10(np.asarray(re_x)) ** -2.584


@benchmark("flat_plate_sst", "Turbulent flat plate Re_L=5e6 (NASA TMR set-up), k-omega SST", "turbulence",
           tiers=("full",))
def b_flat_plate_sst(ctx):
    return _flat_plate("flat_plate_sst", "k-omega-sst")


@benchmark("flat_plate_kepsilon", "Turbulent flat plate Re_L=5e6, realizable k-epsilon + wall functions",
           "turbulence", tiers=("full",))
def b_flat_plate_ke(ctx):
    return _flat_plate("flat_plate_kepsilon", "k-epsilon-realizable")


def _flat_plate(id_, model):
    from .turbulent import WHITE_SRC, run_flat_plate

    res = run_flat_plate(model)
    cf = np.array(res["cf"])
    rex = np.array(res["re_x"])
    sg = _schultz_grunow(rex)
    sel = (rex >= 1e6) & (rex <= 1e7)
    dev_sg = 100 * (cf[sel] - sg[sel]) / sg[sel]
    r = BenchResult(id_, "", "turbulence")
    r.metrics["Cf at x = 0.97 (Re_x = 4.85e6)"] = _m(res["cf_at_x097"], res["white_at_x097"], "", None, None, WHITE_SRC)
    r.metrics["mean deviation from White, 1e6 < Re_x < 1e7"] = _m(res["mean_dev_pct"], 0.0, "%", "within 12 %",
                                                                   abs(res["mean_dev_pct"]) < 12, WHITE_SRC)
    r.metrics["mean deviation from Schultz-Grunow, same range"] = _m(
        float(dev_sg.mean()), 0.0, "%", "within 6 %", abs(dev_sg.mean()) < 6,
        "Schultz-Grunow Cf = 0.370 (log10 Re_x)^-2.584")
    r.metrics["max wall y+"] = _m(res["y_plus_max"], None, "")
    r.metrics["iterations"] = _m(res["iterations"], None, "")
    r.plots.append(Plot(f"Flat plate skin friction ({model})", "Re_x", "Cf", [
        Series(list(rex), list(cf), "FlowTrace"), Series(list(rex), list(white_cf_list(rex)), "White 1974", "line"),
        Series(list(rex), list(sg), "Schultz-Grunow", "line")], logx=True))
    r.details = {k: res[k] for k in ("iterations", "converged", "seconds", "cells")}
    return r


def white_cf_list(rex):
    from .turbulent import white_cf

    return white_cf(np.maximum(rex, 1e3))


@benchmark("bfs_driver_seegmiller_sst", "Backward-facing step (Driver & Seegmiller 1985), k-omega SST", "turbulence",
           tiers=("full",))
def b_bfs(ctx):
    from .turbulent import DS_SRC, DS_XR, run_bfs

    res = run_bfs("k-omega-sst")
    r = BenchResult("bfs_driver_seegmiller_sst", "", "turbulence")
    xr = res["x_r"]
    r.metrics["reattachment length x_r / H"] = _m(xr, DS_XR, "", "within 10 %",
                                                  xr is not None and abs(xr - DS_XR) / DS_XR < 0.10, DS_SRC)
    r.metrics["iterations"] = _m(res["iterations"], None, "")
    r.metrics["cells"] = _m(res["cells"], None, "")
    x, cf = np.array(res["x"]), np.array(res["cf"])
    sel = (x > -4) & (x < 30)
    r.plots.append(Plot("BFS floor skin friction (SST)", "x / H", "Cf", [
        Series(list(x[sel]), list(cf[sel]), "FlowTrace"), Series([DS_XR], [0.0], "Driver & Seegmiller x_r", "ref")]))
    r.details = {k: res[k] for k in ("iterations", "converged", "seconds", "cells")}
    return r


# --------------------------------------------------------------------------
# rotating frames, periodic zones, conjugate heat transfer
# --------------------------------------------------------------------------


@benchmark("taylor_couette", "Taylor-Couette flow: rotating wall and MRF zone vs exact solution", "verification")
def b_taylor_couette(ctx):
    from ..mesh.unstructured import annulus_mesh, set_cell_zone
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    be = _be({"backend": "numpy"})
    variants = {
        "rotating wall": lambda m: (m, dict(boundaries={"inner": BoundarySpec("wall", rotation={"omega": 1.0})})),
        "MRF inner ring": lambda m: (set_cell_zone(m, lambda x: np.linalg.norm(x, axis=1) < 1.5, "rotor"),
                                     dict(mrf={"rotor": {"omega": 1.0}})),
    }
    r = BenchResult("taylor_couette", "", "verification")
    for label, make in variants.items():
        errs = []
        for nr, nt in ((8, 48), (16, 96)):
            m, kw = make(annulus_mesh(1.0, 2.0, nr, nt))
            s = _run_simple(SimpleSolver(m, UnstructuredCase(viscosity=0.1, max_iter=4000, tol=1e-9, **kw), backend=be))
            xc, f = s.cell_centres(), s.fields()
            rad = np.linalg.norm(xc, axis=1)
            ut = (-xc[:, 1] * f["x-velocity"] + xc[:, 0] * f["y-velocity"]) / rad
            errs.append(float(np.abs(ut - (-rad / 3 + 4 / (3 * rad))).max()))
        order = math.log2(errs[0] / errs[1])
        r.metrics[f"{label}: max |u_theta - exact|, 16x96"] = _m(errs[1], 0.0, "", "< 5e-3", errs[1] < 5e-3, "exact")
        r.metrics[f"{label}: observed order"] = _m(order, 2.0, "", "> 1.8", order > 1.8, "exact")
    return r


@benchmark("periodic_poiseuille", "Periodic channel driven by a pressure gradient (translational periodicity)",
           "verification")
def b_periodic(ctx):
    from ..mesh.unstructured import cartesian_mesh_2d, make_periodic
    from ..unstructured.case import UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    errs = []
    for ny, tri in ((8, False), (16, False), (16, True)):
        m = make_periodic(cartesian_mesh_2d(4, ny, 0.5, 1.0, names=("in", "out", "bottom", "top"), triangulate=tri,
                                            warp=0.0), "in", "out")
        s = _run_simple(SimpleSolver(m, UnstructuredCase(viscosity=0.1, body_force=[1.0, 0, 0], max_iter=6000,
                                                         tol=1e-9, relax_u=0.9), backend=_be({"backend": "numpy"})))
        xc, u = s.cell_centres(), s.fields()["x-velocity"]
        errs.append(float(np.abs(u - 5.0 * xc[:, 1] * (1 - xc[:, 1])).max() / 1.25))
    r = BenchResult("periodic_poiseuille", "", "verification")
    r.metrics["quads 4x8: max |u - exact| / u_max"] = _m(errs[0], 0.0, "")
    r.metrics["quads 4x16: max |u - exact| / u_max"] = _m(errs[1], 0.0, "", "< 0.01", errs[1] < 0.01, "analytic")
    r.metrics["triangles 4x16: max |u - exact| / u_max"] = _m(errs[2], 0.0, "", "< 0.01", errs[2] < 0.01, "analytic")
    order = math.log2(errs[0] / errs[1])
    r.metrics["observed order (quads)"] = _m(order, 2.0, "", "> 1.8", order > 1.8, "analytic")
    return r


@benchmark("cht_composite_wall", "Conjugate heat transfer: composite wall (solid | fluid | solid)", "verification")
def b_cht_composite(ctx):
    from ..mesh.unstructured import cartesian_mesh_2d, set_cell_zone
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.conjugate import make_solver

    m = cartesian_mesh_2d(30, 4, 3.0, 1.0)
    m = set_cell_zone(m, lambda x: x[:, 0] < 1.0, "solid-a", "solid")
    m = set_cell_zone(m, lambda x: x[:, 0] > 2.0, "solid-b", "solid")
    c = UnstructuredCase(energy=True, conductivity=1.0, viscosity=0.1,
                         solids={"solid-a": {"conductivity": 5.0}, "solid-b": {"conductivity": 0.5}},
                         boundaries={"left": BoundarySpec("wall", thermal="temperature", value=1.0),
                                     "right": BoundarySpec("wall", thermal="temperature", value=0.0)},
                         max_iter=500, tol=1e-10)
    s = _run_simple(make_solver(m, c, backend=_be({"backend": "numpy"})))
    x, T = s.cell_centres()[:, 0], s.fields()["temperature"]
    q = 1.0 / (1 / 5 + 1 + 1 / 0.5)
    Te = np.where(x < 1, 1 - q * x / 5, np.where(x < 2, 1 - q / 5 - q * (x - 1), 1 - q / 5 - q - q * (x - 2) / 0.5))
    hf = s.heat_flow()
    r = BenchResult("cht_composite_wall", "", "verification")
    r.metrics["max |T - exact|"] = _m(float(np.abs(T - Te).max()), 0.0, "", "< 1e-10", np.abs(T - Te).max() < 1e-10,
                                      "series thermal resistances")
    r.metrics["heat flux"] = _m(hf["left"], q, "W/m", "within 1e-10", abs(hf["left"] - q) < 1e-10, "analytic")
    return r


@benchmark("cht_heated_channel", "Conjugate heat transfer: channel over a heat-generating solid", "verification")
def b_cht_channel(ctx):
    from ..mesh.unstructured import cartesian_mesh_2d, set_cell_zone, split_zone
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.conjugate import make_solver

    m = cartesian_mesh_2d(60, 20, 4.0, 1.25, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"), y0=-0.25)
    m = split_zone(m, "inlet", lambda fc: fc[:, 1] < 0.0, "heater-left", "wall")
    m = split_zone(m, "outlet", lambda fc: fc[:, 1] < 0.0, "heater-right", "wall")
    m = set_cell_zone(m, lambda x: x[:, 1] < 0.0, "heater", "solid")
    c = UnstructuredCase(energy=True, conductivity=0.05, viscosity=0.05,
                         solids={"heater": {"conductivity": 2.0, "source": 4.0}},
                         boundaries={"inlet": BoundarySpec("inlet", normal_speed=1.0, value=0.0),
                                     "outlet": BoundarySpec("outlet")}, max_iter=3000, tol=1e-8)
    s = _run_simple(make_solver(m, c, backend=_be({"backend": "numpy"})))
    q = s.heat_flow()
    total = sum(v for k, v in q.items() if k != "fluid-solid interfaces")
    r = BenchResult("cht_heated_channel", "", "verification")
    r.metrics["heat crossing the fluid-solid interface"] = _m(q["fluid-solid interfaces"], 4.0, "W/m", "within 1e-6",
                                                              abs(q["fluid-solid interfaces"] - 4.0) < 1e-6,
                                                              "source x volume (adiabatic outer walls)")
    r.metrics["global energy imbalance"] = _m(abs(total), 0.0, "W/m", "< 1e-8", abs(total) < 1e-8)
    r.metrics["enthalpy rise inlet -> outlet"] = _m(-(q["inlet"] + q["outlet"]), 4.0, "W/m", "within 1e-6",
                                                    abs(-(q["inlet"] + q["outlet"]) - 4.0) < 1e-6)
    r.details = {"heat_flow": q, "iterations": s.iteration}
    return r


@benchmark("warmstart_proxy", "Warm start: FlowTrace .ip -> SIMPLEC on another mesh (proxy for Fluent)",
           "fluent", tiers=("full",))
def b_warmstart(ctx):
    from ..fluent.warmstart_study import FIELD_TOL, run_study

    out = ctx.get("out_dir")
    folder = os.path.join(os.path.dirname(os.path.abspath(out)), "fluent_warmstart") if out else "benchmarks/fluent_warmstart"
    data = run_study(folder, backend=_be(ctx).name, verbose=False)
    r = BenchResult("warmstart_proxy", "", "fluent")
    for p in data["proxy"]:
        r.metrics[f"{p['case']}: iterations to the converged field, cold / warm"] = _m(
            p["speedup"], None, "x", "> 1", p["speedup"] > 1, f"field within {FIELD_TOL:g} of the converged solution")
        r.metrics[f"{p['case']}: warm run reaches the cold-start solution"] = _m(
            p["max_velocity_difference"], 0.0, "", f"<= {FIELD_TOL:g}", p["warm_converged"])
    r.details = data
    return r


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def machine_info() -> dict:
    from ..hardware.backend import available_backends
    from ..hardware.probe import cached_profile

    p = cached_profile()
    g = p.best_gpu
    return {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "os": p.os, "python": p.python, "cpu": p.cpu_model,
        "cores": f"{p.physical_cores}C/{p.logical_cores}T", "ram_gb": round(p.ram_total / 2 ** 30, 1),
        "gpu": g.name if g else None, "backends": available_backends(),
        "libraries": p.libraries, "flowtrace": _version(),
        "platform": platform.platform(),
    }


def _version() -> str:
    from .. import __version__

    return __version__


def run_benchmarks(ids: Optional[List[str]] = None, tier: str = "quick", backend: str = "auto",
                   out_dir: Optional[str] = None, verbose: bool = True, report: bool = True) -> dict:
    """Run the selected benchmarks; write results.json (+ report) to ``out_dir``."""
    sel = [b for b in REGISTRY.values() if (ids and b.id in ids) or (not ids and tier in b.tiers)]
    if ids:
        missing = set(ids) - set(REGISTRY)
        if missing:
            raise KeyError(f"unknown benchmark(s): {sorted(missing)}; have {sorted(REGISTRY)}")
    ctx = {"tier": tier, "backend": backend, "cpu_backend": "numba" if _has_numba() else "numpy", "out_dir": out_dir}
    results = []
    for b in sel:
        t0 = time.perf_counter()
        if verbose:
            print(f"[bench] {b.id} ...", flush=True)
        try:
            r = b.fn(ctx)
        except Exception as exc:  # a failing benchmark is reported, not fatal
            r = BenchResult(b.id, b.title, b.category, error=f"{type(exc).__name__}: {exc}",
                            details={"traceback": traceback.format_exc()})
        r.id, r.title, r.category = b.id, b.title, b.category
        r.seconds = time.perf_counter() - t0
        results.append(r)
        if verbose:
            mark = "PASS" if r.passed else "FAIL"
            print(f"[bench] {mark} {b.id} ({r.seconds:.1f} s)" + (f"  {r.error}" if r.error else ""), flush=True)
            for k, m in r.metrics.items():
                ref = "" if m.reference is None else f"  ref {m.reference:.6g}"
                flag = "" if m.passed is None else ("  ok" if m.passed else "  FAIL")
                print(f"         {k:<44} {m.value:.6g} {m.unit}{ref}{flag}", flush=True)
    data = {"machine": machine_info(), "tier": tier, "backend": backend,
            "results": [r.to_dict() for r in results],
            "summary": {"total": len(results), "passed": sum(r.passed for r in results)}}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "results.json")
        # merge with an existing file so that partial reruns keep the other entries
        if os.path.exists(path) and ids:
            try:
                with open(path, encoding="utf-8") as fh:
                    old = json.load(fh)
                keep = [r for r in old.get("results", []) if r["id"] not in {x.id for x in results}]
                data["results"] = keep + data["results"]
                data["summary"] = {"total": len(data["results"]),
                                   "passed": sum(bool(r.get("passed")) for r in data["results"])}
            except Exception:
                pass
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        if report:
            from .report import write_report

            write_report(data, out_dir)
    return data


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except Exception:
        return False
