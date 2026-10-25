"""Validation suite: every solver path against published or exact references."""

from __future__ import annotations

import time

import numpy as np

from ..hardware.backend import get_backend


def _run(solver, check_every=10):
    while not solver.finished:
        solver.step()
        if solver.step_count % check_every == 0 and hasattr(solver, "nx"):
            solver.residuals()
    return solver


def run_suite(quick: bool = True, backend: str = "auto", verbose: bool = True) -> bool:
    from ..mesh.unstructured import cartesian_mesh_2d
    from ..structured.case import channel_flow, lid_driven_cavity, natural_convection, taylor_green
    from ..structured.solver import MACSolver
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver
    from .checks import validate_structured
    from .reference import compare_ghia

    be = get_backend(backend)
    n = 64 if quick else 128
    rows = []

    def record(name, value, ref, ok, seconds):
        rows.append((name, value, ref, ok, seconds))
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name:<52} {value:<14.5g} {str(ref):<12} {seconds:6.1f} s")

    cases = [
        ("MAC  Taylor-Green L-inf error", taylor_green(n=n, re=10.0, t_end=0.5)),
        ("MAC  lid-driven cavity Re=100 vs Ghia (max |du|)", lid_driven_cavity(re=100, n=n)),
        ("MAC  natural convection Ra=1e4, Nu vs de Vahl Davis", natural_convection(ra=1e4, n=n)),
        ("MAC  channel Re=100, outlet vs Poiseuille", channel_flow(re=100, nx=4 * n, ny=n // 4 * 1 if quick else 32)),
    ]
    for label, case in cases:
        t0 = time.perf_counter()
        s = _run(MACSolver(case, backend=be))
        v = validate_structured(s)["checks"]
        main = v[0]
        record(label, main["value"], main.get("reference", "-"), bool(main["pass"]), time.perf_counter() - t0)
        div = v[-1]
        record("     " + div["name"], div["value"], "0", bool(div["pass"]), 0.0)

    t0 = time.perf_counter()
    mesh = cartesian_mesh_2d(48, 48, triangulate=True, jitter=0.2)
    uc = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                          max_iter=3000, tol=1e-6)
    s = _run(SimpleSolver(mesh, uc, backend=be))
    from scipy.interpolate import griddata

    xc, f = s.cell_centres(), s.fields()
    y = np.linspace(0.01, 0.99, 99)
    u = griddata(xc, f["x-velocity"], np.column_stack([np.full_like(y, 0.5), y]))
    vv = griddata(xc, f["y-velocity"], np.column_stack([y, np.full_like(y, 0.5)]))
    err = compare_ghia(100, y, u, y, vv)
    record("SIMPLEC skewed triangles, cavity Re=100 vs Ghia", err["u_max_err"], "Ghia", err["u_max_err"] < 0.03,
           time.perf_counter() - t0)
    ok = all(r[3] for r in rows)
    if verbose:
        print("\nAll checks passed." if ok else "\nSome checks failed.")
    return ok
