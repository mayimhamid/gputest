"""Why the cylinder warm start is slower than a cold start (warm-start study, Re = 40).

The study pre-solves the cylinder on the structured grid (12 staircase cells per
diameter) and interpolates the field onto a 0.75x coarser grid (9 cells per
diameter). This script measures the recirculation length behind the body on both
grids, for the interpolated pre-solve field and for SIMPLEC converged on each grid,
and how far the interpolated field is from the target grid's own solution.

Needs the pre-solve field of the study: run
``flowtrace benchmark --only warmstart_proxy --out benchmarks/results`` first, or pass
``--ip`` to another ``flowtrace.ip`` of the cylinder case.

Run:  python benchmarks/warmstart_cylinder_wake.py
Writes benchmarks/results/warmstart_cylinder_wake.md.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def recirculation_length(xc, u, x_rear, y_axis, half_band):
    """Distance from the rear of the body to the last reversed-flow cell centre on the wake axis."""
    sel = (np.abs(xc[:, 1] - y_axis) < half_band) & (xc[:, 0] > x_rear)
    neg = xc[sel, 0][u[sel] < 0]
    return float(neg.max() - x_rear) if neg.size else 0.0


def run(ip_path):
    from flowtrace.fluent import warmstart_study as W
    from flowtrace.fluent.interpolate import interpolate_ip_file
    from flowtrace.hardware.backend import get_backend
    from flowtrace.unstructured.simple import SimpleSolver

    base = W.CASES["cylinder"]
    ob = W._structured_case(base).obstacles[0]
    x_rear, y_axis = ob.x0 + ob.r, ob.y0
    rows = []
    for label, scale in (("source grid (pre-solve resolution)", 1.0), ("target grid of the study", base["target"]["scale"])):
        spec = dict(base, target={"kind": "structured", "scale": scale})
        case = W._structured_case(spec)
        mesh = W._target_mesh(case, spec)
        n_per_d = 2 * ob.r * case.nx * scale / case.lx
        s = SimpleSolver(mesh, W._ucase(case, 20000, 1e-7), backend=get_backend("numpy"))
        xc = s.cell_centres()
        band = 0.6 * case.ly / (case.ny * scale)
        ip = interpolate_ip_file(ip_path, xc)
        while not s.finished:
            s.step()
        f = s.fields()
        mag_ip = np.hypot(ip["x-velocity"], ip["y-velocity"])
        dev = float(np.abs(mag_ip - f["velocity-magnitude"]).max() / np.abs(f["velocity-magnitude"]).max())
        rows.append({"grid": label, "cells_per_d": round(n_per_d, 1), "cells": mesh.n_cells,
                     "L_presolve": recirculation_length(xc, ip["x-velocity"], x_rear, y_axis, band),
                     "L_simplec": recirculation_length(xc, f["x-velocity"], x_rear, y_axis, band),
                     "cell": 2 * ob.r / n_per_d, "deviation": dev, "iterations": s.iteration, "converged": s.converged})
        print(rows[-1], flush=True)
    return rows


def render_markdown(rows) -> str:
    lines = ["| Grid | Cells per D | Recirculation length, pre-solve field | Recirculation length, SIMPLEC converged | "
             "Max-norm distance of the interpolated field |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['grid']} | {r['cells_per_d']:g} | {r['L_presolve']:.2f} D | {r['L_simplec']:.2f} D | "
                     f"{100 * r['deviation']:.0f} % |")
    lines += ["", f"_Lengths from the rear of the cylinder to the last reversed-flow cell centre on the wake axis; "
                  f"resolution one cell ({rows[0]['cell']:.2f} D and {rows[1]['cell']:.2f} D). "
                  "Reproduce: `python benchmarks/warmstart_cylinder_wake.py`._", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ip", default=os.path.join(ROOT, "benchmarks", "fluent_warmstart", "cylinder", "flowtrace.ip"))
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    a = ap.parse_args()
    if not os.path.exists(a.ip):
        raise SystemExit(f"{a.ip} not found: run `flowtrace benchmark --only warmstart_proxy` first")
    md = render_markdown(run(a.ip))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "warmstart_cylinder_wake.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
