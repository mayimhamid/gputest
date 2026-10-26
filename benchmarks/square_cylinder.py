"""Square cylinder in a channel, steady laminar flow: drag, recirculation length, grid convergence.

Three studies (all U = 1, rho = 1, steady SIMPLEC, tol 1e-7 on the scaled residuals):

  meshes   the project's GAMBIT four-mesh independence set (D = 4, inlet 4.5 D upstream, side
           walls 6 D from the axis), side walls no-slip (as in the mesh) or symmetry, Re = 40,
           plus the Celik GCI of Cd and Lr/D on the three finest meshes;
  sweep    Re = 10, 20, 30, 40 on one mesh; checks P1 (Lr/D linear in Re, R^2 >= 0.99) and
           P2 (Cd ~ Re^n with n = -0.66 +- 0.10, Sen, Mittal & Biswas 2011);
  domain   the same physics on generated meshes (D = 1, h = D/50 near the body) in the
           user's domain and in a large one (blockage 8.3 % vs 2.5 %): isolates the domain effect.

Drag is ``solver.wall_forces()`` on the body (pressure + viscous, per unit depth); Lr is where
u(x, 0) behind the rear face changes sign.

Run:  python benchmarks/square_cylinder.py meshes --dir "<folder with SquareCylinder-*.msh>"
      python benchmarks/square_cylinder.py sweep --dir "<folder>" --mesh Fine
      python benchmarks/square_cylinder.py domain
Writes benchmarks/results/square_cylinder_<study>.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from scipy.interpolate import LinearNDInterpolator  # noqa: E402

LEVELS = ("Coarse", "Medium", "Fine", "Finest")
BODY_FACES = {"Coarse": 80, "Medium": 160, "Fine": 240, "Finest": 320}  # faces on the square: h ~ 1/N


def solve(mesh, bnd, mu, body_zone, d, x_rear, x_end, max_iter=8000):
    from flowtrace.hardware.planner import plan_unstructured
    from flowtrace.unstructured.case import UnstructuredCase

    case = UnstructuredCase(boundaries=bnd, density=1.0, viscosity=mu, max_iter=max_iter, tol=1e-7)
    t0 = time.perf_counter()
    plan, s = plan_unstructured(mesh, case)
    while not s.finished:
        s.step()
    f = s.wall_forces()[body_zone]
    q = 0.5 * d
    ip = LinearNDInterpolator(s.cell_centres(), s.fields()["x-velocity"])
    xs = np.linspace(x_rear + 1e-3 * d, x_end, 8000)
    u = ip(xs, np.zeros_like(xs))
    k = np.nonzero((u[:-1] < 0) & (u[1:] >= 0))[0]
    lr = (xs[k[0]] - u[k[0]] * (xs[k[0] + 1] - xs[k[0]]) / (u[k[0] + 1] - u[k[0]]) - x_rear) / d if k.size else math.nan
    mb = s.mass_balance()
    inflow = -sum(v for v in mb.values() if v < 0)
    return {"cells": mesh.n_cells, "Re": d / mu, "backend": plan.backend, "iterations": s.iteration,
            "converged": bool(s.converged), "seconds": round(time.perf_counter() - t0, 1),
            "Cd": f["total"][0] / q, "Cd_pressure": f["pressure"][0] / q, "Cd_viscous": f["viscous"][0] / q,
            "Cl": f["total"][1] / q, "Lr_over_D": lr, "mass_imbalance": sum(mb.values()) / inflow,
            "stability_rollbacks": len(getattr(s, "guard_events", []))}


def user_mesh(folder, level, sides):
    from flowtrace.mesh.fluent_h5 import read_any_mesh
    from flowtrace.unstructured.case import BoundarySpec, default_boundaries

    mesh, _ = read_any_mesh(os.path.join(folder, f"SquareCylinder-{level}.msh"))
    bnd = default_boundaries(mesh)
    if sides == "symmetry":
        bnd["SideWalls"] = BoundarySpec("symmetry")
    return mesh, bnd


def generated_mesh(up, side, down, h0=0.02, grow=1.06, hmax=0.5):
    from flowtrace.mesh.unstructured import cartesian_mesh_2d, extract_cells, set_cell_zone
    from flowtrace.unstructured.case import BoundarySpec

    def axis(lo, hi):
        def walk(start, end, sign):
            pts, h, x = [], h0, start
            while sign * (end - x) > 1e-9:
                h = min(h * grow, hmax)
                x = x + sign * h
                if sign * (x - end) > -0.3 * h:
                    x = end
                pts.append(x)
            return pts
        return np.array(walk(-1.0, lo, -1)[::-1] + list(np.arange(-1.0, 1.0 + h0 / 2, h0)) + walk(1.0, hi, 1))

    mesh = cartesian_mesh_2d(0, 0, xs=axis(-0.5 - up, 0.5 + down), ys=axis(-side, side),
                             names=("inlet", "outlet", "bottom", "top"))
    mesh = set_cell_zone(mesh, lambda xc: (np.abs(xc[:, 0]) < 0.5) & (np.abs(xc[:, 1]) < 0.5), "body")
    mesh, _, _ = extract_cells(mesh, mesh.cell_zone != mesh.zone_by_name("body").id)
    bnd = {"inlet": BoundarySpec("inlet", normal_speed=1.0), "outlet": BoundarySpec("outlet"),
           "bottom": BoundarySpec("symmetry"), "top": BoundarySpec("symmetry"), "interface-body": BoundarySpec("wall")}
    return mesh, bnd


def gci_rows(rows, key):
    from flowtrace.validation.gci import gci

    by = {r["mesh"]: r for r in rows}
    trip = ("Finest", "Fine", "Medium")
    g = gci(*(by[m][key] for m in trip), *(1.0 / BODY_FACES[m] for m in trip)).to_dict()
    return {k: g[k] for k in ("observed_order", "extrapolated", "gci_fine", "asymptotic_ratio", "oscillatory")}


def environment():
    from flowtrace import __version__, native

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, dirty = "", False
    return {"date": dt.datetime.now().isoformat(timespec="seconds"), "flowtrace": __version__, "git_commit": commit,
            "git_dirty": dirty, "native_kernels": native.status()["cpu_kernels"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("study", choices=["meshes", "sweep", "domain"])
    ap.add_argument("--dir", default=r"C:\Users\Hamid\Desktop\Square Cylinder Mesh Independency")
    ap.add_argument("--mesh", default="Fine", help="sweep: which mesh of the set")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    a = ap.parse_args()
    env = environment()
    md = [f"Measured {env['date']}, FlowTrace {env['flowtrace']} @ {env['git_commit'][:10]}"
          f"{' (uncommitted changes)' if env['git_dirty'] else ''}; steady SIMPLEC, residual tolerance 1e-7.", ""]
    data = {"study": a.study, "environment": env}
    if a.study == "meshes":
        rows = []
        for sides in ("wall", "symmetry"):
            for lvl in LEVELS:
                mesh, bnd = user_mesh(a.dir, lvl, sides)
                r = {"mesh": lvl, "sides": sides, **solve(mesh, bnd, 0.1, "BluffWalls", 4.0, -18.0, 30.0)}
                rows.append(r)
                print(json.dumps(r), flush=True)
        data["rows"] = rows
        data["gci"] = {s: {k: gci_rows([r for r in rows if r["sides"] == s], k) for k in ("Cd", "Lr_over_D")}
                       for s in ("wall", "symmetry")}
        md += ["| mesh | cells | side walls | Cd | Cd pressure | Cd viscous | Lr/D | iterations | converged |",
               "|---|---:|---|---:|---:|---:|---:|---:|---|"]
        md += [f"| {r['mesh']} | {r['cells']:,} | {r['sides']} | {r['Cd']:.4f} | {r['Cd_pressure']:.4f} | "
               f"{r['Cd_viscous']:.4f} | {r['Lr_over_D']:.4f} | {r['iterations']} | {r['converged']} |" for r in rows]
        md += ["", "Grid convergence (Celik et al. 2008; Medium/Fine/Finest, h ~ 1 / faces on the body):", "",
               "| side walls | quantity | observed order | extrapolated | GCI fine | asymptotic ratio |",
               "|---|---|---:|---:|---:|---:|"]
        for s, g in data["gci"].items():
            for k, v in g.items():
                md.append(f"| {s} | {k} | {v['observed_order']:.2f} | {v['extrapolated']:.4f} | "
                          f"{100 * v['gci_fine']:.2f} % | {v['asymptotic_ratio']:.2f} |")
    elif a.study == "sweep":
        rows = []
        for re_ in (10, 20, 30, 40):
            mesh, bnd = user_mesh(a.dir, a.mesh, "symmetry")
            r = {"mesh": a.mesh, **solve(mesh, bnd, 4.0 / re_, "BluffWalls", 4.0, -18.0, 30.0)}
            rows.append(r)
            print(json.dumps(r), flush=True)
        re_v = np.array([r["Re"] for r in rows])
        lr, cd = np.array([r["Lr_over_D"] for r in rows]), np.array([r["Cd"] for r in rows])
        a1, b1 = np.polyfit(re_v, lr, 1)
        r2 = 1 - ((lr - (a1 * re_v + b1)) ** 2).sum() / ((lr - lr.mean()) ** 2).sum()
        n = float(np.polyfit(np.log(re_v), np.log(cd), 1)[0])
        data.update(rows=rows, lr_fit={"slope": a1, "intercept": b1, "r2": r2, "onset_Re": -b1 / a1},
                    cd_exponent=n, P1=bool(r2 >= 0.99), P2=bool(abs(n + 0.66) <= 0.10))
        md += ["| Re | Cd | Lr/D | iterations |", "|---:|---:|---:|---:|"]
        md += [f"| {r['Re']:.0f} | {r['Cd']:.4f} | {r['Lr_over_D']:.4f} | {r['iterations']} |" for r in rows]
        md += ["", f"P1: Lr/D = {a1:.5f} Re {b1:+.4f}, R^2 = {r2:.5f}, separation onset Re = {-b1 / a1:.2f} "
               f"(Sen et al. 2011: 1.15) -> {'pass' if data['P1'] else 'FAIL'}.",
               f"P2: Cd ~ Re^{n:.3f} (criterion -0.66 +- 0.10) -> {'pass' if data['P2'] else 'FAIL'}."]
    else:
        rows = []
        for name, (up, side, down) in (("user-like", (4.5, 6.0, 14.5)), ("large", (15.0, 20.0, 30.0))):
            mesh, bnd = generated_mesh(up, side, down)
            for re_ in (10, 40):
                r = {"domain": name, "blockage": 1 / (2 * side),
                     **solve(mesh, bnd, 1.0 / re_, "interface-body", 1.0, 0.5, 0.5 + down - 1, max_iter=12000)}
                rows.append(r)
                print(json.dumps(r), flush=True)
        data["rows"] = rows
        md += ["| domain | blockage | cells | Re | Cd | Lr/D | iterations | converged |",
               "|---|---:|---:|---:|---:|---:|---:|---|"]
        md += [f"| {r['domain']} | {100 * r['blockage']:.1f} % | {r['cells']:,} | {r['Re']:.0f} | {r['Cd']:.4f} | "
               f"{r['Lr_over_D']:.4f} | {r['iterations']} | {r['converged']} |" for r in rows]
        for name in ("user-like", "large"):
            c = {r["Re"]: r["Cd"] for r in rows if r["domain"] == name}
            md.append(f"\n{name}: Cd exponent between Re 10 and 40 = {math.log(c[40.0] / c[10.0]) / math.log(4):.3f}")
    md += ["", f"Reproduce: `python benchmarks/square_cylinder.py {a.study}`", ""]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"square_cylinder_{a.study}.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=float)
    with open(os.path.join(a.out, f"square_cylinder_{a.study}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
