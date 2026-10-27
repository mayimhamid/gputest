"""Experiment: where does SIMPLE/SIMPLEC stop converging on skewed meshes, and why?

Set-up. Lid-driven cavity, nu = 0.1 (Re = 10), 24 x 24 quads sheared uniformly,
x' = x + tan(theta) y, so every face has non-orthogonality theta. For each angle: SIMPLEC with
a fixed pressure under-relaxation alpha_p in {1, 0.8, 0.5, 0.33, 0.2} (stability guard off),
SIMPLEC with automatic alpha_p (guard on) and SIMPLE (alpha_p = 0.3); at most 4,000 iterations,
scaled-residual tolerance 1e-6. A run counts as diverged once the raw continuity residual is
non-finite or above 1e6.

Hypothesis. The pressure-correction equation neglects the non-orthogonal part of its
Laplacian, which grows like tan^2(theta) relative to the retained part, so the largest stable
alpha_p should fall like cos^2(theta) and plain SIMPLE should fail as well.

Run:  python benchmarks/skew_envelope.py   (about 10-20 min on 4 cores)
Writes benchmarks/results/skew_envelope.{json,md}.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANGLES = (0, 10, 20, 30, 35, 40, 45, 50, 55, 60)
RELAX = (1.0, 0.8, 0.5, 0.33, 0.2)


def mesh(theta_deg: float, n: int = 24):
    from flowtrace.mesh.unstructured import cartesian_mesh_2d

    m = cartesian_mesh_2d(n, n)
    m.nodes[:, 0] = m.nodes[:, 0] + math.tan(math.radians(theta_deg)) * m.nodes[:, 1]
    return m


def run(theta: float, algorithm: str = "SIMPLEC", relax_p=None, max_iter: int = 4000) -> dict:
    from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
    from flowtrace.unstructured.diagnostics import summary
    from flowtrace.unstructured.simple import SimpleSolver

    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.1,
                            algorithm=algorithm, relax_p=relax_p, max_iter=max_iter, tol=1e-6, backend="numpy")
    s = SimpleSolver(mesh(theta), case)
    t0 = time.perf_counter()
    diverged_at = None
    with np.errstate(all="ignore"):
        while not s.finished:
            s.step()
            raw = s.residuals()["raw"].get("continuity", 0.0)
            if not np.isfinite(raw) or raw > 1e6:
                diverged_at = s.iteration
                break
        d = s.diagnostics()
    sm = summary(d) if d.get("available") else {}
    return {"theta": theta, "algorithm": algorithm, "relax_p": relax_p, "converged": bool(s.converged),
            "iterations": s.iteration, "diverged_at": diverged_at, "rollbacks": len(s.guard_events),
            "final_relax_p": s.relax_p, "seconds": time.perf_counter() - t0,
            "non_orth_deg": s.fv.non_orthogonality_deg,
            "local_continuity_corrected": sm.get("continuity_corrected_local_max"), "finite": sm.get("finite")}


def label(r: dict) -> str:
    if r["algorithm"] == "SIMPLE":
        return "SIMPLE (0.3)"
    return "auto + guard" if r["relax_p"] is None else f"{r['relax_p']:g}"


def cell(r: dict) -> str:
    if r["converged"]:
        extra = f", {r['rollbacks']} rollback(s) -> {r['final_relax_p']:.2g}" if r["rollbacks"] else ""
        return f"ok {r['iterations']}{extra}"
    if r["diverged_at"]:
        return f"diverged @{r['diverged_at']}"
    return f"not converged ({r['iterations']})"


def render(rows: list, meta: dict) -> str:
    cols = [f"{a:g}" for a in RELAX] + ["auto + guard", "SIMPLE (0.3)"]
    by: dict = {}
    for r in rows:
        by.setdefault(r["theta"], {})[label(r)] = r
    L = [f"Measured {meta['date']} with FlowTrace {meta['flowtrace']} on {meta['cpu']} (Python {meta['python']}).",
         "Cavity Re = 10, 24 x 24 uniformly sheared quads; columns = pressure under-relaxation alpha_p.", "",
         "| theta | min orth. quality | cos^2 theta | " + " | ".join(cols) + " |",
         "|---" * (len(cols) + 3) + "|"]
    for th in sorted(by):
        d = by[th]
        oq = next(iter(d.values())).get("min_orthogonal_quality")
        L.append(f"| {th:g} | {oq:.3f} | {math.cos(math.radians(th)) ** 2:.2f} | "
                 + " | ".join(cell(d[c]) if c in d else "-" for c in cols) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    from flowtrace import __version__
    from flowtrace.hardware.probe import cached_profile
    from flowtrace.mesh.quality import mesh_quality

    rows = []
    for theta in ANGLES:
        oq = mesh_quality(mesh(theta)).min_orthogonal_quality
        for relax in RELAX:
            rows.append(dict(run(theta, relax_p=relax), min_orthogonal_quality=oq))
        rows.append(dict(run(theta), min_orthogonal_quality=oq))
        rows.append(dict(run(theta, algorithm="SIMPLE"), min_orthogonal_quality=oq))
        print(f"theta {theta:g}: " + "; ".join(f"{label(r)} {cell(r)}" for r in rows[-7:]), flush=True)
    meta = {"date": dt.datetime.now().isoformat(timespec="seconds"), "flowtrace": __version__,
            "python": platform.python_version(), "cpu": cached_profile().cpu_model}
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "skew_envelope.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=1)
    with open(os.path.join(out, "skew_envelope.md"), "w", encoding="utf-8") as fh:
        fh.write(render(rows, meta))


if __name__ == "__main__":
    main()
