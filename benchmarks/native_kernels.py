"""Experiment: native C++/OpenMP (g++) and CUDA kernels vs the NumPy/CuPy reference path.

Same SIMPLEC case, same iterations; only FLOWTRACE_NO_NATIVE differs. Every configuration runs
in a fresh process, and reference/native runs are interleaved round by round so that clock
drift on a laptop affects both alike. Reported: median ms per iteration over the rounds
(each round: 10 warm-up + 50 timed iterations) and the max field difference after 60 iterations.

Acceptance criteria (fixed before the run):
  N1  native and reference velocity fields agree after 60 iterations: max |du| / max |u| <= 1e-8;
  N2  no slow-down: native ms/it <= 1.05 x reference ms/it at every size and device.

Run:  python benchmarks/native_kernels.py [--sizes 128,256,512] [--mesh path.msh ...] [--rounds 3]
Writes benchmarks/results/native_kernels.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CHILD = r"""
import json, sys, time
import numpy as np
from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.fluent_h5 import read_any_mesh
from flowtrace.mesh.unstructured import cartesian_mesh_2d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
from flowtrace.unstructured.simple import SimpleSolver
src, backend, out = sys.argv[1], sys.argv[2], sys.argv[3]
be = get_backend(backend)
if src.startswith("cavity"):
    mesh = cartesian_mesh_2d(int(src[6:]), int(src[6:]))
    bnd = {"top": BoundarySpec("wall", velocity=[1.0, 0.0])}
    mu = 0.01
else:
    mesh = read_any_mesh(src)[0]
    bnd = default_boundaries(mesh)
    mu = 0.1
s = SimpleSolver(mesh, UnstructuredCase(boundaries=bnd, viscosity=mu, max_iter=10 ** 6, tol=0.0, backend=backend),
                 backend=be)
for _ in range(10):
    s.step()
ts = []
for _ in range(50):
    t0 = be.timer(); s.step(); ts.append(be.timer() - t0)
np.save(out, s.fields()["x-velocity"])
fv = getattr(s, "fv")
print(json.dumps({"cells": mesh.n_cells, "ms": 1e3 * float(np.median(ts)), "native": fv.nat is not None,
                  "cuda": fv.cu is not None, "stages": {k: 1e3 * v / 60 for k, v in s.timers.items()}}))
"""


def _run(src, backend, native, out):
    env = dict(os.environ)
    env.pop("FLOWTRACE_NO_NATIVE", None)
    if not native:
        env["FLOWTRACE_NO_NATIVE"] = "1"
    cp = subprocess.run([sys.executable, "-W", "ignore", "-c", CHILD, src, backend, out], cwd=ROOT, env=env,
                        capture_output=True, text=True)
    if cp.returncode:
        raise RuntimeError(cp.stderr[-3000:])
    return json.loads(cp.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="128,256,512", help="lid-driven cavity n x n quads")
    ap.add_argument("--mesh", action="append", default=[], help="Fluent meshes (default boundary types, mu = 0.1)")
    ap.add_argument("--backends", default="numpy,cupy")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    import numpy as np

    from flowtrace import __version__, native
    from flowtrace.hardware.backend import available_backends

    srcs = [f"cavity{n}" for n in a.sizes.split(",") if n] + [os.path.abspath(m) for m in a.mesh]
    backends = [b for b in a.backends.split(",") if b in available_backends()]
    tmp = os.path.join(a.out, "_native_fields")
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for src in srcs:
        for b in backends:
            ref, nat = [], []
            for r in range(a.rounds):
                ref.append(_run(src, b, False, os.path.join(tmp, f"ref{r}.npy")))
                nat.append(_run(src, b, True, os.path.join(tmp, f"nat{r}.npy")))
            u_ref, u_nat = np.load(os.path.join(tmp, "ref0.npy")), np.load(os.path.join(tmp, "nat0.npy"))
            du = float(np.abs(u_nat - u_ref).max() / max(np.abs(u_ref).max(), 1e-300))
            t_ref = statistics.median(x["ms"] for x in ref)
            t_nat = statistics.median(x["ms"] for x in nat)
            row = {"case": os.path.basename(src).split(".")[0], "cells": ref[0]["cells"], "backend": b,
                   "kernels": "C++/OpenMP" if nat[0]["native"] else ("CUDA" if nat[0]["cuda"] else "none"),
                   "ref_ms": t_ref, "native_ms": t_nat, "speedup": t_ref / t_nat,
                   "ref_ms_rounds": [x["ms"] for x in ref], "native_ms_rounds": [x["ms"] for x in nat],
                   "ref_stages": ref[0]["stages"], "native_stages": nat[0]["stages"], "max_rel_du": du,
                   "N1": du <= 1e-8, "N2": t_nat <= 1.05 * t_ref}
            rows.append(row)
            print(f"{row['case']:<24} {b:<6} {row['kernels']:<11} ref {t_ref:8.2f} ms  native {t_nat:8.2f} ms  "
                  f"x{row['speedup']:.2f}  du {du:.1e}", flush=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, dirty = "", False
    st = native.status()
    data = {"experiment": "native_kernels", "date": dt.datetime.now().isoformat(timespec="seconds"),
            "flowtrace": __version__, "git_commit": commit, "git_dirty": dirty, "python": platform.python_version(),
            "os": platform.platform(), "cpu": platform.processor(), "logical_cores": os.cpu_count(),
            "native": {k: v for k, v in st.items() if k != "path"}, "rounds": a.rounds, "rows": rows,
            "passed": all(r["N1"] and r["N2"] for r in rows),
            "reproduce": "python benchmarks/native_kernels.py --sizes " + a.sizes
                         + "".join(f" --mesh {os.path.basename(m)}" for m in a.mesh)}
    md = [f"Measured {data['date']} on {data['cpu']} ({data['logical_cores']} threads, OpenMP {st.get('threads')}); "
          f"FlowTrace {__version__} @ {commit[:10]}{' (uncommitted changes)' if dirty else ''}, Python "
          f"{data['python']}. Median of {a.rounds} interleaved rounds of 50 SIMPLEC iterations, FP64.", "",
          "| case | cells | device | kernels | reference ms/it | native ms/it | speed-up | max rel. du | N1 | N2 |",
          "|---|---:|---|---|---:|---:|---:|---:|---|---|"]
    for r in rows:
        md.append(f"| {r['case']} | {r['cells']:,} | {'GPU' if r['backend'] == 'cupy' else 'CPU'} | {r['kernels']} | "
                  f"{r['ref_ms']:.2f} | {r['native_ms']:.2f} | {r['speedup']:.2f}x | {r['max_rel_du']:.1e} | "
                  f"{'pass' if r['N1'] else '**FAIL**'} | {'pass' if r['N2'] else '**FAIL**'} |")
    md += ["", f"Overall: {'all criteria met' if data['passed'] else 'at least one criterion FAILED'}.", "",
           f"Reproduce: `{data['reproduce']}`", ""]
    with open(os.path.join(a.out, "native_kernels.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    with open(os.path.join(a.out, "native_kernels.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
