"""Experiment: wall-clock time of a multi-case study, sequential vs ``flowtrace batch``.

Sequential = what a user does without the batch runner: each case solved in turn in a fresh
process with ``backend="auto"`` (the measured planner). Batch = ``run_batch`` (GPU worker +
one CPU case per remaining physical core). Same cases, same tolerances; the per-case drag
from both modes must agree to round-off (criterion B2) and the batch must be faster (B1).

Acceptance criteria (fixed before the run):
  B1  batch wall time < sequential wall time;
  B2  every case's force on the named wall zone agrees between the modes to 1e-5 (relative).

Run:  python benchmarks/batch_study.py MESH [MESH ...] --zone BluffWalls --mu 0.1 [--name NAME]
Writes benchmarks/results/<NAME>.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CHILD = r"""
import json, sys, time
from flowtrace.hardware.planner import plan_unstructured
from flowtrace.mesh.fluent_h5 import read_any_mesh
from flowtrace.unstructured.case import UnstructuredCase, default_boundaries
t0 = time.perf_counter()
mesh, _ = read_any_mesh(sys.argv[1])
case = UnstructuredCase(boundaries=default_boundaries(mesh), **json.loads(sys.argv[2]))
plan, s = plan_unstructured(mesh, case)
while not s.finished:
    s.step()
print(json.dumps({"backend": plan.backend, "iterations": s.iteration, "converged": bool(s.converged),
                  "seconds": time.perf_counter() - t0, "wall_forces": s.wall_forces()}))
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--zone", required=True, help="wall zone whose force is compared (e.g. BluffWalls)")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=8000)
    ap.add_argument("--name", default="batch_study")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    from flowtrace import __version__
    from flowtrace.runtime.batch import BatchJob, run_batch

    case = {"viscosity": a.mu, "tol": a.tol, "max_iter": a.max_iter}
    names = [os.path.basename(m).split(".")[0] for m in a.meshes]
    t0 = time.perf_counter()
    seq = {}
    for n, m in zip(names, a.meshes):
        cp = subprocess.run([sys.executable, "-W", "ignore", "-c", CHILD, os.path.abspath(m), json.dumps(case)],
                            cwd=ROOT, capture_output=True, text=True)
        if cp.returncode:
            raise SystemExit(cp.stderr[-2000:])
        seq[n] = json.loads(cp.stdout.strip().splitlines()[-1])
        print(f"sequential {n}: {seq[n]['backend']} {seq[n]['iterations']} its {seq[n]['seconds']:.1f} s", flush=True)
    t_seq = time.perf_counter() - t0
    res = run_batch([BatchJob(name=n, mesh=os.path.abspath(m), case=case) for n, m in zip(names, a.meshes)])
    t_batch = res["wall_seconds"]
    rows, b2 = [], True
    for n, r in zip(names, res["results"]):
        fs, fb = seq[n]["wall_forces"][a.zone]["total"][0], r["wall_forces"][a.zone]["total"][0]
        ok = abs(fb - fs) <= 1e-5 * abs(fs)
        b2 &= ok
        rows.append({"case": n, "cells": r["cells"], "seq_backend": seq[n]["backend"], "seq_seconds": seq[n]["seconds"],
                     "seq_iterations": seq[n]["iterations"], "batch_device": r["device"], "batch_seconds": r["seconds"],
                     "batch_iterations": r["iterations"], "force_seq": fs, "force_batch": fb, "B2": ok})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, dirty = "", False
    data = {"experiment": a.name, "date": dt.datetime.now().isoformat(timespec="seconds"), "case": case,
            "zone": a.zone, "meshes": [os.path.abspath(m) for m in a.meshes], "flowtrace": __version__,
            "git_commit": commit, "git_dirty": dirty, "python": platform.python_version(), "os": platform.platform(),
            "cpu": platform.processor(), "logical_cores": os.cpu_count(), "batch_plan": res["plan"],
            "sequential_seconds": t_seq, "batch_seconds": t_batch, "speedup": t_seq / t_batch,
            "B1": t_batch < t_seq, "B2": b2, "rows": rows,
            "reproduce": "python benchmarks/batch_study.py " + " ".join(a.meshes) + f" --zone {a.zone} --mu {a.mu}"}
    md = [f"Measured {data['date']} on {data['cpu']} ({data['logical_cores']} threads); FlowTrace {__version__} @ "
          f"{commit[:10]}{' (uncommitted changes)' if dirty else ''}. Viscosity {a.mu}, tol {a.tol}.", "",
          "| case | cells | sequential: device, its, s | batch: device, its, s | force (seq) | force (batch) | B2 |",
          "|---|---:|---|---|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['case']} | {r['cells']:,} | {r['seq_backend']}, {r['seq_iterations']}, {r['seq_seconds']:.1f} | "
                  f"{r['batch_device']}, {r['batch_iterations']}, {r['batch_seconds']:.1f} | {r['force_seq']:.6f} | "
                  f"{r['force_batch']:.6f} | {'pass' if r['B2'] else '**FAIL**'} |")
    md += ["", f"Whole study: sequential {t_seq:.1f} s, batch {t_batch:.1f} s, speed-up {t_seq / t_batch:.2f}x "
           f"(B1 {'pass' if data['B1'] else '**FAIL**'}).", "", f"Reproduce: `{data['reproduce']}`", ""]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, a.name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    with open(os.path.join(a.out, a.name + ".md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
