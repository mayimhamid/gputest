"""Experiment: does the measured planner place the unstructured SIMPLEC solver well?

Hypothesis. A SIMPLEC iteration on the GPU has a size-independent floor (kernel
launches and host syncs), so the GPU loses below some mesh size and wins above
it. A planner that times the CPU, skips the GPU when the CPU beats the GPU's
floor and otherwise times both, should match the faster fixed choice.

Set-up. Lid-driven cavity, Re = 100, uniform quads n x n, FP64, a fixed number
of SIMPLEC iterations (same work for every configuration, tol = 0). Each run is
a fresh Python process; the time covers device initialisation, mesh generation,
solver set-up, the planner's timing iterations and all iterations (imports excluded).

Configurations: ``numpy`` (fixed CPU), ``cupy`` (fixed GPU, what ``auto`` did
before), ``auto`` (the planner). Oracle = the faster of the two fixed runs.

Acceptance criteria (fixed before the first run):
  A1  the planner chooses the oracle's device whenever the oracle is > 15 % faster;
  A2  planner time <= 1.15 x oracle time at every size (median of the repeats);
  A3  the planner's final field agrees with the fixed run on the same device to
      within 10 x the fixed run's own repeat-to-repeat difference (floor 1e-10),
      i.e. the timing iterations change nothing beyond run-to-run noise.
Rejection: any failure of A1-A3.

A3 was first written as max |du| <= 1e-10. A smoke run (before the measured
study) showed that two fixed NumPy runs in separate processes already differ by
~6e-8 after 20 iterations (within one process the solver is bit-exact), so the
baseline itself could not meet that threshold; it was replaced as above before
any study numbers were taken.

Run:  python benchmarks/planner_unstructured.py [--sizes 32,64,...] [--iters 100] [--repeats 3]
Writes benchmarks/results/planner_unstructured.{json,md}.
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
from flowtrace.hardware.planner import plan_unstructured
from flowtrace.mesh.unstructured import cartesian_mesh_2d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
n, iters, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t0 = time.perf_counter()  # device initialisation (CUDA context) is inside the timer for every mode
mesh = cartesian_mesh_2d(n, n)
case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                        max_iter=iters, tol=0.0, backend=mode)
plan, s = plan_unstructured(mesh, case)
t_plan = time.perf_counter() - t0
while not s.finished:
    s.step()
s.be.synchronize()
total = time.perf_counter() - t0
u = s.fields()["x-velocity"]
peak_rss = None
try:
    import psutil
    mi = psutil.Process().memory_info()
    peak_rss = getattr(mi, "peak_wset", None) or getattr(mi, "rss", None)
except Exception:
    pass
gpu_pool = int(s.be.xp.get_default_memory_pool().total_bytes()) if s.be.is_gpu else 0
np.save(sys.argv[4], u)
print(json.dumps({"n": n, "cells": mesh.n_cells, "mode": mode, "backend": plan.backend, "device": plan.device,
                  "reason": plan.reason, "measured_ms": plan.measured_ms_per_iteration, "skipped": plan.skipped,
                  "plan_seconds": t_plan, "total_seconds": total, "iterations": s.iteration,
                  "peak_rss_bytes": peak_rss, "gpu_pool_bytes": gpu_pool}))
"""


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _environment():
    from flowtrace import __version__
    from flowtrace.hardware.planner import calibrate_unstructured
    from flowtrace.hardware.probe import cached_profile

    p = cached_profile()
    libs = {k: v for k, v in p.libraries.items() if v}
    return {"flowtrace": __version__, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")), "python": platform.python_version(),
            "os": platform.platform(), "cpu": p.cpu_model, "logical_cores": p.logical_cores,
            "ram_bytes": p.ram_total, "gpus": [{"name": g.name, "driver": g.driver, "vram_bytes": g.vram_total,
                                                "compute_capability": g.compute_capability} for g in p.gpus],
            "libraries": libs, "on_battery": p.on_battery, "calibration": calibrate_unstructured()}


def run(sizes, iters, repeats, out_dir):
    import numpy as np

    tmp = os.path.join(out_dir, "_fields")
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for n in sizes:
        for mode in ("numpy", "cupy", "auto"):
            for r in range(repeats):
                f = os.path.join(tmp, f"u_{n}_{mode}_{r}.npy")
                cp = subprocess.run([sys.executable, "-c", CHILD, str(n), str(iters), mode, f], cwd=ROOT,
                                    capture_output=True, text=True)
                if cp.returncode != 0:
                    rows.append({"n": n, "mode": mode, "repeat": r, "error": cp.stderr[-2000:]})
                    print(f"n={n} {mode}: FAILED\n{cp.stderr[-800:]}", flush=True)
                    continue
                rec = json.loads(cp.stdout.strip().splitlines()[-1])
                rec["repeat"], rec["field_file"] = r, f
                rows.append(rec)
                print(f"n={n:>4} {mode:>5} -> {rec['backend']:>5} {rec['total_seconds']:8.2f} s "
                      f"(plan {rec['plan_seconds']:.2f} s)", flush=True)
    # summarise per size
    summary = []
    for n in sizes:
        def med(mode):
            v = [x["total_seconds"] for x in rows if x["n"] == n and x["mode"] == mode and "error" not in x]
            return (statistics.median(v), (max(v) - min(v)) if len(v) > 1 else 0.0) if v else (None, None)
        (tc, sc), (tg, sg), (ta, sa) = med("numpy"), med("cupy"), med("auto")
        autos = [x for x in rows if x["n"] == n and x["mode"] == "auto" and "error" not in x]
        if None in (tc, tg, ta) or not autos:
            summary.append({"n": n, "error": "a configuration failed"})
            continue
        oracle_dev = "cpu" if tc <= tg else "gpu"
        oracle = min(tc, tg)
        chosen = sorted({a["device"] for a in autos})
        a1 = (max(tc, tg) / oracle <= 1.15) or chosen == [oracle_dev]
        a2 = ta <= 1.15 * oracle
        a3, du_auto, du_noise = True, 0.0, 0.0
        for a in autos:
            refs = [x for x in rows if x["n"] == n and x["mode"] == ("cupy" if a["device"] == "gpu" else "numpy")
                    and "error" not in x]
            fields = [np.load(x["field_file"]) for x in refs]
            noise = max((float(np.abs(f - fields[0]).max()) for f in fields[1:]), default=0.0)
            du = float(np.abs(np.load(a["field_file"]) - fields[0]).max())
            du_auto, du_noise = max(du_auto, du), max(du_noise, noise)
            a3 &= du <= max(1e-10, 10.0 * noise)
        cells = autos[0]["cells"]
        summary.append({"n": n, "cells": cells, "cpu_s": tc, "cpu_spread_s": sc, "gpu_s": tg, "gpu_spread_s": sg,
                        "auto_s": ta, "auto_spread_s": sa, "oracle_device": oracle_dev, "planner_device": chosen,
                        "auto_over_oracle": ta / oracle, "old_auto_over_oracle": tg / oracle,
                        "A1": a1, "A2": a2, "A3": a3, "max_du_planner_vs_fixed": du_auto,
                        "max_du_fixed_repeats": du_noise, "reason": autos[0]["reason"],
                        "peak_rss_bytes": max((a["peak_rss_bytes"] or 0) for a in autos),
                        "gpu_pool_bytes": max(x.get("gpu_pool_bytes", 0) for x in rows
                                              if x["n"] == n and "error" not in x)})
    for x in rows:
        x.pop("field_file", None)
    data = {"experiment": "planner_unstructured", "date": dt.datetime.now().isoformat(timespec="seconds"),
            "case": "lid-driven cavity Re=100, uniform quads, SIMPLEC, FP64, second-order",
            "iterations": iters, "repeats": repeats, "stopping": f"fixed {iters} iterations (tol = 0)",
            "environment": _environment(),
            "reproduce": f"python benchmarks/planner_unstructured.py --sizes {','.join(map(str, sizes))} "
                         f"--iters {iters} --repeats {repeats}",
            "summary": summary, "runs": rows}
    data["passed"] = all(s.get("A1") and s.get("A2") and s.get("A3") for s in summary)
    with open(os.path.join(out_dir, "planner_unstructured.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    with open(os.path.join(out_dir, "planner_unstructured.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(data))
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return data


def render_markdown(d) -> str:
    env = d["environment"]
    gpu = ", ".join(f"{g['name']} (driver {g['driver']})" for g in env["gpus"]) or "none"
    cal = env["calibration"]
    lines = [f"Measured {d['date']} on {env['cpu']} ({env['logical_cores']} threads) + {gpu}; "
             f"FlowTrace {env['flowtrace']} @ {env['git_commit'][:10]}{' (uncommitted changes)' if env['git_dirty'] else ''}, "
             f"Python {env['python']}, {env['os']}.",
             f"Case: {d['case']}; {d['stopping']}; median of {d['repeats']} fresh processes (spread = max - min).",
             f"Calibration: GPU floor {cal.get('gpu_floor_ms', float('nan')):.1f} ms/iteration, "
             f"CPU {cal['cpu_us_per_cell']:.2f} us/cell/iteration.", "",
             "| cells | CPU (s) | GPU = old auto (s) | planner (s) | planner chose | oracle | planner / oracle "
             "| old auto / oracle | A1 | A2 | A3 |",
             "|---:|---:|---:|---:|---|---|---:|---:|---|---|---|"]
    ok = lambda b: "pass" if b else "**FAIL**"  # noqa: E731
    for s in d["summary"]:
        if "error" in s:
            lines.append(f"| n={s['n']} | failed | | | | | | | | | |")
            continue
        lines.append(f"| {s['cells']:,} | {s['cpu_s']:.2f} ± {s['cpu_spread_s']:.2f} | "
                     f"{s['gpu_s']:.2f} ± {s['gpu_spread_s']:.2f} | "
                     f"{s['auto_s']:.2f} ± {s['auto_spread_s']:.2f} | {'/'.join(s['planner_device']).upper()} | "
                     f"{s['oracle_device'].upper()} | {s['auto_over_oracle']:.2f} | {s['old_auto_over_oracle']:.2f} | "
                     f"{ok(s['A1'])} | {ok(s['A2'])} | {ok(s['A3'])} |")
    lines += ["", f"Overall: {'all acceptance criteria met' if d['passed'] else 'at least one criterion FAILED'}.",
              "", f"Reproduce: `{d['reproduce']}`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="32,64,96,128,192,256")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    a = ap.parse_args()
    res = run([int(x) for x in a.sizes.split(",")], a.iters, a.repeats, a.out)
    print(render_markdown(res))
