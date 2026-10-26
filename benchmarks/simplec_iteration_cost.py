"""Cost of one SIMPLEC iteration on the CPU and on the GPU, by mesh size.

Lid-driven cavity at Re = 100 on uniform quads, FP64, the same solver set-up the
planner uses for its calibration. For every size and backend: build the solver,
run 3 warm-up iterations (JIT, AMG set-up, CUDA kernel compilation), then report
the median of up to 20 device-synchronised iterations (at most ~10 s of timing).

Run:  python benchmarks/simplec_iteration_cost.py [--sizes 32,64,128,256,512]
Writes benchmarks/results/simplec_iteration_cost.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(sizes, out_dir):
    from flowtrace.hardware.backend import available_backends, get_backend
    from flowtrace.hardware.planner import _cavity_solver, time_iterations
    from flowtrace.hardware.probe import cached_profile

    backends = ["numpy"] + (["cupy"] if "cupy" in available_backends() else [])
    rows = []
    for n in sizes:
        row = {"n": n, "cells": n * n}
        for name in backends:
            be = get_backend(name)
            s = _cavity_solver(n, be, "fp64")
            ts = time_iterations(s, 20, warmup=3, budget_s=10.0)
            row[name] = 1e3 * statistics.median(ts)
            print(f"{n * n:>8} cells  {name:>5}  {row[name]:8.1f} ms/iteration ({len(ts)} timed)", flush=True)
            del s
            be.free_pools()
        rows.append(row)
    p = cached_profile()
    d = {"date": dt.datetime.now().isoformat(timespec="seconds"), "cpu": p.cpu_model,
         "gpu": p.gpus[0].name if p.gpus and "cupy" in backends else None, "backends": backends, "rows": rows,
         "reproduce": "python benchmarks/simplec_iteration_cost.py --sizes " + ",".join(map(str, sizes))}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "simplec_iteration_cost.json"), "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    with open(os.path.join(out_dir, "simplec_iteration_cost.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(d))
    return d


def render_markdown(d) -> str:
    label = {"numpy": "CPU (NumPy)", "cupy": "GPU (CuPy)"}
    lines = ["| cells | " + " | ".join(f"{r['cells']:,}".replace(",", " ") for r in d["rows"]) + " |",
             "|---|" + "---:|" * len(d["rows"])]
    for b in d["backends"]:
        lines.append(f"| {label[b]} | " + " | ".join(f"{r[b]:.1f}" for r in d["rows"]) + " |")
    gpu = f" and {d['gpu']}" if d["gpu"] else ""
    lines += ["", f"_ms per iteration, FP64, median of up to 20 iterations; measured {d['date']} on "
                  f"{d['cpu']}{gpu}. Reproduce: `{d['reproduce']}`._", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="32,64,128,256,512")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    a = ap.parse_args()
    print(render_markdown(run([int(x) for x in a.sizes.split(",")], a.out)))
