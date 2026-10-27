"""Experiment: is FP32 accurate enough for the structured MAC solver, and can it converge?

Set-up. Built-in steady presets (cavity Re = 100 at 32^2, 64^2, 128^2; cavity Re = 1000 at
128^2; natural convection Ra = 1e4 and 1e5 at 64^2) on the CPU NumPy backend, once in FP64
and once in FP32 (the same IEEE single-precision arithmetic as a GPU's FP32 path, up to
summation order). residuals() is polled every monitor_interval steps, as the engine does,
because it also drives the steady-state test. FP32 gets 3x the FP64 step count.

Questions. (1) How far is the FP32 field from the FP64 one, compared with the FP64
discretisation error against the published reference? (2) Does FP32 reach the case's steady
tolerance, and where does its normalised residual stall? The stall level is compared with the
bound u32 L / (U dt), u32 = 2^-24 (the round-off of one step's velocity change, scaled the way
the residual is).

Run:  python benchmarks/precision_fp32.py   (about 15 min on 4 cores)
Writes benchmarks/results/precision_fp32.{json,md}.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("cavity", {"re": 100, "n": 32}), ("cavity", {"re": 100, "n": 64}), ("cavity", {"re": 100, "n": 128}),
         ("cavity", {"re": 1000, "n": 128}), ("natural_convection", {"ra": 1e4, "n": 64}),
         ("natural_convection", {"ra": 1e5, "n": 64})]
U32 = 2.0 ** -24


def run(name: str, params: dict, dtype, max_steps=None):
    from flowtrace.hardware.backend import get_backend
    from flowtrace.structured.case import make_preset
    from flowtrace.structured.solver import MACSolver
    from flowtrace.validation.checks import validate_structured

    c = make_preset(name, **params)
    if max_steps:
        c.max_steps = max_steps
    s = MACSolver(c, backend=get_backend("numpy"), dtype=dtype)
    t0 = time.perf_counter()
    floor = float("inf")
    with np.errstate(all="ignore"):
        while not s.finished:
            s.step()
            if s.step_count % c.monitor_interval == 0:
                floor = min(floor, max(s.residuals()["normalized"].values()))
    checks = [{k: ch[k] for k in ("name", "value") if k in ch} for ch in validate_structured(s)["checks"]]
    L = min(c.lx, c.ly)
    return s, {"steps": s.step_count, "converged": bool(s.converged), "min_normalized_residual": floor,
               "dt": float(s.dt), "round_off_bound": U32 * L / (c.u_ref * float(s.dt)),
               "seconds": time.perf_counter() - t0, "checks": checks}


def main() -> None:
    from flowtrace import __version__
    from flowtrace.hardware.probe import cached_profile

    rows = []
    for name, params in CASES:
        s64, r64 = run(name, params, np.float64)
        s32, r32 = run(name, params, np.float32, max_steps=max(3 * r64["steps"], 2000))
        f64, f32 = s64.fields(), s32.fields()
        diff = {k: float(np.abs(np.asarray(f64[k], float) - np.asarray(f32[k], float)).max()
                         / max(float(np.abs(np.asarray(f64[k], float)).max()), 1e-300))
                for k in ("x-velocity", "y-velocity", "temperature") if k in f64 and k in f32}
        rows.append({"case": name, "params": params, "steady_tol": s64.case.steady_tol, "fp64": r64, "fp32": r32,
                     "fp32_vs_fp64_max_rel": diff})
        print(json.dumps(rows[-1]), flush=True)
    meta = {"date": dt.datetime.now().isoformat(timespec="seconds"), "flowtrace": __version__,
            "python": platform.python_version(), "cpu": cached_profile().cpu_model}
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "precision_fp32.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=1)
    L = [f"Measured {meta['date']} with FlowTrace {meta['flowtrace']} on {meta['cpu']} (Python {meta['python']}), "
         "CPU NumPy backend.", "",
         "| Case | FP32 vs FP64 field (max rel.) | Validated quantity FP64 / FP32 | Steady tol. | FP32 reached it? "
         "| FP32 residual floor | u32 L/(U dt) | floor / bound |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f64, f32 = r["fp64"], r["fp32"]
        q64 = {c["name"]: c["value"] for c in f64["checks"]}
        q32 = {c["name"]: c["value"] for c in f32["checks"]}
        key = next(k for k in q64 if "continuity" not in k)
        conv = "yes" if f32["converged"] else f"no ({f32['steps']:,} steps vs {f64['steps']:,} for FP64)"
        L.append(f"| {r['case']} {', '.join(f'{k}={v:g}' for k, v in r['params'].items())} | "
                 f"{max(r['fp32_vs_fp64_max_rel'].values()):.1e} | {key}: {q64[key]:.5g} / {q32[key]:.5g} | "
                 f"{r['steady_tol']:g} | {conv} | {f32['min_normalized_residual']:.2e} | {f32['round_off_bound']:.2e} | "
                 f"{f32['min_normalized_residual'] / f32['round_off_bound']:.2f} |")
    with open(os.path.join(out, "precision_fp32.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
