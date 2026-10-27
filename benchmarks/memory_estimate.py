"""Check: is the pre-run memory estimate an upper bound for what a run really allocates?

Peak traced allocation (tracemalloc: NumPy, SciPy and pyamg arrays, native-kernel copies) of
mesh set-up, solver construction and 15 iterations, against hardware.memory.estimate_unstructured
(unstructured SIMPLEC) and estimate_structured (MAC), on the CPU. Criterion: estimate >= peak.
tracemalloc does not see memory that compiled code allocates itself, so on Linux every SIMPLEC
case is also run in a fresh process: after the imports and a warm-up solve that loads the native
kernels, the peak resident set is reset (/proc/self/clear_refs), and the growth of the peak
(VmHWM) over the resident set at that moment (VmRSS) is compared with the same estimate.
GPU memory (CuPy pool) is not measured here.

Run:  python benchmarks/memory_estimate.py   Writes benchmarks/results/memory_estimate.{json,md}.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def unstructured(mesh, energy=False, turbulence="laminar") -> dict:
    from flowtrace.hardware.memory import estimate_unstructured
    from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
    from flowtrace.unstructured.simple import SimpleSolver

    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, energy=energy,
                            turbulence=turbulence, max_iter=10 ** 6, tol=0, backend="numpy")
    tracemalloc.start()
    s = SimpleSolver(mesh, case)
    for _ in range(15):
        s.step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    est = estimate_unstructured(mesh.n_cells, mesh.n_faces, mesh.dim, 8, energy, "cpu", 0)
    return {"solver": "SIMPLEC", "case": f"{mesh.dim}-D, {mesh.n_cells:,} cells, {turbulence}" + (", energy" if energy else ""),
            "peak_MB": peak / 1e6, "estimate_MB": est.required / 1e6}


RSS_CHILD = r"""
import json, sys
from flowtrace.mesh.unstructured import cartesian_mesh_2d, extrude_to_3d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.simple import SimpleSolver

n, nz, energy, turbulence = json.loads(sys.argv[1])
def case(): return UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, energy=energy,
                                    turbulence=turbulence, max_iter=10 ** 6, tol=0, backend="numpy")
w = SimpleSolver(cartesian_mesh_2d(8, 8), case())  # loads native kernels, warms caches
for _ in range(3):
    w.step()
del w
mesh = cartesian_mesh_2d(n, n) if nz == 0 else extrude_to_3d(cartesian_mesh_2d(n, n), nz, 1.0)
def status(key):
    with open("/proc/self/status") as fh:
        return next(int(line.split()[1]) * 1024 for line in fh if line.startswith(key + ":"))
with open("/proc/self/clear_refs", "w") as fh:
    fh.write("5")  # reset the peak resident set to the current one
base = status("VmRSS")
s = SimpleSolver(mesh, case())
for _ in range(15):
    s.step()
print(json.dumps({"rss_growth_MB": (status("VmHWM") - base) / 1e6}))
"""


def rss_growth(n: int, nz: int = 0, energy: bool = False, turbulence: str = "laminar"):
    """Peak-RSS growth (MB) of a SIMPLEC case in a fresh process; None where Linux /proc is missing."""
    if not os.path.exists("/proc/self/clear_refs"):
        return None
    out = subprocess.run([sys.executable, "-c", RSS_CHILD, json.dumps([n, nz, energy, turbulence])],
                         capture_output=True, text=True, check=True, cwd=os.path.dirname(HERE))
    return json.loads(out.stdout.strip().splitlines()[-1])["rss_growth_MB"]


def structured(name, **params) -> dict:
    from flowtrace.hardware.backend import get_backend
    from flowtrace.hardware.memory import estimate_structured
    from flowtrace.structured.case import make_preset
    from flowtrace.structured.solver import MACSolver

    c = make_preset(name, **params)
    tracemalloc.start()
    s = MACSolver(c, backend=get_backend("numpy"), dtype=np.float64)
    for _ in range(10):
        s.step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    est = estimate_structured(c.nx, c.ny, c.energy, 8, "mg" if c.obstacles else "fft", True, bool(c.obstacles), "cpu", 0)
    return {"solver": "MAC", "case": f"{name} {c.nx}x{c.ny}", "peak_MB": peak / 1e6, "estimate_MB": est.required / 1e6}


def main() -> None:
    from flowtrace import __version__
    from flowtrace.hardware.probe import cached_profile
    from flowtrace.mesh.unstructured import cartesian_mesh_2d, extrude_to_3d

    rows = [dict(unstructured(cartesian_mesh_2d(n, n)), rss_MB=rss_growth(n)) for n in (64, 128, 256)]
    rows += [dict(unstructured(cartesian_mesh_2d(128, 128), energy=True), rss_MB=rss_growth(128, energy=True)),
             dict(unstructured(cartesian_mesh_2d(128, 128), turbulence="k-omega-sst"),
                  rss_MB=rss_growth(128, turbulence="k-omega-sst")),
             dict(unstructured(extrude_to_3d(cartesian_mesh_2d(32, 32), 32, 1.0)), rss_MB=rss_growth(32, nz=32))]
    rows += [structured("cavity", n=n) for n in (128, 256, 512)]
    rows += [structured("natural_convection", n=256), structured("cylinder", n_per_d=16)]
    for r in rows:
        r["estimate_over_peak"] = r["estimate_MB"] / r["peak_MB"]
        r["upper_bound"] = r["estimate_MB"] >= r["peak_MB"]
        if r.get("rss_MB") is not None:
            r["estimate_over_rss"] = r["estimate_MB"] / r["rss_MB"]
            r["upper_bound"] = r["upper_bound"] and r["estimate_MB"] >= r["rss_MB"]
    meta = {"date": dt.datetime.now().isoformat(timespec="seconds"), "flowtrace": __version__,
            "python": platform.python_version(), "cpu": cached_profile().cpu_model}
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "memory_estimate.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=1)
    L = [f"Measured {meta['date']} with FlowTrace {meta['flowtrace']} on {meta['cpu']} (CPU). Peak allocated: "
         "tracemalloc. Peak RSS growth: VmHWM - VmRSS of a fresh process whose peak was reset after the imports "
         "and a warm-up solve (Linux).", "",
         "| Solver | Case | Peak allocated | Peak RSS growth | Estimate | Estimate / peak | Estimate / RSS growth | "
         "Upper bound |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['solver']} | {r['case']} | {r['peak_MB']:.1f} MB | "
          + (f"{r['rss_MB']:.1f} MB" if r.get("rss_MB") is not None else "-")
          + f" | {r['estimate_MB']:.1f} MB | {r['estimate_over_peak']:.2f} | "
          + (f"{r['estimate_over_rss']:.2f}" if r.get("estimate_over_rss") is not None else "-")
          + f" | {'yes' if r['upper_bound'] else '**no**'} |" for r in rows]
    with open(os.path.join(out, "memory_estimate.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
