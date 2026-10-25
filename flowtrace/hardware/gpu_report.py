"""GPU validation log: does the CuPy path give the CPU answer, and when is it faster?

``flowtrace gpu-report`` runs on the machine at hand and writes
``gpu_<host>.json`` and ``gpu_<host>.md``:

* what the probe sees (GPU, driver, CUDA runtime, VRAM);
* micro-benchmarks of every backend (bandwidth, GEMM, DCT, launch latency, PCIe);
* correctness: MAC and SIMPLEC solvers on CuPy vs NumPy (FP64), max difference
  after a fixed number of steps, and a short FP32 run;
* speed: measured ms/step of the MAC solver per backend over a range of grid
  sizes, and the planner's choice for each size.

Without a usable GPU the report says so and records only the CPU numbers;
it never invents GPU results.
"""

from __future__ import annotations

import json
import os
import platform
import time

import numpy as np


def _cuda_versions(be) -> dict:
    try:
        rt = be.xp.cuda.runtime
        return {"runtime": rt.runtimeGetVersion(), "driver": rt.driverGetVersion(),
                "cupy": be.xp.__version__}
    except Exception as exc:
        return {"error": str(exc)}


def _time_steps(solver, steps: int, repeats: int = 3) -> float:
    """ms per step: warm-up (JIT/NVRTC compilation, clock ramp-up), then best of ``repeats`` blocks."""
    be = solver.be
    for _ in range(15 if be.is_gpu else 3):
        solver.step()
    best = float("inf")
    for _ in range(repeats):
        t0 = be.timer()
        for _ in range(steps):
            solver.step()
        best = min(best, (be.timer() - t0) / steps * 1e3)
    return best


def correctness(gpu_be, cpu_be) -> dict:
    from ..mesh.unstructured import cartesian_mesh_2d
    from ..structured.case import make_preset
    from ..structured.solver import MACSolver
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    out = {}
    for preset, params in (("cavity", {"n": 64}), ("natural_convection", {"n": 64}), ("cylinder", {"n_per_d": 8}),
                           ("rayleigh_benard", {"nx": 64, "ny": 32})):
        c = make_preset(preset, **params)
        c.precision = "fp64"
        a = MACSolver(c, backend=cpu_be, dtype=np.float64)
        b = MACSolver(c, backend=gpu_be, dtype=np.float64)
        for _ in range(50):
            a.step()
            b.step()
        du = float(np.abs(a.U - gpu_be.to_numpy(b.U)).max())
        out[f"MAC {preset} fp64, 50 steps: max|u_gpu - u_cpu|"] = du
        out[f"MAC {preset} fp64: max|div u| on GPU"] = b.max_divergence()
    c = make_preset("cavity", n=128)
    c.precision = "fp32"
    s = MACSolver(c, backend=gpu_be, dtype=np.float32)
    for _ in range(200):
        s.step()
    out["MAC cavity fp32, 200 steps: max|div u|"] = s.max_divergence()
    m = cartesian_mesh_2d(32, 32, triangulate=True, warp=0.05)
    uc = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                          max_iter=100, tol=1e-12, pressure_tol=1e-9)
    a = SimpleSolver(m, uc, backend=cpu_be)
    b = SimpleSolver(m, uc, backend=gpu_be)
    for _ in range(100):
        a.step()
        b.step()
    out["SIMPLEC cavity tri fp64, 100 its: max|u_gpu - u_cpu|"] = float(
        np.abs(a.fields()["x-velocity"] - b.fields()["x-velocity"]).max())
    return out


def speed(backends, sizes=(64, 128, 256, 512, 1024)) -> dict:
    from ..hardware.backend import get_backend
    from ..hardware.planner import plan_structured
    from ..structured.case import lid_driven_cavity
    from ..structured.solver import MACSolver, resolve_dtype

    table = {}
    for n in sizes:
        row = {}
        for name in backends:
            be = get_backend(name)
            c = lid_driven_cavity(re=1000, n=n)
            try:
                s = MACSolver(c, backend=be, dtype=resolve_dtype(c, be))
                steps = 20 if n <= 256 else (8 if n <= 512 else 4)
                row[name] = {"ms_per_step": round(_time_steps(s, steps), 3), "precision": str(s.dtype),
                             "pressure": s.poisson_detail}
                if s.use_cuda:  # same backend with the vectorised CuPy kernels, for comparison
                    s.use_cuda = False
                    row[name]["ms_per_step_vectorised"] = round(_time_steps(s, steps), 3)
                    row[name]["kernels"] = "fused CUDA C++ (NVRTC)"
                del s
                be.free_pools()
            except Exception as exc:
                row[name] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            plan, _ = plan_structured(lid_driven_cavity(re=1000, n=n), autotune=True, keep_solver=False)
            row["planner_choice"] = plan.backend
        except Exception as exc:
            row["planner_choice"] = f"error: {exc}"
        table[str(n)] = row
    return table


def run_gpu_report(out_dir: str = "benchmarks/results", sizes=(64, 128, 256, 512, 1024),
                   verbose: bool = True) -> dict:
    from ..hardware.backend import BackendUnavailable, available_backends, get_backend
    from ..hardware.bench import run_benchmarks
    from ..hardware.probe import probe_system
    from ..validation.benchmarks import _clean, machine_info

    prof = probe_system()
    backends = available_backends()
    data = {"machine": machine_info(), "gpus": [g.__dict__ for g in prof.gpus], "backends": backends,
            "notes": list(prof.notes)}
    gpu_be = None
    try:
        gpu_be = get_backend("cupy")
        data["cuda"] = _cuda_versions(gpu_be)
    except BackendUnavailable as exc:
        data["gpu_status"] = f"no usable CuPy GPU backend: {exc}"
    if verbose:
        print("[gpu-report] micro-benchmarks ...", flush=True)
    data["micro"] = run_benchmarks(backends, quick=True, use_cache=False)["results"]
    if gpu_be is not None:
        if verbose:
            print("[gpu-report] GPU vs CPU correctness ...", flush=True)
        t0 = time.perf_counter()
        data["correctness"] = correctness(gpu_be, get_backend("numpy"))
        data["correctness_seconds"] = time.perf_counter() - t0
    if verbose:
        print("[gpu-report] MAC ms/step per backend and grid ...", flush=True)
    data["speed"] = speed(backends, sizes)
    data = _clean(data)
    os.makedirs(out_dir, exist_ok=True)
    host = (platform.node() or "host").replace(" ", "_")
    with open(os.path.join(out_dir, f"gpu_{host}.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    with open(os.path.join(out_dir, f"gpu_{host}.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(data))
    return data


def render_markdown(data: dict) -> str:
    m = data["machine"]
    L = [f"## GPU log - {m.get('gpu') or 'no GPU'}", "",
         f"Measured {m.get('date')} on {m.get('cpu')} ({m.get('cores')}), {m.get('platform')}.", ""]
    if data.get("cuda"):
        c = data["cuda"]
        L.append(f"CuPy {c.get('cupy')}, CUDA runtime {c.get('runtime')}, driver API {c.get('driver')}.")
        L.append("")
    if data.get("gpu_status"):
        L += [f"**GPU path not exercised:** {data['gpu_status']}", ""]
    L += ["### Micro-benchmarks", "", "| Backend | Bandwidth GB/s | GEMM FP64 GF/s | GEMM FP32 GF/s | DCT 512² ms | "
          "Stencil Mcell/s | Launch µs | H2D GB/s |", "|---|---|---|---|---|---|---|---|"]
    for k, r in data["micro"].items():
        if "error" in r:
            L.append(f"| {k} | error: {r['error']} |||||||")
            continue
        L.append(f"| {k} | {r['bandwidth_gbs']} | {r['gemm_gflops_fp64']} | {r['gemm_gflops_fp32']} | {r['dct512_ms']} | "
                 f"{r['stencil_mlups']} | {r.get('launch_us', '-')} | {r.get('h2d_gbs', '-')} |")
    if data.get("correctness"):
        L += ["", "### GPU vs CPU correctness", "", "| Check | Value |", "|---|---|"]
        for k, v in data["correctness"].items():
            L.append(f"| {k} | {v:.3e} |")
    L += ["", "### MAC solver, lid-driven cavity Re=1000: ms per time step", ""]
    names = data["backends"]
    L.append("| Grid | " + " | ".join(names) + " | Planner choice |")
    L.append("|---" * (len(names) + 2) + "|")
    for n, row in data["speed"].items():
        cells = []
        for b in names:
            r = row.get(b, {})
            txt = (f"{r['ms_per_step']} ({r['precision'].replace('float', 'fp')})" if "ms_per_step" in r
                   else r.get("error", "-"))
            if "ms_per_step_vectorised" in r:
                txt += f"; vectorised {r['ms_per_step_vectorised']}"
            cells.append(txt)
        L.append(f"| {n}² | " + " | ".join(cells) + f" | {row.get('planner_choice')} |")
    L.append("")
    return "\n".join(L)
