"""Run many independent unstructured cases at once, on every device of the machine.

One SIMPLEC solve of a small or medium mesh cannot keep a modern machine busy:
on the 23k-cell square-cylinder mesh the CPU iteration is ~40 ms of many small
NumPy/SciPy operations (one core), and the GPU iteration sits near its launch
floor. Studies are made of many such solves (mesh-independence sets, Reynolds
or property sweeps, design variants), and those are independent. ``run_batch``
therefore runs them concurrently:

* one GPU worker takes the cases the cost model says the GPU is faster for
  (CPU prediction above ``GPU_ADVANTAGE`` x the measured GPU floor), largest first;
* every other physical core runs one case on the CPU (single-threaded BLAS, so
  the workers do not fight over cores), largest first;
* an idle GPU worker takes a CPU-bound case only when every CPU worker is busy.

Each case runs in its own process with an explicit backend, so the result of a
case does not depend on what else ran beside it (up to the ~1e-8 cross-process
round-off noise of the AMG setup). Every result carries the device it ran on.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

GPU_ADVANTAGE = 1.5
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


@dataclass
class BatchJob:
    name: str
    mesh: str  # path of a Fluent mesh/case (.msh, .cas, .cas.h5, .msh.h5)
    case: dict = field(default_factory=dict)  # UnstructuredCase keyword arguments (boundaries as dicts)
    out_dir: str = ""  # if set: <out_dir>/<name>.ip (Fluent interpolation file) and <name>.npz
    solver: str = "pressure-based"  # or "density-based": CompressibleCase keyword arguments in ``case``


def _run_one(job: dict, backend: str) -> dict:
    """Worker: solve one case on the given backend and report it (runs in a child process)."""
    t0 = time.perf_counter()
    res = {"name": job["name"], "mesh": job["mesh"], "backend": backend, "error": None}
    try:
        import numpy as np

        from ..hardware.backend import get_backend
        from ..mesh.fluent_h5 import read_any_mesh
        from ..unstructured.case import UnstructuredCase, default_boundaries
        from ..unstructured.conjugate import make_solver

        mesh, _ = read_any_mesh(job["mesh"])
        if job.get("solver", "pressure-based") == "density-based":
            from ..compressible.run import build_case, solve

            out = solve(mesh, build_case(mesh, job.get("case"), backend), out_dir=job.get("out_dir") or "",
                        name=job["name"], backend=get_backend(backend))
            out.pop("solver")
            out.pop("describe")
            res.update(out)
            res["seconds"] = time.perf_counter() - t0
            return res
        kw = dict(job.get("case") or {})
        bnd = default_boundaries(mesh)
        for k, v in (kw.pop("boundaries", None) or {}).items():
            if k not in bnd:
                raise ValueError(f"zone {k!r} is not a boundary zone of {job['mesh']} ({', '.join(sorted(bnd))})")
            bnd[k] = v
        kw["backend"] = backend
        case = UnstructuredCase(boundaries=bnd, **kw)
        solver = make_solver(mesh, case, backend=get_backend(backend))
        t1 = time.perf_counter()
        while not solver.finished:
            solver.step()
        res.update({"cells": mesh.n_cells, "device": solver.be.device, "iterations": solver.iteration,
                    "converged": bool(solver.converged), "solve_seconds": time.perf_counter() - t1,
                    "residuals": solver.residuals()["scaled"], "mass_balance": solver.mass_balance(),
                    "wall_forces": solver.wall_forces(),
                    "heat_flow": solver.heat_flow() if case.energy else {},
                    "stability_rollbacks": len(getattr(getattr(solver, "flow", solver), "guard_events", []))})
        try:
            from ..unstructured.diagnostics import summary

            res["diagnostics"] = summary(solver.diagnostics())
        except Exception as exc:  # a diagnostics failure must not turn a solved case into an error
            res["diagnostics_error"] = f"{type(exc).__name__}: {exc}"
        if job.get("out_dir"):
            from ..fluent.export import write_interpolation_file

            os.makedirs(job["out_dir"], exist_ok=True)
            f = solver.fields()
            base = os.path.join(job["out_dir"], job["name"])
            keep = ("x-velocity", "y-velocity", "z-velocity", "pressure", "temperature", "turb-kinetic-energy",
                    "specific-diss-rate", "turb-diss-rate")
            res["ip"] = write_interpolation_file(base + ".ip", solver.cell_centres(),
                                                 {k: v for k, v in f.items() if k in keep})
            np.savez_compressed(base + ".npz", centres=solver.cell_centres(), **f)
            res["npz"] = base + ".npz"
    except Exception as exc:  # one failing case must not stop the batch
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-3000:]
    res["seconds"] = time.perf_counter() - t0
    return res


def _cells(path: str) -> int:
    from ..mesh.fluent_h5 import read_any_mesh

    return read_any_mesh(path)[0].n_cells


def plan_batch(jobs: List[BatchJob], cpu_workers: Optional[int] = None, use_gpu: Optional[bool] = None,
               precision: str = "fp64") -> dict:
    """Decide the worker counts and each job's preferred device (no solving)."""
    from ..hardware.backend import available_backends
    from ..hardware.planner import calibrate_unstructured

    have_gpu = "cupy" in available_backends() if use_gpu is None else bool(use_gpu)
    phys = max(1, (os.cpu_count() or 2) // 2)
    try:
        import psutil  # type: ignore

        phys = psutil.cpu_count(logical=False) or phys
    except Exception:
        pass
    n_cpu = cpu_workers if cpu_workers is not None else max(1, phys - (1 if have_gpu else 0))
    cal = calibrate_unstructured(gpu="cupy" if have_gpu else None, cpu="numpy", precision=precision)
    floor = cal.get("gpu_floor_ms") if have_gpu else None
    rows = []
    for j in jobs:
        n = _cells(j.mesh)
        pred_cpu = cal["cpu_us_per_cell"] * 1e-3 * n
        prefer = "gpu" if floor is not None and pred_cpu > GPU_ADVANTAGE * floor else "cpu"
        rows.append({"name": j.name, "cells": n, "pred_cpu_ms": pred_cpu, "prefer": prefer})
    return {"cpu_workers": n_cpu, "gpu_workers": 1 if have_gpu else 0, "gpu_floor_ms": floor,
            "cpu_us_per_cell": cal["cpu_us_per_cell"], "jobs": rows}


def run_batch(jobs: List[BatchJob], cpu_workers: Optional[int] = None, use_gpu: Optional[bool] = None,
              precision: str = "fp64", log: Callable[[str], None] = print) -> dict:
    """Solve every job; returns {"plan": .., "results": [..] (in job order), "wall_seconds": ..}."""
    t0 = time.perf_counter()
    plan = plan_batch(jobs, cpu_workers, use_gpu, precision)
    info = {r["name"]: r for r in plan["jobs"]}
    by_name = {j.name: j for j in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("job names must be unique")
    pending = sorted(by_name, key=lambda n: -info[n]["cells"])
    log(f"Batch: {len(jobs)} cases, {plan['cpu_workers']} CPU worker(s)"
        + (f" + 1 GPU worker (floor {plan['gpu_floor_ms']:.1f} ms/it)" if plan["gpu_workers"] else ""))
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:  # inherited by the spawned workers: one core each
        os.environ[k] = "1"
    results: Dict[str, dict] = {}
    try:
        cpu_pool = ProcessPoolExecutor(max_workers=plan["cpu_workers"])
        gpu_pool = ProcessPoolExecutor(max_workers=1) if plan["gpu_workers"] else None
        running = {}
        busy = {"cpu": 0, "gpu": 0}

        def take(device: str) -> Optional[str]:
            for n in pending:  # largest first
                if info[n]["prefer"] == device:
                    pending.remove(n)
                    return n
            if device == "gpu" and pending and busy["cpu"] >= plan["cpu_workers"]:
                return pending.pop(0)  # every CPU worker is busy: the GPU helps
            if device == "cpu" and pending and gpu_pool is None:
                return pending.pop(0)
            return None

        def fill():
            for device, pool, cap in (("gpu", gpu_pool, 1), ("cpu", cpu_pool, plan["cpu_workers"])):
                while pool is not None and busy[device] < cap:
                    n = take(device)
                    if n is None:
                        break
                    backend = "cupy" if device == "gpu" else "numpy"
                    fut = pool.submit(_run_one, asdict(by_name[n]), backend)
                    running[fut] = (n, device)
                    busy[device] += 1
                    log(f"  start {n} ({info[n]['cells']:,} cells) on {device.upper()}")

        fill()
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                n, device = running.pop(fut)
                busy[device] -= 1
                try:
                    r = fut.result()
                except Exception as exc:  # the worker process itself died
                    r = {"name": n, "error": f"worker failed: {exc}"}
                results[n] = r
                if r.get("error"):
                    log(f"  FAILED {n} on {device.upper()}: {r['error']}")
                else:
                    log(f"  done  {n} on {device.upper()}: {r['iterations']} its, "
                        f"{'converged' if r['converged'] else 'NOT converged'}, {r['seconds']:.1f} s")
            fill()
            if not running and pending:  # nothing could take the rest (e.g. GPU-only jobs, no GPU)
                fill_device = "gpu" if gpu_pool is not None else "cpu"
                n = pending.pop(0)
                pool = gpu_pool if fill_device == "gpu" else cpu_pool
                fut = pool.submit(_run_one, asdict(by_name[n]), "cupy" if fill_device == "gpu" else "numpy")
                running[fut] = (n, fill_device)
                busy[fill_device] += 1
        cpu_pool.shutdown()
        if gpu_pool is not None:
            gpu_pool.shutdown()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    wall = time.perf_counter() - t0
    log(f"Batch finished in {wall:.1f} s")
    return {"plan": plan, "results": [results[j.name] for j in jobs], "wall_seconds": wall}
