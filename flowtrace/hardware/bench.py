"""Micro-benchmarks that turn "what hardware is there" into "how fast is it".

The planner's cost model needs a handful of measured rates per backend:
sustained memory bandwidth (stencils are bandwidth bound), GEMM throughput
(dense fast-diagonalisation Poisson solves), DCT throughput (spectral Poisson
solves), kernel-launch latency and host<->device copy bandwidth. Each probe is
best-of-N and sized to finish in well under a second.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterable, Optional

import numpy as np

from .backend import available_backends, get_backend

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".flowtrace")
CACHE_FILE = os.path.join(CACHE_DIR, "bench.json")


def _best(fn, be, repeat=5) -> float:
    fn()  # warm-up (JIT, plan caches, allocator)
    best = float("inf")
    for _ in range(repeat):
        t0 = be.timer()
        fn()
        best = min(best, be.timer() - t0)
    return best


def _triad(be, n: int) -> float:
    xp = be.xp
    b = xp.ones(n, dtype=xp.float64)
    c = xp.ones(n, dtype=xp.float64)
    a = xp.empty_like(b)

    def run():
        xp.multiply(c, 3.0, out=a)
        xp.add(a, b, out=a)

    t = _best(run, be)
    # multiply: read c + write a; add: read a, b + write a  -> 5 streams
    return 5 * n * 8 / t / 1e9


def _gemm(be, n: int, dtype) -> float:
    xp = be.xp
    a = xp.ones((n, n), dtype=dtype)
    b = xp.ones((n, n), dtype=dtype)
    t = _best(lambda: a @ b, be, repeat=3)
    return 2.0 * n ** 3 / t / 1e9


def _dct(be, n: int) -> float:
    x = be.xp.ones((n, n), dtype=be.xp.float64)
    t = _best(lambda: be.fft.dctn(x, type=2, norm="ortho", **be.fft_kwargs), be)
    return t * 1e3


def _stencil(be, n: int) -> float:
    xp = be.xp
    p = xp.ones((n + 2, n + 2), dtype=xp.float64)
    out = xp.empty((n, n), dtype=xp.float64)
    if be.jit:
        from ..structured.kernels import laplacian5_numba

        def run():
            laplacian5_numba(p, out)
    else:
        def run():
            out[...] = p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4.0 * p[1:-1, 1:-1]

    t = _best(run, be)
    return n * n / t / 1e6


def _launch_latency(be) -> float:
    xp = be.xp
    a = xp.ones(1, dtype=xp.float32)

    def run():
        for _ in range(50):
            a.__iadd__(1.0)

    return _best(run, be) / 50 * 1e6


def _copy_bandwidth(be, nbytes: int) -> tuple:
    xp = be.xp
    host = np.ones(nbytes // 8, dtype=np.float64)
    dev = xp.asarray(host)
    h2d = _best(lambda: xp.asarray(host), be, repeat=3)
    d2h = _best(lambda: dev.get(), be, repeat=3)
    return nbytes / h2d / 1e9, nbytes / d2h / 1e9


def benchmark_backend(name: str, quick: bool = True) -> dict:
    be = get_backend(name)
    gpu = be.is_gpu
    n_stream = (1 << 25) if gpu else (1 << 23)
    n_gemm = 2048 if gpu else (512 if quick else 1024)
    res = {
        "backend": name,
        "device": be.device,
        "bandwidth_gbs": round(_triad(be, n_stream), 2),
        "gemm_gflops_fp64": round(_gemm(be, n_gemm, be.xp.float64), 2),
        "gemm_gflops_fp32": round(_gemm(be, n_gemm, be.xp.float32), 2),
        "dct512_ms": round(_dct(be, 512), 3),
        "stencil_mlups": round(_stencil(be, 1024), 1),
    }
    if gpu:
        res["launch_us"] = round(_launch_latency(be), 2)
        h2d, d2h = _copy_bandwidth(be, 64 << 20)
        res["h2d_gbs"] = round(h2d, 2)
        res["d2h_gbs"] = round(d2h, 2)
    else:
        res["launch_us"] = 0.0
    be.free_pools()
    return res


def _fingerprint() -> str:
    from .probe import probe_system

    prof = probe_system()
    gpus = ",".join(g.name for g in prof.gpus)
    libs = ",".join(f"{k}={v}" for k, v in sorted(prof.libraries.items()) if v)
    return f"{prof.cpu_model}|{prof.logical_cores}|{gpus}|{libs}"


def run_benchmarks(backends: Optional[Iterable[str]] = None, quick: bool = True,
                   use_cache: bool = True) -> dict:
    """Benchmark the given (default: all available) backends, with caching."""
    names = list(backends) if backends else available_backends()
    fp = _fingerprint()
    cache = load_cached() if use_cache else None
    if cache and cache.get("fingerprint") == fp and all(n in cache["results"] for n in names):
        return cache
    results = {}
    for n in names:
        try:
            results[n] = benchmark_backend(n, quick=quick)
        except Exception as exc:  # keep going: one broken backend must not hide the others
            results[n] = {"backend": n, "error": str(exc)}
    out = {"fingerprint": fp, "timestamp": time.time(), "results": results}
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    except OSError:
        pass
    return out


def load_cached() -> Optional[dict]:
    try:
        with open(CACHE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
