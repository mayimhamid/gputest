"""Array backends: one numerical code path, several devices.

Solvers are written against ``backend.xp`` (NumPy-compatible API) and
``backend.fft`` (SciPy-compatible FFT/DCT API). The CPU backends use NumPy and
SciPy (multi-threaded FFTs); ``numba`` additionally enables JIT stencil kernels;
``cupy`` keeps every field resident in GPU memory so that only scalars and
throttled visualisation frames ever cross the PCIe bus.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

BACKEND_NAMES = ("numpy", "numba", "cupy")


class BackendUnavailable(RuntimeError):
    pass


class Backend:
    def __init__(self, name: str = "numpy", device_index: int = 0):
        if name not in BACKEND_NAMES:
            raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}")
        self.name = name
        self.device_index = device_index
        self.jit = False
        if name == "cupy":
            try:
                import cupy  # type: ignore
                import cupyx.scipy.fft as cufft  # type: ignore

                cupy.cuda.Device(device_index).use()
                cupy.arange(4).sum().item()
            except Exception as exc:
                raise BackendUnavailable(f"CuPy GPU backend unavailable: {exc}") from exc
            self.xp = cupy
            self.fft = cufft
            self.device = "gpu"
            self.fft_kwargs = {}
        else:
            import scipy.fft as sfft

            self.xp = np
            self.fft = sfft
            self.device = "cpu"
            self.fft_kwargs = {"workers": os.cpu_count() or 1}
            if name == "numba":
                try:
                    import numba  # type: ignore  # noqa: F401
                except Exception as exc:
                    raise BackendUnavailable(f"Numba backend unavailable: {exc}") from exc
                self.jit = True
        self._fft_support = {}

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Backend({self.name!r}, device={self.device!r})"

    @property
    def is_gpu(self) -> bool:
        return self.device == "gpu"

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_numpy(self, a) -> np.ndarray:
        if self.is_gpu and not isinstance(a, np.ndarray):
            return a.get()
        return np.asarray(a)

    def scalar(self, a) -> float:
        """Bring a 0-d device value to the host (synchronises on GPU)."""
        return float(a.item() if hasattr(a, "item") else a)

    def synchronize(self) -> None:
        if self.is_gpu:
            self.xp.cuda.Stream.null.synchronize()

    def timer(self) -> float:
        self.synchronize()
        return time.perf_counter()

    def mem_info(self) -> tuple:
        """(free, total) bytes of the device the fields live on."""
        if self.is_gpu:
            return tuple(int(x) for x in self.xp.cuda.Device(self.device_index).mem_info)
        from .probe import _ram

        total, avail = _ram()
        return avail, total

    def free_pools(self) -> None:
        if self.is_gpu:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()

    def supports_transform(self, kind: str, type_: int) -> bool:
        """Whether ``fft.{kind}(type=type_)`` (dct/dst) works on this backend."""
        key = (kind, type_)
        if key not in self._fft_support:
            try:
                x = self.xp.ones((4, 3))
                getattr(self.fft, kind)(x, type=type_, axis=0, norm="ortho")
                self._fft_support[key] = True
            except Exception:
                self._fft_support[key] = False
        return self._fft_support[key]


_cache = {}


def get_backend(name: Optional[str] = None, device_index: int = 0) -> Backend:
    """Return a cached backend. ``None``/``'auto'`` picks GPU, then Numba, then NumPy."""
    if name in (None, "auto"):
        for cand in ("cupy", "numba", "numpy"):
            try:
                return get_backend(cand, device_index)
            except BackendUnavailable:
                continue
    key = (name, device_index)
    if key not in _cache:
        _cache[key] = Backend(name, device_index)
    return _cache[key]


def available_backends() -> list:
    names = []
    for cand in BACKEND_NAMES:
        try:
            get_backend(cand)
            names.append(cand)
        except BackendUnavailable:
            pass
    return names
