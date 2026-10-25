"""A NumPy-backed stand-in for CuPy, used to exercise FlowTrace's GPU code paths
on machines without an NVIDIA GPU.

It mimics the parts of the CuPy API FlowTrace touches (device queries, memory
pools, streams, ``cupyx.scipy.fft`` with CuPy's restriction to DCT/DST types 2
and 3, ``cupyx.scipy.sparse`` with ``.get()``), so every ``backend.is_gpu``
branch runs. It says nothing about CUDA performance, only about correctness of
the host-side logic.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import scipy.fft as _sfft
import scipy.sparse as _ssp


class FakeArray(np.ndarray):
    def get(self):
        return np.array(self)


def _wrap(f):
    def g(*a, **k):
        r = f(*a, **k)
        if isinstance(r, np.ndarray) and not isinstance(r, FakeArray):
            return r.view(FakeArray)
        return r

    g.__name__ = getattr(f, "__name__", "f")
    return g


class _Device:
    def __init__(self, i=0):
        self.i = i

    def use(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def mem_info(self):
        return (6 << 30, 8 << 30)


class _Pool:
    def free_all_blocks(self):
        pass


def install():
    if "cupy" in sys.modules and getattr(sys.modules["cupy"], "__fake__", False):
        return sys.modules["cupy"]
    cupy = types.ModuleType("cupy")
    cupy.__fake__ = True
    cupy.__version__ = "fake-13.0"
    for name in dir(np):
        if name.startswith("_"):
            continue
        attr = getattr(np, name)
        if isinstance(attr, type) or isinstance(attr, types.ModuleType) or not callable(attr):
            setattr(cupy, name, attr)
        else:
            setattr(cupy, name, _wrap(attr))
    cupy.ndarray = FakeArray
    cuda = types.ModuleType("cupy.cuda")
    cuda.Device = _Device
    stream = types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None))
    cuda.Stream = stream
    rt = types.ModuleType("cupy.cuda.runtime")
    rt.getDeviceCount = lambda: 1
    rt.getDeviceProperties = lambda i: {"name": b"NVIDIA GeForce RTX 4070 Laptop GPU (fake)", "major": 8, "minor": 9,
                                        "multiProcessorCount": 36, "memoryClockRate": 8001000, "memoryBusWidth": 128}
    rt.driverGetVersion = lambda: 12040
    cuda.runtime = rt
    cupy.cuda = cuda
    cupy.get_default_memory_pool = lambda: _Pool()
    cupy.get_default_pinned_memory_pool = lambda: _Pool()

    cupyx = types.ModuleType("cupyx")
    cscipy = types.ModuleType("cupyx.scipy")
    cfft = types.ModuleType("cupyx.scipy.fft")

    def _restricted(fn):
        def g(x, type=2, axis=-1, norm=None, **kw):
            if type not in (2, 3):
                raise NotImplementedError("CuPy only implements DCT/DST types 2 and 3")
            return np.asarray(fn(np.asarray(x), type=type, axis=axis, norm=norm)).view(FakeArray)
        return g

    for nm in ("dct", "idct", "dst", "idst"):
        setattr(cfft, nm, _restricted(getattr(_sfft, nm)))
    cfft.dctn = lambda x, type=2, norm=None, **kw: np.asarray(_sfft.dctn(np.asarray(x), type=type, norm=norm)).view(FakeArray)
    cfft.fft = lambda x, axis=-1, **kw: np.asarray(_sfft.fft(np.asarray(x), axis=axis)).view(FakeArray)
    cfft.ifft = lambda x, axis=-1, **kw: np.asarray(_sfft.ifft(np.asarray(x), axis=axis)).view(FakeArray)

    csparse = types.ModuleType("cupyx.scipy.sparse")

    class csr_matrix(_ssp.csr_matrix):
        def get(self):
            return _ssp.csr_matrix(self)

        def __matmul__(self, other):
            r = _ssp.csr_matrix.__matmul__(self, other)
            return r.view(FakeArray) if isinstance(r, np.ndarray) else r

    csparse.csr_matrix = csr_matrix
    cscipy.fft = cfft
    cscipy.sparse = csparse
    cupyx.scipy = cscipy
    for name, mod in (("cupy", cupy), ("cupy.cuda", cuda), ("cupy.cuda.runtime", rt), ("cupyx", cupyx),
                      ("cupyx.scipy", cscipy), ("cupyx.scipy.fft", cfft), ("cupyx.scipy.sparse", csparse)):
        sys.modules[name] = mod
    return cupy


def uninstall():
    for name in ("cupy", "cupy.cuda", "cupy.cuda.runtime", "cupyx", "cupyx.scipy", "cupyx.scipy.fft",
                 "cupyx.scipy.sparse"):
        sys.modules.pop(name, None)
