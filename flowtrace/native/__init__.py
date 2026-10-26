"""Native (C++/OpenMP and CUDA) kernels for the unstructured finite-volume solver.

The NumPy/SciPy and CuPy code paths stay the reference implementation; these
kernels replace their hottest, most call-heavy parts:

* ``fvm_kernels.cpp`` (C API in ``fvm_kernels.h``): parallel CSR products, fused
  least-squares and Green-Gauss gradients, face interpolation, matrix assembly
  and a complete Jacobi-BiCGSTAB solve. It is compiled at first use with g++ (or
  ``$FLOWTRACE_CXX``) and cached in ``~/.flowtrace/native/``. On Windows it is
  linked statically, so it needs no compiler runtime DLLs.
* ``fvm_kernels.cu``: the same face/cell kernels for the GPU, compiled at first
  use by NVRTC through CuPy (no host compiler needed, unlike nvcc on Windows).

Without a compiler, or with ``FLOWTRACE_NO_NATIVE=1``, everything falls back to
the NumPy/CuPy path. ``status()`` reports what is active and why.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "fvm_kernels.cpp")
HEADER = os.path.join(HERE, "fvm_kernels.h")
CUDA_SOURCE = os.path.join(HERE, "fvm_kernels.cu")
CACHE = os.path.join(os.path.expanduser("~"), ".flowtrace", "native")
ABI = 2

_lib = None
_failure: Optional[str] = None
_info: dict = {}


def _compiler() -> Optional[str]:
    for cand in (os.environ.get("FLOWTRACE_CXX"), "g++", "c++", "clang++"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    return None


def _flags() -> list:
    flags = ["-O3", "-std=c++17", "-fopenmp", "-shared", "-ffp-contract=off"]
    if os.environ.get("FLOWTRACE_NATIVE_PORTABLE"):
        flags.append("-march=x86-64-v2" if platform.machine().lower() in ("amd64", "x86_64") else "-O3")
    else:
        flags.append("-march=native")
    if sys.platform == "win32":
        flags += ["-static", "-static-libgcc", "-static-libstdc++"]  # no MinGW runtime DLLs at load time
    else:
        flags.append("-fPIC")
    return flags


def build(force: bool = False, verbose: bool = False) -> str:
    """Compile the C++ kernels (if needed) and return the path of the shared library."""
    cxx = _compiler()
    if cxx is None:
        raise RuntimeError("no C++ compiler found (install g++ or set FLOWTRACE_CXX)")
    try:
        version = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cannot run {cxx}: {exc}") from exc
    flags = _flags()
    h = hashlib.sha256()
    for path in (SOURCE, HEADER):
        with open(path, "rb") as fh:
            h.update(fh.read())
    h.update(" ".join([version, platform.processor(), platform.machine(), *flags]).encode())
    ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    out_dir = os.path.join(CACHE, h.hexdigest()[:16])
    lib = os.path.join(out_dir, "flowtrace_fvm" + ext)
    if os.path.exists(lib) and not force:
        return lib
    os.makedirs(out_dir, exist_ok=True)
    tmp = lib + f".{os.getpid()}.tmp"
    cmd = [cxx, *flags, "-I", HERE, SOURCE, "-o", tmp]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if cp.returncode != 0:
        raise RuntimeError(f"compiling the native kernels failed:\n{' '.join(cmd)}\n{cp.stderr[-3000:]}")
    os.replace(tmp, lib)  # atomic: concurrent builders (batch workers) never load a half-written file
    with open(os.path.join(out_dir, "build.txt"), "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n" + version + "\n")
    if verbose:
        print(" ".join(cmd))
    return lib


def _bind(lib):
    P = ctypes.c_void_p
    I64, I = ctypes.c_int64, ctypes.c_int
    sig = {
        "ft_abi_version": ([], I), "ft_max_threads": ([], I),
        "ft_csr_spmm": ([I64, I64, P, P, P, P, P], None),
        "ft_face_interp": ([I64, I64, P, P, P, P, P], None),
        "ft_ls_gradient": ([I64, I64] + [P] * 14, None),
        "ft_gg_gradient": ([I64, I64] + [P] * 16, None),
        "ft_assemble": ([I64, I64, I64] + [P] * 7, None),
        "ft_bicgstab_jacobi": ([I64, P, P, P, P, P, P, ctypes.c_double, I64, I, P, P, P], I),
        "ft_hllc_flux": ([I64, I64, ctypes.c_double, P, P, P, P, P], None),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    return lib


def load():
    """The native library (ctypes.CDLL), or None if unavailable (see ``status()``)."""
    global _lib, _failure
    if _lib is not None or _failure is not None:
        return _lib
    if os.environ.get("FLOWTRACE_NO_NATIVE"):
        _failure = "disabled by FLOWTRACE_NO_NATIVE"
        return None
    try:
        path = build()
        lib = _bind(ctypes.CDLL(path))
        if lib.ft_abi_version() != ABI:
            raise RuntimeError("stale native library (ABI mismatch)")
        _lib = lib
        _info.update(path=path, threads=lib.ft_max_threads())
    except Exception as exc:
        _failure = str(exc)
    return _lib


def available() -> bool:
    return load() is not None


class disabled:
    """Context manager: solvers built inside it use the NumPy reference path (tests, comparisons)."""

    def __enter__(self):
        global _lib, _failure
        load()
        self._saved = (_lib, _failure)
        _lib, _failure = None, "disabled (flowtrace.native.disabled)"
        return self

    def __exit__(self, *exc):
        global _lib, _failure
        _lib, _failure = self._saved
        return False


def status() -> dict:
    load()
    return {"cpu_kernels": _lib is not None, "reason": _failure, **_info}


# ----------------------------------------------------------------------------- helpers
def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


class NativeCSR:
    """A SciPy CSR matrix whose products ``A @ x`` (1-D or 2-D x) run in the parallel kernel.

    Every other attribute is the wrapped SciPy matrix's, so it can be used wherever
    the SciPy matrix was (pyamg, ``.diagonal()``, ``.T``, ...).
    """

    def __init__(self, A):
        import scipy.sparse as sp

        self.csr = sp.csr_matrix(A, dtype=np.float64)
        self.csr.sort_indices()
        self.indptr, self.indices, self.data = i32(self.csr.indptr), i32(self.csr.indices), f64(self.csr.data)
        self.shape = self.csr.shape
        self.dtype = self.csr.dtype

    def __matmul__(self, x):
        lib = _lib
        x = np.asarray(x)
        if lib is None or x.dtype != np.float64 or x.ndim not in (1, 2) or x.shape[0] != self.shape[1]:
            return self.csr @ x
        x = np.ascontiguousarray(x)
        k = 1 if x.ndim == 1 else x.shape[1]
        y = np.empty((self.shape[0],) if x.ndim == 1 else (self.shape[0], k))
        lib.ft_csr_spmm(self.shape[0], k, _p(self.indptr), _p(self.indices), _p(self.data), _p(x), _p(y))
        return y

    def __getattr__(self, name):
        return getattr(self.csr, name)


def csr_from_parts(data, indices, indptr, shape) -> NativeCSR:
    """NativeCSR around existing CSR arrays, without SciPy's validation pass."""
    import scipy.sparse as sp

    m = NativeCSR.__new__(NativeCSR)
    m.data, m.indices, m.indptr = f64(data), i32(indices), i32(indptr)
    m.csr = sp.csr_matrix((m.data, m.indices, m.indptr), shape=shape)
    m.shape, m.dtype = shape, np.dtype(np.float64)
    return m


def bicgstab_jacobi(A, b, x0, inv_diag, tol: float, maxiter: int, relative_to_initial: bool = True, lib=None):
    """Native Jacobi-BiCGSTAB on a CSR matrix; returns (x, KrylovInfo) like krylov.bicgstab."""
    from ..linalg.krylov import KrylovInfo

    lib = lib or load()
    if lib is None:
        raise RuntimeError("native kernels unavailable: " + str(_failure))
    indptr, indices, data = (A.indptr, A.indices, A.data) if isinstance(A, NativeCSR) else \
        (i32(A.indptr), i32(A.indices), f64(A.data))
    n = A.shape[0]
    b, inv = f64(b), f64(inv_diag)
    x = np.array(x0, dtype=np.float64, copy=True, order="C")
    work = np.empty(7 * n)
    it, res = ctypes.c_int64(0), ctypes.c_double(0.0)
    ok = lib.ft_bicgstab_jacobi(n, _p(indptr), _p(indices), _p(data), _p(b), _p(inv), _p(x), float(tol),
                                int(maxiter), int(bool(relative_to_initial)), _p(work), ctypes.byref(it),
                                ctypes.byref(res))
    return x, KrylovInfo(int(it.value), float(res.value), bool(ok))


def hllc_flux(WL, WR, n, A, gamma: float, lib=None) -> np.ndarray:
    """HLLC flux (nf, dim+2) times the face areas, on the CPU (see ``ft_hllc_flux``)."""
    lib = lib or load()
    WL, WR, n, A = f64(WL), f64(WR), f64(n), f64(A)
    out = np.empty_like(WL)
    lib.ft_hllc_flux(WL.shape[0], n.shape[1], float(gamma), _p(WL), _p(WR), _p(n), _p(A), _p(out))
    return out
