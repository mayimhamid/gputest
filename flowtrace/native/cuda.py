"""CUDA finite-volume kernels (``fvm_kernels.cu``) for the unstructured solver on CuPy.

Compiled at first use by NVRTC through ``cupy.RawModule`` (FP64). ``available()``
is False without a real CUDA device, when compilation fails, or with
``FLOWTRACE_NO_NATIVE=1`` / ``FLOWTRACE_NO_CUDA_KERNELS=1``; the solver then keeps
the vectorised CuPy path.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from . import CUDA_SOURCE

_module = None
_failure: Optional[str] = None
THREADS = 256


def available() -> bool:
    global _module, _failure
    import sys

    cp_mod = sys.modules.get("cupy")
    if cp_mod is not None and getattr(cp_mod, "__fake__", False):
        return False  # the tests' NumPy-backed CuPy stand-in is active: never use (or cache) real kernels
    if _module is not None:
        return True
    if _failure is not None:
        return False
    if os.environ.get("FLOWTRACE_NO_NATIVE") or os.environ.get("FLOWTRACE_NO_CUDA_KERNELS"):
        _failure = "disabled by environment"
        return False
    try:
        import cupy  # type: ignore

        if getattr(cupy, "__fake__", False):
            return False
        with open(CUDA_SOURCE, encoding="utf-8") as fh:
            src = fh.read()
        mod = cupy.RawModule(code=src, options=("--std=c++14",))
        mod.get_function("ft_ls_gradient")  # forces compilation
        _module = mod
        return True
    except Exception as exc:
        _failure = f"{type(exc).__name__}: {exc}"
        return False


def failure_reason() -> Optional[str]:
    return _failure


def _blocks(n: int) -> int:
    return max(1, math.ceil(n / THREADS))


class FVKernels:
    """GPU counterparts of the FVMesh operators, bound to one mesh's device arrays."""

    def __init__(self, fv, dxm):
        import cupy as cp  # type: ignore

        self.cp, self.fv = cp, fv
        self.f = {n: _module.get_function(n) for n in ("ft_face_interp", "ft_ls_gradient", "ft_gg_gradient",
                                                       "ft_assemble", "ft_dot2", "ft_bicg_p", "ft_bicg_s",
                                                       "ft_bicg_x")}
        i32 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.int32))  # noqa: E731
        f64 = lambda a: cp.ascontiguousarray(cp.asarray(a, dtype=cp.float64))  # noqa: E731
        Cd, Cb = fv.Cdiff.tocsr(), fv.Cb.tocsr()
        Cd.sort_indices()
        Cb.sort_indices()
        self.iptr, self.iidx, self.isgn = i32(Cd.indptr), i32(Cd.indices), f64(Cd.data)
        self.bptr, self.bidx = i32(Cb.indptr), i32(Cb.indices)
        self.P, self.N = cp.ascontiguousarray(fv.P.astype(cp.int64)), cp.ascontiguousarray(fv.N.astype(cp.int64))
        self.w, self.wl, self.d, self.wlb, self.db = f64(fv.w), f64(fv.wl), f64(fv.d), f64(fv.wlb), f64(fv.db)
        self.Ginv, self.S, self.Sb, self.vol = f64(fv.Ginv), f64(fv.S), f64(fv.Sb), f64(fv.vol)
        self.dxm = f64(dxm)
        self.slot_diag, self.slot_PN, self.slot_NP = i32(fv.slot_diag), i32(fv.slot_PN), i32(fv.slot_NP)
        self.acc = cp.zeros(2, dtype=cp.float64)
        self.red_blocks = min(1024, _blocks(fv.nc))

    # ---------------------------------------------------------------- face / cell operators
    def face_interp(self, a, k):
        cp = self.cp
        a = cp.ascontiguousarray(a, dtype=cp.float64)
        nf = self.fv.n_if
        out = cp.empty((nf,) if k == 1 else (nf, k), dtype=cp.float64)
        self.f["ft_face_interp"]((_blocks(nf * k),), (THREADS,),
                                 (np.int64(nf), np.int64(k), self.w, self.P, self.N, a, out))
        return out

    def ls_gradient(self, phi, phib):
        cp, fv = self.cp, self.fv
        phi, phib = cp.ascontiguousarray(phi, dtype=cp.float64), cp.ascontiguousarray(phib, dtype=cp.float64)
        out = cp.empty((fv.nc, fv.dim), dtype=cp.float64)
        self.f["ft_ls_gradient"]((_blocks(fv.nc),), (THREADS,),
                                 (np.int64(fv.nc), np.int64(fv.dim), self.iptr, self.iidx, self.bptr, self.bidx,
                                  self.P, self.N, self.wl, self.d, self.wlb, self.db, self.Ginv, phi, phib, out))
        return out

    def gg_gradient(self, phi, phib, grad=None):
        cp, fv = self.cp, self.fv
        phi, phib = cp.ascontiguousarray(phi, dtype=cp.float64), cp.ascontiguousarray(phib, dtype=cp.float64)
        g = cp.ascontiguousarray(grad, dtype=cp.float64) if grad is not None else phi
        out = cp.empty((fv.nc, fv.dim), dtype=cp.float64)
        self.f["ft_gg_gradient"]((_blocks(fv.nc),), (THREADS,),
                                 (np.int64(fv.nc), np.int64(fv.dim), self.iptr, self.iidx, self.isgn, self.bptr,
                                  self.bidx, self.P, self.N, self.w, self.S, self.Sb, self.vol, self.dxm, phi, phib, g,
                                  np.int32(grad is not None), out))
        return out

    def matrix(self, diag, aPN, aNP):
        cp, fv = self.cp, self.fv
        data = cp.zeros(fv.nnz, dtype=cp.float64)
        self.f["ft_assemble"]((_blocks(fv.nc),), (THREADS,),
                              (np.int64(fv.nc), self.iptr, self.iidx, self.isgn, self.slot_diag, self.slot_PN,
                               self.slot_NP, cp.ascontiguousarray(diag, dtype=cp.float64),
                               cp.ascontiguousarray(aPN, dtype=cp.float64), cp.ascontiguousarray(aNP, dtype=cp.float64),
                               data))
        return fv._sparse.csr_matrix((data, fv.indices_d, fv.indptr_d), shape=(fv.nc, fv.nc))

    # ---------------------------------------------------------------- Krylov
    def _red(self, name, args, n):
        self.acc.fill(0.0)
        self.f[name]((self.red_blocks,), (THREADS,), args)
        return self.acc.get()

    def bicgstab_jacobi(self, A, b, x0, inv, tol, maxiter):
        """Jacobi-BiCGSTAB with fused vector kernels: 4 host synchronisations per iteration
        (the Python version needs 6) and ~7 launches (16). Same algorithm and stopping
        rules as krylov.bicgstab(relative_to_initial=True)."""
        from ..linalg.krylov import KrylovInfo

        cp = self.cp
        n = b.shape[0]
        N64 = np.int64(n)
        b = cp.ascontiguousarray(b, dtype=cp.float64)
        inv = cp.ascontiguousarray(inv, dtype=cp.float64)
        x = cp.array(x0, dtype=cp.float64, copy=True, order="C")
        r = b - A @ x
        rr, bb = self._red("ft_dot2", (N64, r, r, b, b, np.int32(1), self.acc), n)
        rnorm, b_abs = math.sqrt(rr), math.sqrt(bb)
        bnorm = rnorm or 1.0
        if rnorm / bnorm < tol or rnorm <= 1e-14 * b_abs:
            return x, KrylovInfo(0, rnorm / bnorm, True)
        r_hat = r.copy()
        p, v = cp.zeros_like(b), cp.zeros_like(b)
        h, hs, s, xn = cp.empty_like(b), cp.empty_like(b), cp.empty_like(b), cp.empty_like(b)
        rho = alpha = omega = 1.0
        rho_new = rr  # r_hat . r with r_hat = r
        k = 0
        for k in range(1, maxiter + 1):
            if rho_new == 0.0:
                break
            beta = (rho_new / rho) * (alpha / omega)
            rho = rho_new
            self.f["ft_bicg_p"]((_blocks(n),), (THREADS,), (N64, np.float64(beta), np.float64(omega), r, v, inv, p, h))
            v = A @ h
            denom = self._red("ft_dot2", (N64, r_hat, v, r_hat, v, np.int32(0), self.acc), n)[0]
            if denom == 0.0:
                break
            alpha = rho / denom
            ss = self._red("ft_bicg_s", (N64, np.float64(alpha), r, v, inv, s, hs, self.acc), n)[0]
            snorm = math.sqrt(ss)
            if snorm / bnorm < tol:
                x += alpha * h
                return x, KrylovInfo(k, snorm / bnorm, True)
            t = A @ hs
            tt, ts = self._red("ft_dot2", (N64, t, t, t, s, np.int32(1), self.acc), n)
            omega = ts / tt if tt > 0 else 0.0
            rr, rho_new = self._red("ft_bicg_x", (N64, np.float64(alpha), np.float64(omega), x, h, hs, s, t, r_hat, xn, r,
                                                   self.acc), n)
            rnorm = math.sqrt(rr) if rr >= 0 else float("nan")
            if not math.isfinite(rnorm):
                return x, KrylovInfo(k, float("nan"), False)
            x, xn = xn, x
            if rnorm / bnorm < tol or rnorm <= 1e-14 * b_abs:
                return x, KrylovInfo(k, rnorm / bnorm, True)
            if omega == 0.0:
                break
        return x, KrylovInfo(k, rnorm / bnorm, False)
