r"""Direct fast solvers for separable Poisson/Helmholtz problems.

Solves ``(sigma - Laplacian) u = f`` on tensor-product grids by diagonalising
the 1-D operator along every axis:

* uniform spacing with matching boundary conditions -> fast trigonometric
  transforms (DCT/DST, O(N log N)), e.g. all-Neumann pressure -> DCT-II;
* anything else (non-uniform spacing, mixed BCs, small N, or a backend without
  that transform) -> dense generalised eigen-decomposition
  ``K v = lambda M v``; applying it is a GEMM, which maps onto BLAS on CPUs and
  onto Tensor Cores (TF32/FP64-MMA) on GPUs.

Both paths are exact: there is no iteration, so the pressure projection leaves
a divergence at round-off level. The shift ``sigma`` may change on every call
(adaptive time steps) at O(N) cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

NEUMANN, DIRICHLET, PERIODIC = "neumann", "dirichlet", "periodic"


@dataclass
class Axis1D:
    """One axis of a tensor-product grid, seen from the unknowns' side.

    ``stagger='cell'``: unknowns at cell centres; a Dirichlet boundary value
    sits on the bounding face (half a cell away).
    ``stagger='node'``: unknowns at interior nodes; Dirichlet values sit on the
    boundary nodes (a full spacing away). Only uniform spacing.
    ``faces``: optional face coordinates (n+1) for non-uniform cell grids.
    """

    n: int
    h: float = 1.0
    lo: str = NEUMANN
    hi: str = NEUMANN
    stagger: str = "cell"
    faces: Optional[Sequence[float]] = None

    def __post_init__(self):
        if (self.lo == PERIODIC) != (self.hi == PERIODIC):
            raise ValueError("periodic must be set on both ends of an axis")
        if self.stagger not in ("cell", "node"):
            raise ValueError("stagger must be 'cell' or 'node'")
        if self.faces is not None:
            self.faces = np.asarray(self.faces, dtype=float)
            if self.faces.shape != (self.n + 1,):
                raise ValueError("faces must have n+1 entries")

    @property
    def uniform(self) -> bool:
        if self.faces is None:
            return True
        w = np.diff(self.faces)
        return bool(np.allclose(w, w[0], rtol=1e-12, atol=0))

    def matrices(self):
        """Return (K, M): symmetric PSD stiffness and diagonal mass (as vector)."""
        n = self.n
        if self.faces is not None:
            f = self.faces
            w = np.diff(f)
            c = 0.5 * (f[:-1] + f[1:])
            d = np.diff(c)
            d_lo, d_hi = c[0] - f[0], f[-1] - c[-1]
            d_wrap = d_lo + d_hi
        else:
            h = self.h
            w = np.full(n, h)
            d = np.full(max(n - 1, 0), h)
            if self.stagger == "cell":
                d_lo = d_hi = 0.5 * h
            else:
                d_lo = d_hi = h
            d_wrap = h
        K = np.zeros((n, n))
        if n > 1:
            cnd = 1.0 / d
            idx = np.arange(n - 1)
            K[idx, idx] += cnd
            K[idx + 1, idx + 1] += cnd
            K[idx, idx + 1] -= cnd
            K[idx + 1, idx] -= cnd
        if self.lo == PERIODIC:
            c = 1.0 / d_wrap
            if n == 1:
                pass
            else:
                K[0, 0] += c
                K[-1, -1] += c
                K[0, -1] -= c
                K[-1, 0] -= c
        else:
            if self.lo == DIRICHLET:
                K[0, 0] += 1.0 / d_lo
            if self.hi == DIRICHLET:
                K[-1, -1] += 1.0 / d_hi
        return K, w

    def fast_transform(self):
        """(kind, type, eigenvalues) for an O(N log N) transform, else None."""
        if not self.uniform:
            return None
        n, h = self.n, (self.h if self.faces is None else self.faces[1] - self.faces[0])
        k = np.arange(n)
        s = 4.0 / h ** 2
        if self.lo == PERIODIC:
            return ("fft", 0, s * np.sin(np.pi * k / n) ** 2)
        if self.stagger == "node":
            if self.lo == DIRICHLET and self.hi == DIRICHLET:
                return ("dst", 1, s * np.sin(np.pi * (k + 1) / (2 * (n + 1))) ** 2)
            return None
        pair = (self.lo, self.hi)
        if pair == (NEUMANN, NEUMANN):
            return ("dct", 2, s * np.sin(np.pi * k / (2 * n)) ** 2)
        if pair == (DIRICHLET, DIRICHLET):
            return ("dst", 2, s * np.sin(np.pi * (k + 1) / (2 * n)) ** 2)
        if pair == (NEUMANN, DIRICHLET):
            return ("dct", 4, s * np.sin(np.pi * (2 * k + 1) / (4 * n)) ** 2)
        if pair == (DIRICHLET, NEUMANN):
            return ("dst", 4, s * np.sin(np.pi * (2 * k + 1) / (4 * n)) ** 2)
        return None


class _AxisPlan:
    def __init__(self, axis: Axis1D, axis_index: int, backend, dtype, method: str):
        self.axis = axis
        self.index = axis_index
        self.method = None
        xp = backend.xp
        fast = axis.fast_transform()
        if method in ("auto", "fast") and fast is not None:
            kind, typ, lam = fast
            ok = kind == "fft" or backend.supports_transform(kind, typ)
            if ok:
                self.method = kind
                self.type = typ
                self.lam = lam
        if self.method is None:
            if method == "fast" and fast is None:
                raise ValueError(f"axis {axis_index}: no fast transform for this axis")
            K, w = axis.matrices()
            # Symmetric form: Ms^{-1/2} K Ms^{-1/2} q = lambda q, V = Ms^{-1/2} q.
            winv = 1.0 / np.sqrt(w)
            A = (K * winv[:, None]) * winv[None, :]
            lam, Q = np.linalg.eigh(A)
            lam = np.maximum(lam, 0.0)
            V = Q * winv[:, None]  # V^T M V = I
            self.method = "eig"
            self.lam = lam
            self.fwd = xp.asarray((V * w[:, None]).T, dtype=dtype)  # V^T M
            self.inv = xp.asarray(V, dtype=dtype)
        self.lam_dev = xp.asarray(self.lam, dtype=dtype)

    @property
    def is_complex(self) -> bool:
        return self.method == "fft"


def _apply_matrix(xp, W, a, axis):
    if a.ndim == 2:
        return W @ a if axis == 0 else a @ W.T
    moved = xp.moveaxis(a, axis, -1)
    return xp.moveaxis(moved @ W.T, -1, axis)


class SeparableSolver:
    """Exact solver for ``(sigma - L) u = f`` with L the tensor-product Laplacian.

    ``methods`` may be ``'auto'`` (fast transforms where possible), ``'eig'``
    (force dense GEMM diagonalisation) or a per-axis list of those.
    """

    def __init__(self, axes: Sequence[Axis1D], backend, dtype=np.float64, methods="auto"):
        self.backend = backend
        self.xp = backend.xp
        self.dtype = np.dtype(dtype)
        if isinstance(methods, str):
            methods = [methods] * len(axes)
        self.plans = [_AxisPlan(ax, i, backend, self.dtype, m) for i, (ax, m) in enumerate(zip(axes, methods))]
        self.shape = tuple(ax.n for ax in axes)
        xp = self.xp
        lam = xp.zeros(self.shape, dtype=self.dtype)
        for p in self.plans:
            sh = [1] * len(self.shape)
            sh[p.index] = -1
            lam = lam + p.lam_dev.reshape(sh)
        self.lam_sum = lam
        self.singular = all(ax.lo in (NEUMANN, PERIODIC) and ax.hi in (NEUMANN, PERIODIC) for ax in axes)
        self._denom_cache = (None, None)

    @property
    def methods(self):
        return [p.method + (str(p.type) if p.method in ("dct", "dst") else "") for p in self.plans]

    def _inv_denominator(self, sigma: float):
        if self._denom_cache[0] == sigma:
            return self._denom_cache[1]
        xp = self.xp
        den = self.lam_sum + self.dtype.type(sigma)
        if sigma == 0.0 and self.singular:
            tol = 1e-9 * float(self.backend.scalar(den.max()))
            den = xp.where(den > tol, den, xp.inf)
        inv = 1.0 / den
        self._denom_cache = (sigma, inv)
        return inv

    def solve(self, f, sigma: float = 0.0):
        """Return u with (sigma - L) u = f. For the singular all-Neumann/periodic
        Poisson problem the mean of ``f`` is discarded and mean(u) = 0."""
        xp, fft, kw = self.xp, self.backend.fft, self.backend.fft_kwargs
        a = f.astype(self.dtype, copy=False)
        real_plans = [p for p in self.plans if not p.is_complex]
        cplx_plans = [p for p in self.plans if p.is_complex]
        for p in real_plans:
            if p.method == "eig":
                a = _apply_matrix(xp, p.fwd, a, p.index)
            elif p.method == "dct":
                a = fft.dct(a, type=p.type, axis=p.index, norm="ortho", **kw)
            else:
                a = fft.dst(a, type=p.type, axis=p.index, norm="ortho", **kw)
        for p in cplx_plans:
            a = fft.fft(a, axis=p.index, **kw)
        a = a * self._inv_denominator(float(sigma))
        for p in cplx_plans:
            a = fft.ifft(a, axis=p.index, **kw)
        if cplx_plans:
            a = a.real
        for p in reversed(real_plans):
            if p.method == "eig":
                a = _apply_matrix(xp, p.inv, a, p.index)
            elif p.method == "dct":
                a = fft.idct(a, type=p.type, axis=p.index, norm="ortho", **kw)
            else:
                a = fft.idst(a, type=p.type, axis=p.index, norm="ortho", **kw)
        return xp.ascontiguousarray(a.astype(self.dtype, copy=False))

    def apply(self, u, sigma: float = 0.0):
        """Matrix-free application of (sigma - L) with homogeneous BCs (for checks)."""
        xp = self.xp
        out = sigma * u
        for p in self.plans:
            K, w = p.axis.matrices()
            Lm = xp.asarray(K / w[:, None], dtype=self.dtype)
            out = out + _apply_matrix(xp, Lm, u, p.index)
        return out
