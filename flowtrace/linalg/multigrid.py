"""Geometric multigrid-preconditioned CG for cell-centred 2-D operators.

Used where the fast separable solver does not apply: immersed obstacles
(masked cells), variable coefficients. The operator is written in integrated
(finite-volume) form

    (A x)_P = sum_f c_f (x_P - x_N) + s_P x_P

with face conductances ``c_f`` (0 across solid/Neumann faces; the Dirichlet
half-cell conductance on Dirichlet boundary faces) and a diagonal shift
``s_P`` (Helmholtz terms). Coarse levels aggregate 2x2 cells; conductances
combine in parallel/series, shifts add, so the hierarchy is built once in
O(N). Smoothing is red-black Gauss-Seidel, fully vectorised so the same code
runs on NumPy and CuPy. The cycle is symmetric, hence a valid CG
preconditioner, and the coarsest level is a sparse direct factorisation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..structured import kernels as _k
from .krylov import pcg


class _Level:
    def __init__(self, xp, cx, cy, shift, active, jit=False):
        self.xp = xp
        self.jit = jit
        self.cx, self.cy, self.shift, self.active = cx, cy, shift, active
        self.nx, self.ny = active.shape
        diag = cx[:-1, :] + cx[1:, :] + cy[:, :-1] + cy[:, 1:] + shift
        self.diag = xp.where(active, diag, 1.0)
        self.inv_diag = 1.0 / xp.where(self.diag > 0, self.diag, 1.0)
        ii, jj = np.indices((self.nx, self.ny))
        red = ((ii + jj) % 2 == 0)
        self.red = xp.asarray(red)
        self.black = xp.asarray(~red)

    def neighbour_sum(self, x):
        cx, cy = self.cx, self.cy
        s = self.xp.zeros_like(x)
        s[1:, :] += cx[1:-1, :] * x[:-1, :]
        s[:-1, :] += cx[1:-1, :] * x[1:, :]
        s[:, 1:] += cy[:, 1:-1] * x[:, :-1]
        s[:, :-1] += cy[:, 1:-1] * x[:, 1:]
        return s

    def apply(self, x):
        if self.jit:
            out = self.xp.empty_like(x)
            _k.apply_nb(x, self.cx, self.cy, self.diag, out)
            return out
        return self.diag * x - self.neighbour_sum(x)

    def smooth(self, x, b, colours):
        xp = self.xp
        if self.jit:
            for mask in colours:
                _k.rbgs_nb(x, b, self.cx, self.cy, self.inv_diag, 0 if mask is self.red else 1)
            return x
        for mask in colours:
            xnew = (b + self.neighbour_sum(x)) * self.inv_diag
            xp.copyto(x, xnew, where=mask)
        return x

    def coarsen(self):
        xp = self.xp
        cx, cy = self.cx, self.cy
        cxc = 0.5 * (cx[0::2, 0::2] + cx[0::2, 1::2])
        cyc = 0.5 * (cy[0::2, 0::2] + cy[1::2, 0::2])
        s = self.shift
        sc = s[0::2, 0::2] + s[1::2, 0::2] + s[0::2, 1::2] + s[1::2, 1::2]
        a = self.active
        ac = a[0::2, 0::2] | a[1::2, 0::2] | a[0::2, 1::2] | a[1::2, 1::2]
        return _Level(xp, xp.ascontiguousarray(cxc), xp.ascontiguousarray(cyc),
                      xp.ascontiguousarray(sc), ac, self.jit)

    def to_sparse(self, to_host):
        cx, cy = to_host(self.cx), to_host(self.cy)
        diag = to_host(self.diag)
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(nx, ny)
        rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
        c = cx[1:-1, :]
        rows += [idx[1:, :].ravel(), idx[:-1, :].ravel()]
        cols += [idx[:-1, :].ravel(), idx[1:, :].ravel()]
        vals += [-c.ravel(), -c.ravel()]
        c = cy[:, 1:-1]
        rows += [idx[:, 1:].ravel(), idx[:, :-1].ravel()]
        cols += [idx[:, :-1].ravel(), idx[:, 1:].ravel()]
        vals += [-c.ravel(), -c.ravel()]
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nx * ny, nx * ny))
        return A


class MGPoisson2D:
    """Multigrid-preconditioned CG for the integrated-form operator above."""

    def __init__(self, cx, cy, shift, active, backend, coarse_max: int = 1024,
                 pre: int = 2, post: int = 2):
        self.backend = backend
        xp = self.xp = backend.xp
        dtype = cx.dtype
        active = xp.asarray(active, dtype=bool)
        # No conductance into or out of inactive cells.
        cx = cx.copy()
        cy = cy.copy()
        cx[1:-1, :] *= (active[1:, :] & active[:-1, :])
        cx[0, :] *= active[0, :]
        cx[-1, :] *= active[-1, :]
        cy[:, 1:-1] *= (active[:, 1:] & active[:, :-1])
        cy[:, 0] *= active[:, 0]
        cy[:, -1] *= active[:, -1]
        shift = xp.broadcast_to(xp.asarray(shift, dtype=dtype), active.shape) * active
        jit = bool(getattr(backend, "jit", False)) and _k.HAVE_NUMBA
        self.levels = [_Level(xp, xp.ascontiguousarray(cx), xp.ascontiguousarray(cy),
                              xp.ascontiguousarray(shift), active, jit)]
        while True:
            L = self.levels[-1]
            if L.nx * L.ny <= coarse_max or L.nx % 2 or L.ny % 2 or min(L.nx, L.ny) < 4:
                break
            self.levels.append(L.coarsen())
        self.pre, self.post = pre, post
        Lc = self.levels[-1]
        A = Lc.to_sparse(backend.to_numpy)
        has_dirichlet = bool(backend.to_numpy(
            (cx[0, :].sum() + cx[-1, :].sum() + cy[:, 0].sum() + cy[:, -1].sum() + shift.sum())) > 0)
        self.singular = not has_dirichlet
        if self.singular:
            eps = 1e-10 * float(abs(A.diagonal()).mean())
            A = A + eps * sp.identity(A.shape[0], format="csc")
        self._coarse = spla.splu(A.tocsc())
        self.active = active
        self.n_active = int(backend.to_numpy(active).sum())
        self.dtype = dtype

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def _coarse_solve(self, b):
        bh = self.backend.to_numpy(b).ravel().astype(np.float64)
        x = self._coarse.solve(bh).reshape(b.shape)
        return self.xp.asarray(x, dtype=self.dtype)

    def _vcycle(self, k, b):
        L = self.levels[k]
        if k == len(self.levels) - 1:
            return self._coarse_solve(b)
        xp = self.xp
        x = xp.zeros_like(b)
        for _ in range(self.pre):
            L.smooth(x, b, (L.red, L.black))
        r = (b - L.apply(x)) * L.active
        rc = r[0::2, 0::2] + r[1::2, 0::2] + r[0::2, 1::2] + r[1::2, 1::2]
        ec = self._vcycle(k + 1, xp.ascontiguousarray(rc))
        e = xp.repeat(xp.repeat(ec, 2, axis=0), 2, axis=1)
        x += e * L.active
        for _ in range(self.post):
            L.smooth(x, b, (L.black, L.red))
        return x * L.active

    def precondition(self, r):
        return self._vcycle(0, r)

    def project(self, v):
        if not self.singular:
            return v
        m = (v * self.active).sum() / self.n_active
        return (v - m) * self.active

    def apply(self, x):
        return self.levels[0].apply(x)

    def solve(self, b, x0=None, tol: float = 1e-8, maxiter: int = 200):
        b = b * self.active
        if self.singular:
            b = self.project(b)
        x, info = pcg(self.apply, b, self.xp, M=self.precondition, x0=x0, tol=tol,
                      maxiter=maxiter, project=self.project if self.singular else None)
        if self.singular:
            x = self.project(x)
        return x, info
