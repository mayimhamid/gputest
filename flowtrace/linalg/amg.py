"""Smoothed-aggregation algebraic multigrid (setup on CPU, cycles on any backend).

Used as the preconditioner for the pressure-correction equation of the
unstructured SIMPLE solver (the role Fluent's AMG plays). The hierarchy is
built on the host with SciPy sparse matrices; the V-cycle only needs sparse
mat-vecs and vector updates, so it runs unchanged on CuPy/cuSPARSE.
If ``pyamg`` is installed and the backend is the CPU, pyamg's (faster,
compiled) setup is used instead.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:  # optional JIT for the aggregation loop
    from numba import njit

    _jit = njit(cache=True)
except Exception:  # pragma: no cover
    def _jit(f):
        return f


@_jit
def _aggregate(indptr, indices, n):
    agg = -np.ones(n, dtype=np.int64)
    na = 0
    # pass 1: seed aggregates from nodes whose strong neighbourhood is free
    for i in range(n):
        if agg[i] >= 0:
            continue
        free = True
        for k in range(indptr[i], indptr[i + 1]):
            if agg[indices[k]] >= 0:
                free = False
                break
        if free:
            agg[i] = na
            for k in range(indptr[i], indptr[i + 1]):
                agg[indices[k]] = na
            na += 1
    # pass 2: attach leftovers to a neighbouring aggregate
    for i in range(n):
        if agg[i] >= 0:
            continue
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if agg[j] >= 0:
                agg[i] = agg[j]
                break
    # pass 3: isolated nodes become their own aggregates
    for i in range(n):
        if agg[i] < 0:
            agg[i] = na
            na += 1
    return agg, na


def _strength(A: sp.csr_matrix, theta: float) -> sp.csr_matrix:
    A = A.tocoo()
    d = np.abs(A.diagonal())
    mask = (A.row != A.col) & (np.abs(A.data) >= theta * np.sqrt(d[A.row] * d[A.col]))
    S = sp.csr_matrix((np.ones(mask.sum()), (A.row[mask], A.col[mask])), shape=A.shape)
    S.sort_indices()
    return S


def _spectral_radius(M, iters=15, seed=0) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(M.shape[0])
    lam = 1.0
    for _ in range(iters):
        y = M @ x
        lam = np.linalg.norm(y) / max(np.linalg.norm(x), 1e-300)
        x = y / max(np.linalg.norm(y), 1e-300)
    return lam


class SmoothedAggregationAMG:
    def __init__(self, A, backend, theta: float = 0.08, max_coarse: int = 300,
                 max_levels: int = 20, smooth_sweeps: int = 2, omega: float = 0.8,
                 singular: bool = False):
        self.backend = backend
        self.xp = backend.xp
        self.omega = omega
        self.sweeps = smooth_sweeps
        A = sp.csr_matrix(A, dtype=np.float64)
        host_levels = []
        while A.shape[0] > max_coarse and len(host_levels) < max_levels - 1:
            S = _strength(A, theta)
            agg, na = _aggregate(S.indptr.astype(np.int64), S.indices.astype(np.int64), A.shape[0])
            if na >= A.shape[0] * 0.9:
                break
            T = sp.csr_matrix((np.ones(A.shape[0]), (np.arange(A.shape[0]), agg)), shape=(A.shape[0], na))
            cnt = np.asarray(T.sum(axis=0)).ravel()
            T = T @ sp.diags(1.0 / np.sqrt(cnt))
            Dinv = 1.0 / A.diagonal()
            DA = sp.diags(Dinv) @ A
            rho = _spectral_radius(DA)
            P = (T - (4.0 / 3.0 / rho) * (DA @ T)).tocsr()
            Ac = (P.T @ A @ P).tocsr()
            host_levels.append((A, P, Dinv))
            A = Ac
        self.coarse_n = A.shape[0]
        Ad = A.toarray()
        if singular:
            Ad = Ad + 1e-10 * np.abs(np.diag(Ad)).mean() * np.eye(Ad.shape[0])
        try:
            self._coarse_inv = np.linalg.inv(Ad)
        except np.linalg.LinAlgError:
            self._coarse_inv = np.linalg.pinv(Ad)
        self.levels = []
        for Ah, Ph, Dinv in host_levels:
            self.levels.append((self._dev_matrix(Ah), self._dev_matrix(Ph), self._dev_matrix(Ph.T.tocsr()),
                                self.xp.asarray(Dinv)))
        self.coarse_inv = self.xp.asarray(self._coarse_inv)

    def _dev_matrix(self, M):
        if self.backend.is_gpu:
            import cupyx.scipy.sparse as csp  # type: ignore

            return csp.csr_matrix(M)
        return M

    @property
    def n_levels(self) -> int:
        return len(self.levels) + 1

    def _cycle(self, k, b):
        if k == len(self.levels):
            return self.coarse_inv @ b
        A, P, R, Dinv = self.levels[k]
        w = self.omega
        x = w * Dinv * b
        for _ in range(self.sweeps - 1):
            x = x + w * Dinv * (b - A @ x)
        r = b - A @ x
        x = x + P @ self._cycle(k + 1, R @ r)
        for _ in range(self.sweeps):
            x = x + w * Dinv * (b - A @ x)
        return x

    def __call__(self, b):
        dt = b.dtype
        return self._cycle(0, b.astype(self.xp.float64)).astype(dt, copy=False)


def make_amg(A, backend, singular=False):
    """Return a preconditioner callable for SPD (or singular-consistent) ``A``."""
    A = getattr(A, "csr", A)  # flowtrace.native.NativeCSR wraps a SciPy matrix
    if not backend.is_gpu:
        try:
            import pyamg  # type: ignore

            # pyamg estimates spectral radii from random vectors drawn from NumPy's global
            # RNG, so the preconditioner (and every iterate) differed from run to run
            # (measured ~6e-8 in u after 20 iterations). Seed it for reproducible runs and
            # leave the caller's global RNG state untouched.
            state = np.random.get_state()
            np.random.seed(20261026)
            try:
                ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(A), max_coarse=300)
            finally:
                np.random.set_state(state)
            M = ml.aspreconditioner(cycle="V")
            M.n_levels = len(ml.levels)
            return M
        except Exception:
            pass
    return SmoothedAggregationAMG(A, backend, singular=singular)
