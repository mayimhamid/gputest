"""Backend-agnostic Krylov solvers (NumPy or CuPy arrays, callable operators).

Only the scalars needed for the iteration control (dot products, norms) are
transferred to the host; vectors stay on the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class KrylovInfo:
    iterations: int
    residual: float  # final relative residual ||b - A x|| / ||b||
    converged: bool


def _dot(xp, a, b):
    return float(xp.vdot(a, b).real)


def pcg(A: Callable, b, xp, M: Optional[Callable] = None, x0=None, tol: float = 1e-8,
        maxiter: int = 500, project: Optional[Callable] = None):
    """Preconditioned conjugate gradients for SPD (or consistent PSD) systems.

    ``project`` removes the null space (e.g. the constant for all-Neumann
    pressure) from residuals so that singular-but-consistent systems converge.
    """
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x) if x0 is not None else b.copy()
    if project is not None:
        r = project(r)
    bnorm = float(xp.linalg.norm(b)) or 1.0
    rnorm = float(xp.linalg.norm(r))
    if rnorm / bnorm < tol:
        return x, KrylovInfo(0, rnorm / bnorm, True)
    z = M(r) if M is not None else r
    if project is not None:
        z = project(z)
    p = z.copy()
    rz = _dot(xp, r, z)
    for k in range(1, maxiter + 1):
        Ap = A(p)
        pAp = _dot(xp, p, Ap)
        if pAp <= 0.0:
            return x, KrylovInfo(k, rnorm / bnorm, False)
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        rnorm = float(xp.linalg.norm(r))
        if rnorm / bnorm < tol:
            return x, KrylovInfo(k, rnorm / bnorm, True)
        z = M(r) if M is not None else r
        if project is not None:
            z = project(z)
        rz_new = _dot(xp, r, z)
        beta = rz_new / rz
        rz = rz_new
        p = z + beta * p
    return x, KrylovInfo(maxiter, rnorm / bnorm, False)


def bicgstab(A: Callable, b, xp, M: Optional[Callable] = None, x0=None, tol: float = 1e-8,
             maxiter: int = 500, relative_to_initial: bool = False):
    """Right-preconditioned BiCGSTAB for non-symmetric systems (e.g. momentum).

    With ``relative_to_initial`` the tolerance applies to ||r|| / ||r0|| (the
    termination rule of segregated CFD solvers: reduce the residual of the
    current outer iteration by a fixed factor).
    """
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x) if x0 is not None else b.copy()
    rnorm = float(xp.linalg.norm(r))
    bnorm = (rnorm if relative_to_initial else float(xp.linalg.norm(b))) or 1.0
    if rnorm / bnorm < tol:
        return x, KrylovInfo(0, rnorm / bnorm, True)
    r_hat = r.copy()
    rho = alpha = omega = 1.0
    v = xp.zeros_like(b)
    p = xp.zeros_like(b)
    for k in range(1, maxiter + 1):
        rho_new = _dot(xp, r_hat, r)
        if rho_new == 0.0:
            break
        beta = (rho_new / rho) * (alpha / omega)
        rho = rho_new
        p = r + beta * (p - omega * v)
        phat = M(p) if M is not None else p
        v = A(phat)
        denom = _dot(xp, r_hat, v)
        if denom == 0.0:
            break
        alpha = rho / denom
        s = r - alpha * v
        snorm = float(xp.linalg.norm(s))
        if snorm / bnorm < tol:
            x += alpha * phat
            return x, KrylovInfo(k, snorm / bnorm, True)
        shat = M(s) if M is not None else s
        t = A(shat)
        tt = _dot(xp, t, t)
        omega = _dot(xp, t, s) / tt if tt > 0 else 0.0
        x += alpha * phat + omega * shat
        r = s - omega * t
        rnorm = float(xp.linalg.norm(r))
        if rnorm / bnorm < tol:
            return x, KrylovInfo(k, rnorm / bnorm, True)
        if omega == 0.0:
            break
    return x, KrylovInfo(k, rnorm / bnorm, False)
