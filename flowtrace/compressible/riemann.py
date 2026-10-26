"""Exact solution of the 1-D Riemann problem for an ideal gas (Toro, *Riemann Solvers and
Numerical Methods for Fluid Dynamics*, 3rd ed., ch. 4). Used to verify the compressible solver.
"""

from __future__ import annotations

import numpy as np


def _f(p, rk, pk, ck, g):
    """Pressure function f_K(p) and its derivative (shock or rarefaction branch)."""
    if p > pk:  # shock
        A, B = 2.0 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
        sq = np.sqrt(A / (p + B))
        return (p - pk) * sq, sq * (1.0 - 0.5 * (p - pk) / (B + p))
    pr = p / pk  # rarefaction
    return 2 * ck / (g - 1) * (pr ** ((g - 1) / (2 * g)) - 1), 1.0 / (rk * ck) * pr ** (-(g + 1) / (2 * g))


def star_state(left, right, gamma=1.4, tol=1e-12):
    """(p*, u*) between the left and right states (rho, u, p)."""
    rL, uL, pL = left
    rR, uR, pR = right
    g = gamma
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    if 2 * (cL + cR) / (g - 1) <= uR - uL:
        raise ValueError("the initial data generate vacuum")
    p = max(tol, 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rL + rR) * (cL + cR))  # PVRS guess
    for _ in range(100):
        fL, dL = _f(p, rL, pL, cL, g)
        fR, dR = _f(p, rR, pR, cR, g)
        pn = max(tol, p - (fL + fR + uR - uL) / (dL + dR))
        if abs(pn - p) <= tol * 0.5 * (pn + p):
            p = pn
            break
        p = pn
    fL, _ = _f(p, rL, pL, cL, g)
    fR, _ = _f(p, rR, pR, cR, g)
    return p, 0.5 * (uL + uR) + 0.5 * (fR - fL)


def sample(x, t, left, right, x0=0.5, gamma=1.4):
    """Exact (rho, u, p) at positions ``x`` and time ``t`` > 0."""
    rL, uL, pL = left
    rR, uR, pR = right
    g = gamma
    ps, us = star_state(left, right, g)
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    s = (np.asarray(x, dtype=float) - x0) / t
    rho, u, p = np.empty_like(s), np.empty_like(s), np.empty_like(s)
    for i, si in enumerate(s):
        if si <= us:  # left of the contact
            if ps > pL:  # left shock
                SL = uL - cL * np.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g))
                if si <= SL:
                    rho[i], u[i], p[i] = rL, uL, pL
                else:
                    rho[i] = rL * ((ps / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pL + 1))
                    u[i], p[i] = us, ps
            else:  # left rarefaction
                cs = cL * (ps / pL) ** ((g - 1) / (2 * g))
                if si <= uL - cL:
                    rho[i], u[i], p[i] = rL, uL, pL
                elif si >= us - cs:
                    rho[i], u[i], p[i] = rL * (ps / pL) ** (1 / g), us, ps
                else:
                    c = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - si))
                    u[i] = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + si)
                    rho[i] = rL * (c / cL) ** (2 / (g - 1))
                    p[i] = pL * (c / cL) ** (2 * g / (g - 1))
        else:  # right of the contact
            if ps > pR:  # right shock
                SR = uR + cR * np.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
                if si >= SR:
                    rho[i], u[i], p[i] = rR, uR, pR
                else:
                    rho[i] = rR * ((ps / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pR + 1))
                    u[i], p[i] = us, ps
            else:  # right rarefaction
                cs = cR * (ps / pR) ** ((g - 1) / (2 * g))
                if si >= uR + cR:
                    rho[i], u[i], p[i] = rR, uR, pR
                elif si <= us + cs:
                    rho[i], u[i], p[i] = rR * (ps / pR) ** (1 / g), us, ps
                else:
                    c = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - si))
                    u[i] = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + si)
                    rho[i] = rR * (c / cR) ** (2 / (g - 1))
                    p[i] = pR * (c / cR) ** (2 * g / (g - 1))
    return rho, u, p


def oblique_shock(mach: float, theta_deg: float, gamma: float = 1.4):
    """Weak attached oblique shock: (beta_deg, p2/p1, rho2/rho1, M2) for wedge half-angle theta."""
    th = np.radians(theta_deg)
    g, M = gamma, mach
    lo, hi = np.arcsin(1.0 / M) + 1e-9, np.radians(90.0)
    # tan(theta) = 2 cot(b) (M^2 sin^2 b - 1) / (M^2 (g + cos 2b) + 2): weak root by bisection below beta_max
    f = lambda b: 2 / np.tan(b) * (M ** 2 * np.sin(b) ** 2 - 1) / (M ** 2 * (g + np.cos(2 * b)) + 2) - np.tan(th)  # noqa
    bs = np.linspace(lo, hi, 20001)
    vals = np.array([f(b) for b in bs])
    k = int(np.argmax(vals))  # beta at maximum deflection
    if vals[k] < 0:
        raise ValueError("detached shock: wedge angle above the maximum deflection")
    a, b = lo, bs[k]
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(m) > 0:
            b = m
        else:
            a = m
    beta = 0.5 * (a + b)
    mn1 = M * np.sin(beta)
    p21 = 1 + 2 * g / (g + 1) * (mn1 ** 2 - 1)
    r21 = (g + 1) * mn1 ** 2 / ((g - 1) * mn1 ** 2 + 2)
    mn2 = np.sqrt((1 + (g - 1) / 2 * mn1 ** 2) / (g * mn1 ** 2 - (g - 1) / 2))
    M2 = mn2 / np.sin(beta - th)
    return float(np.degrees(beta)), float(p21), float(r21), float(M2)


def isentropic_area_mach(area_ratio: float, supersonic: bool, gamma: float = 1.4) -> float:
    """Mach number for A/A* (quasi-1-D isentropic nozzle flow)."""
    g = gamma
    f = lambda M: 1 / M * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M ** 2)) ** ((g + 1) / (2 * (g - 1))) - area_ratio  # noqa
    a, b = (1.0, 50.0) if supersonic else (1e-8, 1.0)
    for _ in range(200):
        m = 0.5 * (a + b)
        if (f(a) > 0) == (f(m) > 0):
            a = m
        else:
            b = m
    return 0.5 * (a + b)
