"""Automatic validation against reference data for the built-in cases."""

from __future__ import annotations

import math

import numpy as np

from .reference import DVD, compare_ghia, ghia


def strouhal(times, signal, length: float = 1.0, speed: float = 1.0, discard: float = 0.4):
    """Dominant frequency of a probe signal (after discarding the start-up)."""
    t = np.asarray(times, dtype=float)
    s = np.asarray(signal, dtype=float)
    if t.size < 32:
        return None
    keep = t >= t[0] + discard * (t[-1] - t[0])
    t, s = t[keep], s[keep]
    s = s - s.mean()
    if np.allclose(s, 0, atol=1e-8):
        return None
    # resample uniformly, then FFT peak with parabolic refinement
    n = int(2 ** math.ceil(math.log2(max(t.size, 64))))
    tu = np.linspace(t[0], t[-1], n)
    su = np.interp(tu, t, s) * np.hanning(n)
    spec = np.abs(np.fft.rfft(su))
    freqs = np.fft.rfftfreq(n, tu[1] - tu[0])
    k = int(np.argmax(spec[1:]) + 1)
    if 1 <= k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        shift = 0.0
    f = (k + shift) * (freqs[1] - freqs[0])
    return f * length / speed


def validate_structured(solver, probe_log=None) -> dict:
    """Return {"checks": [...]} comparing a finished structured run with references."""
    c = solver.case
    checks = []
    name = c.name.lower()
    if name.startswith("lid-driven cavity"):
        re = 1.0 / c.nu
        if ghia(re) is not None:
            (y, u), (x, v) = solver.centerline()
            err = compare_ghia(re, y, u, x, v)
            checks.append({"name": f"Centreline u vs Ghia et al. (Re={re:g})", "value": err["u_max_err"],
                           "unit": "max |du|", "pass": err["u_max_err"] < 0.03})
            checks.append({"name": f"Centreline v vs Ghia et al. (Re={re:g})", "value": err["v_max_err"],
                           "unit": "max |dv|", "pass": err["v_max_err"] < 0.03})
    if name.startswith("natural convection"):
        ra = c.buoyancy[1] / c.nu
        key = min(DVD, key=lambda k: abs(math.log10(k) - math.log10(max(ra, 1.0))))
        if abs(key - ra) / key < 0.05:
            nu = solver.nusselt("left")
            ref = DVD[key]["nu"]
            checks.append({"name": f"Mean Nusselt vs de Vahl Davis (Ra={key:g})", "value": nu, "reference": ref,
                           "unit": "Nu", "pass": nu is not None and abs(nu - ref) / ref < 0.03})
    if name.startswith("taylor-green"):
        G = 2
        nx, ny = c.nx, c.ny
        xf = np.arange(nx + 1) * solver.dx
        yc = (np.arange(ny) + 0.5) * solver.dy
        ue = np.sin(xf)[:, None] * np.cos(yc)[None, :] * np.exp(-2 * c.nu * solver.t)
        uh = solver.be.to_numpy(solver.U)[G:G + nx + 1, G:G + ny]
        err = float(np.abs(uh - ue).max())
        checks.append({"name": "Taylor-Green L-inf error vs exact solution", "value": err, "unit": "|u - u_exact|",
                       "pass": err < 5.0 / nx ** 2})
    if name.startswith("channel flow"):
        u, _v = solver.cell_velocity()
        u = solver.be.to_numpy(u)
        yc = (np.arange(c.ny) + 0.5) / c.ny
        ue = 6.0 * yc * (1 - yc) * c.left.velocity
        err = float(np.abs(u[-2] - ue).max())
        checks.append({"name": "Outlet profile vs Poiseuille", "value": err, "unit": "max |du|",
                       "pass": err < 0.05 * c.left.velocity})
    if probe_log and len(probe_log.get("t", [])) > 64 and name.startswith("cylinder wake"):
        st = strouhal(probe_log["t"], probe_log["v"], length=2 * c.obstacles[0].r, speed=c.left.velocity)
        if st is not None:
            checks.append({"name": "Strouhal number (probe in the wake)", "value": st, "reference": 0.165,
                           "unit": "St", "pass": 0.13 < st < 0.20})
    div = solver.max_divergence()
    checks.append({"name": "Discrete continuity max|div u|", "value": div, "unit": "1/s",
                   "pass": div < (1e-8 if solver.dtype == np.float64 else 1e-3)})
    return {"checks": checks}
