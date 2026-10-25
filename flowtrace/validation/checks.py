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


def validate_unstructured(solver, info=None) -> dict:
    """Checks for a finished unstructured (SIMPLE/SIMPLEC) run."""
    info = info or {}
    checks = []
    kind = info.get("kind")
    c = solver.case
    if kind == "channel":
        xc = solver.cell_centres()
        f = solver.fields()
        sel = xc[:, 0] < xc[:, 0].min() + 1e-9 + (xc[:, 0].max() - xc[:, 0].min()) / 2
        y, u = xc[sel, 1], f["x-velocity"][sel]
        o = np.argsort(y)
        y, u = y[o], u[o]
        yy = np.concatenate([[0.0], y, [2.0]])
        uu = np.concatenate([[0.0], u, [0.0]])
        ub = float(np.sum(0.5 * (uu[1:] + uu[:-1]) * np.diff(yy)) / 2.0)
        ref = info["ub_plus_ref"]
        checks.append({"name": f"Bulk velocity U_b+ vs DNS (Re_tau={info['re_tau']:g})", "value": ub,
                       "reference": ref, "unit": "U_b/u_tau", "pass": abs(ub - ref) / ref < 0.03})
        yp = y[y < 1.0] * info["re_tau"]
        up = u[y < 1.0]
        u100 = float(np.interp(100.0, yp, up))
        ref100 = np.log(100.0) / 0.41 + 5.2
        checks.append({"name": "Log law at y+ = 100", "value": u100, "reference": ref100, "unit": "u+",
                       "pass": abs(u100 - ref100) / ref100 < 0.06})
        rep = solver.wall_report()
        fx = sum(r["force_total"][0] for r in rep.values())
        vol = float(solver.fv.geometry.total_volume)
        drive = abs(c.periodic_pressure_gradient[0]) * vol
        checks.append({"name": "Wall shear balances the driving pressure gradient", "value": fx / drive,
                       "reference": 1.0, "unit": "F_wall / (dp/dx V)", "pass": abs(fx / drive - 1.0) < 0.01})
    elif kind == "flat_plate":
        prof = solver.wall_profile("plate")
        x = 0.97
        cf = float(np.interp(x, prof["x"], prof["cf"]))
        ref = 0.00271  # NASA TMR 2DZP, SST, finest grid, x = 0.97
        checks.append({"name": "Skin friction at x = 0.97 vs NASA TMR SST", "value": cf, "reference": ref,
                       "unit": "Cf", "pass": abs(cf - ref) / ref < 0.08})
        x2 = 1.5
        cf2 = float(np.interp(x2, prof["x"], prof["cf"]))
        sch = 0.0592 * (x2 * info["re_l"]) ** -0.2
        checks.append({"name": "Skin friction at x = 1.5 vs Schlichting 0.0592 Re_x^-1/5", "value": cf2,
                       "reference": float(sch), "unit": "Cf", "pass": abs(cf2 - sch) / sch < 0.08})
        checks.append({"name": "Max wall y+ on the plate", "value": float(prof["yplus"].max()), "reference": 1.0,
                       "unit": "y+", "pass": True})
    elif kind == "radiation_slab":
        rep = solver.wall_report()
        q = rep["hot"]["heat_flow_rad"] / info["height"]
        checks.append({"name": "P-1 radiative flux vs exact slab solution", "value": q, "reference": info["q_exact"],
                       "unit": "W/m^2", "pass": abs(q - info["q_exact"]) / info["q_exact"] < 0.01})
        bal = (rep["hot"]["heat_flow_rad"] + rep["cold"]["heat_flow_rad"]) / rep["hot"]["heat_flow_rad"]
        checks.append({"name": "Radiative energy balance (hot + cold)", "value": bal, "reference": 0.0,
                       "unit": "relative", "pass": abs(bal) < 1e-4})
    elif kind == "taylor_green":
        xc = solver.cell_centres()
        ue = np.sin(xc[:, 0]) * np.cos(xc[:, 1]) * np.exp(-2 * info["nu"] * solver.time)
        err = float(np.abs(solver.fields()["x-velocity"] - ue).max())
        checks.append({"name": "Transient Taylor-Green error vs exact (periodic mesh)", "value": err,
                       "reference": 0.0, "unit": "|u - u_exact|", "pass": err < 0.02})
    elif kind == "mixing_layer":
        from math import erf

        xc = solver.cell_centres()
        Y = solver.fields()["tracer"]
        cols = np.unique(np.round(xc[:, 0], 12))
        x0 = cols[np.argmin(np.abs(cols - 1.0))]
        sel = np.abs(xc[:, 0] - x0) < 1e-9
        y = xc[sel, 1]
        ye = np.array([0.5 * (1.0 + erf((yy - 0.5) / np.sqrt(4.0 * info["D"] * x0))) for yy in y])
        err = float(np.abs(Y[sel] - ye).max())
        checks.append({"name": "Species profile at x = 1 vs analytic erf solution", "value": err, "reference": 0.0,
                       "unit": "max |dY|", "pass": err < 0.02})
        fb = solver.be.to_numpy(solver.Fb)
        yb = solver.be.to_numpy(solver.Y["tracer"])[solver.fv.Pb_h] if hasattr(solver.fv, "Pb_h") else None
        if yb is not None:
            inflow = float(-(fb[fb < 0] * solver.be.to_numpy(solver.Yb["tracer"])[fb < 0]).sum())
            outflow = float((fb[fb > 0] * yb[fb > 0]).sum())
            checks.append({"name": "Species conservation (out / in)", "value": outflow / inflow, "reference": 1.0,
                           "unit": "ratio", "pass": abs(outflow / inflow - 1.0) < 1e-4})
    mb = solver.mass_balance()
    if mb:
        scale = max(max(abs(v) for v in mb.values()), 1e-30)
        net = abs(sum(mb.values()))
        checks.append({"name": "Global mass balance", "value": net / scale if scale > 1e-20 else net,
                       "unit": "relative", "pass": net <= 1e-6 * scale + 1e-12})
    return {"checks": checks}
