"""Turbulence-model validation cases (RANS, unstructured SIMPLEC solver).

* Fully developed channel at Re_tau = 392 against the DNS of Moser, Kim &
  Mansour, Phys. Fluids 11 (1999) 943 (``chan395.means``, 16 points sampled
  from the published file; u_tau = 1, delta = 1).
* Zero-pressure-gradient flat plate, Re_L = 5e6 per unit length (the NASA TMR
  2-D flat-plate set-up, fully turbulent from the leading edge), skin friction
  against White's correlation Cf = 0.455 / ln^2(0.06 Re_x) (F. M. White,
  *Viscous Fluid Flow*, 1974).
* Backward-facing step of Driver & Seegmiller, AIAA J. 23 (1985) 163:
  expansion ratio 1.125, Re_H = 36 000, measured reattachment x_r/H = 6.26 +- 0.10.
"""

from __future__ import annotations

import time

import numpy as np

MKM_SRC = "Moser, Kim & Mansour, Phys. Fluids 11 (1999), Re_tau = 392.24"
RE_TAU = 392.24
DNS_YPLUS = np.array([1.0628, 1.8888, 4.9810, 9.5307, 15.521, 19.797, 29.858, 50.958, 68.804, 98.400,
                      150.92, 198.91, 251.08, 301.61, 349.01, 392.24])
DNS_UPLUS = np.array([1.0605, 1.8799, 4.8032, 8.1754, 10.836, 11.966, 13.479, 14.902, 15.614, 16.493,
                      17.664, 18.535, 19.256, 19.758, 20.046, 20.133])
WHITE_SRC = "White (1974) Cf = 0.455/ln^2(0.06 Re_x)"
DS_SRC = "Driver & Seegmiller, AIAA J. 23 (1985): x_r/H = 6.26 +- 0.10"
DS_XR = 6.26


def white_cf(re_x):
    return 0.455 / np.log(0.06 * np.asarray(re_x)) ** 2


def _run(s, log_every=0, label=""):
    t0 = time.perf_counter()
    while not s.finished:
        s.step()
        if log_every and s.iteration % log_every == 0:
            r = s.residuals()["scaled"]
            print(f"  [{label}] it {s.iteration}: " + " ".join(f"{k}={v:.1e}" for k, v in r.items()), flush=True)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
# channel
# --------------------------------------------------------------------------

def channel_mesh(model: str):
    from ..mesh.grading import geometric
    from ..mesh.unstructured import cartesian_mesh_2d, make_periodic

    if model == "k-omega-sst":  # wall-resolved: first cell centre at y+ ~ 0.25
        ys = geometric(50, 1.0, 1.3e-3)
    else:  # wall functions: first cell centre at y+ ~ 25
        ys = np.linspace(0.0, 1.0, 9)
    m = cartesian_mesh_2d(4, len(ys) - 1, xs=np.linspace(0.0, 0.2, 5), ys=ys,
                          names=("in", "out", "wall", "centre"), types=("wall", "wall", "wall", "symmetry"))
    return make_periodic(m, "in", "out")


def run_channel(model: str = "k-omega-sst", backend: str = "numpy", max_iter: int = 20000, verbose=False) -> dict:
    """Half channel (wall at y=0, symmetry at y=1), periodic in x, driven by dp/dx = -1."""
    from ..hardware.backend import get_backend
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    m = channel_mesh(model)
    init = {"k": 1.0, "epsilon": 0.5} if "epsilon" in model else None
    c = UnstructuredCase(name=f"channel Re_tau=392 {model}",
                         boundaries={"wall": BoundarySpec("wall"), "centre": BoundarySpec("symmetry")},
                         viscosity=1.0 / RE_TAU, body_force=[1.0, 0.0, 0.0], turbulence=model,
                         turb_reference_velocity=15.0, init_velocity=[15.0, 0.0, 0.0], turb_init=init,
                         relax_u=0.9, max_iter=max_iter, tol=1e-6, backend=backend)
    s = SimpleSolver(m, c, backend=get_backend(backend))
    secs = _run(s, 1000 if verbose else 0, "channel")
    f, xc = s.fields(), s.cell_centres()
    first = np.isclose(xc[:, 0], xc[:, 0].min())
    order = np.argsort(xc[first, 1])
    y, u = xc[first, 1][order], f["x-velocity"][first][order]
    tau = s.turb.wall_shear()[s.turb.wall_b]
    U = np.interp(DNS_YPLUS / RE_TAU, y, u)  # u_tau = 1 -> U+ = U
    err = 100.0 * (U - DNS_UPLUS) / DNS_UPLUS
    sel = DNS_YPLUS >= (1.0 if model == "k-omega-sst" else 30.0)
    return {"model": model, "iterations": s.iteration, "converged": s.converged,
            "stability_rollbacks": len(s.guard_events), "seconds": secs,
            "cells": m.n_cells, "tau_w": float(tau.mean()), "y_plus_first": float(s.turb.yplus[s.turb.wall_b].max()),
            "yplus": DNS_YPLUS.tolist(), "U_dns": DNS_UPLUS.tolist(), "U": U.tolist(), "err_pct": err.tolist(),
            "max_abs_err_pct": float(np.abs(err[sel]).max()), "centre_err_pct": float(err[-1]),
            "profile_y": y.tolist(), "profile_u": u.tolist(), "compared_from_yplus": float(DNS_YPLUS[sel][0])}


# --------------------------------------------------------------------------
# flat plate
# --------------------------------------------------------------------------

def flat_plate_mesh(model: str):
    from ..mesh.grading import geometric
    from ..mesh.unstructured import cartesian_mesh_2d, split_zone

    up = -geometric(12, 0.33333, 5e-3)[::-1]
    plate = geometric(64, 2.0, 2e-3)
    xs = np.concatenate([up[:-1], plate])
    ys = geometric(70, 1.0, 5e-6) if model == "k-omega-sst" else geometric(32, 1.0, 4e-4)
    m = cartesian_mesh_2d(0, 0, xs=xs, ys=ys, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "symmetry", "pressure-outlet"))
    return split_zone(m, "bottom", lambda fc: fc[:, 0] > 0.0, "plate", "wall")


def run_flat_plate(model: str = "k-omega-sst", backend: str = "numpy", max_iter: int = 20000, verbose=False) -> dict:
    from ..hardware.backend import get_backend
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    re = 5e6
    m = flat_plate_mesh(model)
    k_inf = 2.25e-7  # TMR free stream (M = 0.2): k = 9e-9 a^2, omega = 1e-6 rho a^2 / mu
    w_inf = 2.5e-5 * re
    free = dict(k=k_inf, omega=w_inf, epsilon=0.09 * k_inf * w_inf)
    bnd = {"inlet": BoundarySpec("inlet", velocity=[1.0, 0.0], **free),
           "outlet": BoundarySpec("outlet", **free), "top": BoundarySpec("outlet", **free),
           "bottom": BoundarySpec("symmetry"), "plate": BoundarySpec("wall")}
    c = UnstructuredCase(name=f"flat plate {model}", boundaries=bnd, viscosity=1.0 / re, turbulence=model,
                         init_velocity=[1.0, 0.0, 0.0], turb_init={"k": k_inf, "omega": w_inf, "epsilon": free["epsilon"]},
                         max_iter=max_iter, tol=1e-6, backend=backend)
    s = SimpleSolver(m, c, backend=get_backend(backend))
    secs = _run(s, 1000 if verbose else 0, "plate")
    tau = s.turb.wall_shear()
    h = s.be.to_numpy
    wall = s.turb.wall_b & (h(s.fv.bzone) == m.zone_by_name("plate").id)
    x = h(s.fv.xfb)[wall, 0]
    order = np.argsort(x)
    x, cf = x[order], 2.0 * tau[wall][order]
    rex = re * x
    ref = white_cf(rex)
    sel = (rex >= 1e6) & (rex <= 1e7)
    dev = 100.0 * (cf[sel] - ref[sel]) / ref[sel]
    return {"model": model, "iterations": s.iteration, "converged": s.converged,
            "stability_rollbacks": len(s.guard_events), "seconds": secs, "cells": m.n_cells,
            "x": x.tolist(), "cf": cf.tolist(), "cf_white": ref.tolist(), "re_x": rex.tolist(),
            "mean_dev_pct": float(dev.mean()), "max_abs_dev_pct": float(np.abs(dev).max()),
            "cf_at_x097": float(np.interp(0.97, x, cf)), "white_at_x097": float(white_cf(re * 0.97)),
            "y_plus_max": float(s.turb.yplus[wall].max())}


# --------------------------------------------------------------------------
# backward-facing step
# --------------------------------------------------------------------------

def bfs_mesh(model: str, scale: float = 1.0):
    from ..fluent.convert import masked_block_mesh
    from ..mesh.grading import geometric

    n = lambda k: max(4, int(round(k * scale)))  # noqa: E731
    first = 1e-3 if model == "k-omega-sst" else 2e-2
    xs_up = -geometric(n(40), 110.0, 0.02)[::-1]
    xs_dn = geometric(n(110), 30.0, 0.02)
    xs = np.concatenate([xs_up[:-1], xs_dn])
    ys_lo = geometric(n(20), 0.5, first)
    ys_lo = np.concatenate([ys_lo, 1.0 - ys_lo[::-1][1:]])  # clustered at the floor and at the step lip
    ys_hi = geometric(n(60), 8.0, first, start=1.0)
    ys = np.concatenate([ys_lo[:-1], ys_hi])
    xc = 0.5 * (xs[1:] + xs[:-1])
    yc = 0.5 * (ys[1:] + ys[:-1])
    solid = (xc[:, None] < 0.0) & (yc[None, :] < 1.0)
    return masked_block_mesh(xs, ys, solid, names=("inlet", "outlet", "floor", "top"),
                             types=("velocity-inlet", "pressure-outlet", "wall", "symmetry"),
                             obstacle=("step", "wall"), name=f"BFS {model}")


def run_bfs(model: str = "k-omega-sst", backend: str = "numpy", max_iter: int = 20000, scale: float = 1.0,
            verbose=False) -> dict:
    from ..hardware.backend import get_backend
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    re = 36000.0
    m = bfs_mesh(model, scale)
    free = dict(turb_intensity=0.001, turb_viscosity_ratio=1.0)
    bnd = {"inlet": BoundarySpec("inlet", velocity=[1.0, 0.0], **free), "outlet": BoundarySpec("outlet", **free),
           "floor": BoundarySpec("wall"), "step": BoundarySpec("wall"), "top": BoundarySpec("symmetry")}
    c = UnstructuredCase(name=f"BFS Driver-Seegmiller {model}", boundaries=bnd, viscosity=1.0 / re,
                         turbulence=model, turb_intensity=0.001, turb_viscosity_ratio=1.0,
                         init_velocity=[1.0, 0.0, 0.0], max_iter=max_iter, tol=1e-6, backend=backend)
    s = SimpleSolver(m, c, backend=get_backend(backend))
    secs = _run(s, 1000 if verbose else 0, "BFS")
    h = s.be.to_numpy
    floor = h(s.fv.bzone) == m.zone_by_name("floor").id
    x = h(s.fv.xfb)[floor, 0]
    U = h(s.U)[h(s.fv.Pb)[floor], 0]
    order = np.argsort(x)
    x, U = x[order], U[order]
    tau = s.turb.wall_shear()[floor][order] * np.sign(U)
    # reattachment: last sign change from backward (U < 0) to forward flow behind the step
    xr = None
    idx = np.nonzero((U[:-1] < 0) & (U[1:] >= 0) & (x[:-1] > 0.2))[0]
    if idx.size:
        i = idx[-1]
        xr = float(x[i] - U[i] * (x[i + 1] - x[i]) / (U[i + 1] - U[i]))
    return {"model": model, "iterations": s.iteration, "converged": s.converged,
            "stability_rollbacks": len(s.guard_events), "seconds": secs, "cells": m.n_cells,
            "x_r": xr, "x": x.tolist(), "cf": (2.0 * tau).tolist()}
