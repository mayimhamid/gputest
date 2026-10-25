"""Full-multigrid style grid sequencing for steady problems.

Like Fluent's FMG initialisation: converge the same case on 1/4 and 1/2
resolution first (16x and 4x fewer cells, 2-4x larger time steps), then
interpolate velocity, pressure and temperature to the next grid and finish
there. The finest grid only has to remove the small discretisation
difference instead of developing the whole flow from rest.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .case import StructuredCase
from .solver import G


def fmg_factors(case: StructuredCase, min_cells: int = 24, max_levels: int = 2) -> List[int]:
    """Coarsening factors (coarsest first) usable for this case, e.g. [4, 2]."""
    if case.steady_tol is None:
        return []
    out = []
    f = 2
    while len(out) < max_levels:
        if case.nx % f or case.ny % f or min(case.nx, case.ny) // f < min_cells:
            break
        out.append(f)
        f *= 2
    return sorted(out, reverse=True)


def coarse_case(case: StructuredCase, factor: int) -> StructuredCase:
    d = case.to_dict()
    d["nx"] = case.nx // factor
    d["ny"] = case.ny // factor
    d["name"] = f"{case.name} [FMG 1/{factor}]"
    c = StructuredCase.from_dict(d)
    c.steady_tol = max(case.steady_tol or 1e-6, 1e-6) * 30
    c.probes = []
    return c


def _weights(x_f: np.ndarray, x_c: np.ndarray) -> np.ndarray:
    W = np.zeros((x_f.size, x_c.size))
    eye = np.eye(x_c.size)
    for k in range(x_c.size):
        W[:, k] = np.interp(x_f, x_c, eye[k])
    return W


def prolong(coarse, fine) -> None:
    """Interpolate the state of solver ``coarse`` into solver ``fine`` (in place)."""
    cs, fs = coarse.case, fine.case
    h = coarse.be.to_numpy

    def axes(nx, ny, lx, ly, kind):
        xf = np.arange(nx + 1) * lx / nx
        xc = (np.arange(nx) + 0.5) * lx / nx
        yf = np.arange(ny + 1) * ly / ny
        yc = (np.arange(ny) + 0.5) * ly / ny
        return {"u": (xf, yc), "v": (xc, yf), "c": (xc, yc)}[kind]

    def interp(arr, kind):
        xc_, yc_ = axes(cs.nx, cs.ny, cs.lx, cs.ly, kind)
        xf_, yf_ = axes(fs.nx, fs.ny, fs.lx, fs.ly, kind)
        return _weights(xf_, xc_) @ arr @ _weights(yf_, yc_).T

    xp, dt = fine.xp, fine.dtype
    U = h(coarse.U)[G:G + cs.nx + 1, G:G + cs.ny]
    V = h(coarse.V)[G:G + cs.nx, G:G + cs.ny + 1]
    fine.U[G:G + fs.nx + 1, G:G + fs.ny] = xp.asarray(interp(U, "u"), dtype=dt)
    fine.V[G:G + fs.nx, G:G + fs.ny + 1] = xp.asarray(interp(V, "v"), dtype=dt)
    fine.P[...] = xp.asarray(interp(h(coarse.P), "c"), dtype=dt)
    if fine.energy and coarse.energy:
        T = h(coarse.T)[G:G + cs.nx, G:G + cs.ny]
        fine.T[G:G + fs.nx, G:G + fs.ny] = xp.asarray(interp(T, "c"), dtype=dt)
    fine.t = coarse.t
    fine._fill_velocity()
    fine._apply_obstacles()
    fine._project(dt=1.0, update_pressure=False)
    fine.NU_old = fine.NV_old = fine.NT_old = None
    fine.dt_prev = None
    fine.converged = False
    fine._res_count = 0
    fine._res_norm = {}
