"""Built-in Fluent-style cases (mesh + case) for the unstructured solver.

Each preset returns ``(mesh, case, info)``; ``info`` carries the reference
data used by :func:`validate_unstructured`.
"""

from __future__ import annotations

import numpy as np

from ..mesh.unstructured import cartesian_mesh_2d, split_zone, stretch
from .case import BoundarySpec, UnstructuredCase


def turbulent_channel(re_tau: float = 395.0, ny: int = 80, yplus1: float = 0.5, nx: int = 2, **kw):
    """Fully developed plane channel (half-height h = 1, u_tau = 1), periodic in x,
    driven by dp/dx = -rho u_tau^2 / h. Compare with DNS (Moser, Kim & Mansour 1999)."""
    nu = 1.0 / re_tau
    ys = stretch(ny, 2.0, first=yplus1 * nu, both=True)
    mesh = cartesian_mesh_2d(0, 0, xs=np.linspace(0.0, 0.2, nx + 1), ys=ys,
                             names=("inflow", "outflow", "bottom-wall", "top-wall"),
                             types=("periodic", "periodic", "wall", "wall"))
    case = UnstructuredCase(
        name=f"Turbulent channel Re_tau={re_tau:g} (k-omega SST)", viscosity=nu, density=1.0,
        turbulence="k-omega-sst", periodic=[["inflow", "outflow"]], periodic_pressure_gradient=[-1.0, 0.0, 0.0],
        init_velocity=[17.0, 0.0, 0.0], ref_velocity=17.0, max_iter=4000, tol=1e-6, relax_u=0.8, **kw)
    # DNS bulk velocity (MKM 1999: Re_b = 2h U_b / nu ~ 13 750 at Re_tau = 392); Dean's correlation otherwise
    ub_dean = (2.0 / 0.073 * (2.0 * re_tau) ** 0.25) ** (1.0 / 1.75)
    return mesh, case, {"kind": "channel", "re_tau": re_tau, "ub_plus_ref": 17.54 if abs(re_tau - 395) < 10 else ub_dean,
                        "ub_plus_dean": ub_dean}


def flat_plate(re_l: float = 5.0e6, nx: int = 96, ny: int = 64, yplus1: float = 0.6, **kw):
    """Zero-pressure-gradient flat plate (NASA TMR 2DZP-like): plate from x = 0 to 2,
    symmetry ahead of it, U = 1, Re per unit length ``re_l``."""
    nu = 1.0 / re_l
    cf_est = 0.0027
    first = yplus1 * nu / np.sqrt(0.5 * cf_est)
    ys = stretch(ny, 1.0, first=first)
    xa = -stretch(nx // 6, 0.33, first=2e-3)[::-1]
    xb = stretch(nx - nx // 6, 2.0, first=2e-3)
    xs = np.concatenate([xa, xb[1:]])
    mesh = cartesian_mesh_2d(0, 0, xs=xs, ys=ys, names=("inlet", "outlet", "bottom", "top"),
                             types=("velocity-inlet", "pressure-outlet", "wall", "pressure-outlet"))
    mesh = split_zone(mesh, "bottom", lambda fc: np.where(fc[:, 0] < 0.0, "upstream", "plate"),
                      {"upstream": "symmetry", "plate": "wall"})
    case = UnstructuredCase(
        name=f"Flat plate Re_L={re_l:g} (k-omega SST)", viscosity=nu, density=1.0, turbulence="k-omega-sst",
        boundaries={"inlet": BoundarySpec("inlet", velocity=[1.0, 0.0], turb_intensity=0.00039,
                                          turb_viscosity_ratio=0.009),
                    "outlet": BoundarySpec("outlet"), "top": BoundarySpec("outlet"),
                    "upstream": BoundarySpec("symmetry"), "plate": BoundarySpec("wall")},
        init_velocity=[1.0, 0.0, 0.0], ref_velocity=1.0, ref_area=2.0, max_iter=6000, tol=1e-6,
        turb_intensity=0.00039, turb_viscosity_ratio=0.009, **kw)
    return mesh, case, {"kind": "flat_plate", "re_l": re_l}


def radiation_slab(tau: float = 1.0, nx: int = 60, t_hot: float = 1000.0, t_cold: float = 500.0, **kw):
    """Grey absorbing slab between two black walls in radiative equilibrium (no
    conduction or flow). Exact P-1 flux: q = sigma (T1^4 - T2^4) / (1 + 3 tau / 4)."""
    L = 1.0
    mesh = cartesian_mesh_2d(nx, 2, L, 0.1, names=("hot", "cold", "bottom", "top"),
                             types=("wall", "wall", "symmetry", "symmetry"))
    case = UnstructuredCase(
        name=f"Radiation slab tau={tau:g} (P-1)", viscosity=1.0, density=1.0, energy=True,
        conductivity=1e-9, cp=1.0, radiation="p1", absorption=tau / L, init_T=0.5 * (t_hot + t_cold),
        boundaries={"hot": BoundarySpec("wall", thermal="temperature", value=t_hot, emissivity=1.0),
                    "cold": BoundarySpec("wall", thermal="temperature", value=t_cold, emissivity=1.0),
                    "bottom": BoundarySpec("symmetry"), "top": BoundarySpec("symmetry")},
        max_iter=3000, tol=1e-8, **kw)
    sigma = 5.670374419e-8
    q = sigma * (t_hot ** 4 - t_cold ** 4) / (1.0 + 0.75 * tau)
    return mesh, case, {"kind": "radiation_slab", "q_exact": q, "height": 0.1}


def periodic_taylor_green(n: int = 32, re: float = 10.0, dt: float = 0.05, t_end: float = 1.0, **kw):
    """Decaying Taylor-Green vortices on a doubly periodic unstructured mesh (transient)."""
    L = 2.0 * np.pi
    mesh = cartesian_mesh_2d(n, n, L, L, names=("left", "right", "bottom", "top"), types=("periodic",) * 4)
    case = UnstructuredCase(
        name=f"Taylor-Green (transient, periodic) Re={re:g}", viscosity=1.0 / re,
        periodic=[["left", "right"], ["bottom", "top"]], unsteady=True, dt=dt, t_end=t_end,
        inner_iterations=30, inner_tol=1e-7, max_iter=10 ** 7, **kw)
    return mesh, case, {"kind": "taylor_green", "nu": 1.0 / re, "init": "taylor_green"}


DESCRIPTIONS = {
    "turbulent_channel": ("Turbulent channel (k-ω SST, periodic)",
                          "Fully developed channel at Re_tau, periodic in x and driven by a mean pressure gradient; "
                          "compared with DNS bulk velocity and the log law."),
    "flat_plate": ("Flat plate boundary layer (k-ω SST)",
                   "Zero-pressure-gradient plate at Re_L = 5e6 with y+ < 1; skin friction vs NASA TMR SST results."),
    "radiation_slab": ("Radiating slab (P-1)", "Grey absorbing medium between black walls in radiative equilibrium; "
                                               "wall flux vs the exact P-1 solution."),
    "periodic_taylor_green": ("Taylor-Green vortex (transient, periodic mesh)",
                              "Unsteady pressure-based solver with BDF2 on a doubly periodic unstructured mesh; "
                              "error vs the exact decaying solution."),
}

PRESETS = {
    "turbulent_channel": turbulent_channel,
    "flat_plate": flat_plate,
    "radiation_slab": radiation_slab,
    "periodic_taylor_green": periodic_taylor_green,
}


def apply_initial_condition(solver, info):
    """Analytic initial fields some presets need (e.g. Taylor-Green)."""
    if info.get("init") == "taylor_green":
        xp = solver.xp
        xc = solver.cell_centres()
        U0 = np.stack([np.sin(xc[:, 0]) * np.cos(xc[:, 1]), -np.cos(xc[:, 0]) * np.sin(xc[:, 1])], axis=1)
        solver.U = xp.asarray(U0, dtype=solver.dtype)
        solver.F = solver.case.density * xp.sum(solver.fv.face_interp_vec(solver.U) * solver.fv.S, axis=1)
        solver._old, solver._F_old = {}, []
        solver._store_old()


def make_unstructured_preset(name: str, **params):
    if name not in PRESETS:
        raise KeyError(f"unknown unstructured preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name](**params)
