r"""Method of Manufactured Solutions (MMS): code verification by observed order.

A smooth, divergence-free field that satisfies every boundary condition of a
closed box is chosen, the Navier-Stokes (+ energy) residual of that field is
added as a source term, and the discrete solution is compared with the exact
one on a sequence of grids. The observed order ``p = log2(e_2h / e_h)`` must
match the formal order of the discretisation (2 for both solvers).

Manufactured solution on [0, 1]^2 (stream function psi = sin^2(pi x) sin^2(pi y)):

    u =  pi sin^2(pi x) sin(2 pi y)        (u = v = 0 on every wall)
    v = -pi sin(2 pi x) sin^2(pi y)        (div u = 0 exactly)
    p =  cos(pi x) cos(pi y)               (zero mean, dp/dn = 0 on the walls)
    T =  sin(pi x) sin(pi y)               (T = 0 on every wall)

Sources (rho = 1, cp = 1):
    f   = (u . grad) u + grad p - nu lap u
    f_T = u . grad T - kappa lap T
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

PI = math.pi


def exact(x, y) -> Dict[str, np.ndarray]:
    a, b = PI * np.asarray(x, dtype=float), PI * np.asarray(y, dtype=float)
    return {
        "u": PI * np.sin(a) ** 2 * np.sin(2 * b),
        "v": -PI * np.sin(2 * a) * np.sin(b) ** 2,
        "p": np.cos(a) * np.cos(b),
        "T": np.sin(a) * np.sin(b),
    }


def sources(x, y, nu: float, kappa: float) -> Dict[str, np.ndarray]:
    a, b = PI * np.asarray(x, dtype=float), PI * np.asarray(y, dtype=float)
    s, c = np.sin, np.cos
    u = PI * s(a) ** 2 * s(2 * b)
    v = -PI * s(2 * a) * s(b) ** 2
    ux = PI ** 2 * s(2 * a) * s(2 * b)
    uy = 2 * PI ** 2 * s(a) ** 2 * c(2 * b)
    lap_u = 2 * PI ** 3 * c(2 * a) * s(2 * b) - 4 * PI ** 3 * s(a) ** 2 * s(2 * b)
    vx = -2 * PI ** 2 * c(2 * a) * s(b) ** 2
    vy = -PI ** 2 * s(2 * a) * s(2 * b)
    lap_v = 4 * PI ** 3 * s(2 * a) * s(b) ** 2 - 2 * PI ** 3 * s(2 * a) * c(2 * b)
    px = -PI * s(a) * c(b)
    py = -PI * c(a) * s(b)
    T = s(a) * s(b)
    Tx = PI * c(a) * s(b)
    Ty = PI * s(a) * c(b)
    return {
        "u": u * ux + v * uy + px - nu * lap_u,
        "v": u * vx + v * vy + py - nu * lap_v,
        "T": u * Tx + v * Ty + 2 * PI ** 2 * kappa * T,
    }


@dataclass
class MMSResult:
    solver: str
    grids: List[int]
    h: List[float]
    errors: Dict[str, List[float]]  # quantity -> L2 error per grid
    errors_inf: Dict[str, List[float]]
    orders: Dict[str, List[float]] = field(default_factory=dict)  # observed order between successive grids
    iterations: List[int] = field(default_factory=list)
    seconds: List[float] = field(default_factory=list)

    def __post_init__(self):
        for k, e in self.errors.items():
            self.orders[k] = [math.log(e[i] / e[i + 1]) / math.log(self.h[i] / self.h[i + 1])
                              for i in range(len(e) - 1)]

    def to_dict(self) -> dict:
        return {"solver": self.solver, "grids": self.grids, "h": self.h, "errors_l2": self.errors,
                "errors_linf": self.errors_inf, "observed_order": self.orders,
                "iterations": self.iterations, "seconds": self.seconds}


def _l2(e: np.ndarray, w: np.ndarray = None) -> float:
    e = np.asarray(e, dtype=float).ravel()
    if w is None:
        return float(np.sqrt(np.mean(e ** 2)))
    w = np.asarray(w, dtype=float).ravel()
    return float(np.sqrt(np.sum(w * e ** 2) / np.sum(w)))


def run_mac(grids: Sequence[int] = (16, 32, 64), nu: float = 0.5, kappa: float = 0.5, energy: bool = True,
            scheme: str = "central", backend: str = "numpy", tol: float = 1e-10, max_steps: int = 200000) -> MMSResult:
    """Steady MMS on the staggered MAC solver (all walls, no-slip, T = 0)."""
    from ..hardware.backend import get_backend
    from ..structured.case import Side, StructuredCase
    from ..structured.solver import G, MACSolver

    be = get_backend(backend)
    hs, errs, errs_inf, its, secs = [], {"u": [], "v": [], "p": []}, {"u": [], "v": [], "p": []}, [], []
    if energy:
        errs["T"], errs_inf["T"] = [], []
    for n in grids:
        wall_T = dict(thermal="temperature", value=0.0) if energy else {}
        case = StructuredCase(
            name=f"MMS MAC n={n}", nx=n, ny=n, lx=1.0, ly=1.0,
            left=Side("wall", **wall_T), right=Side("wall", **wall_T),
            bottom=Side("wall", **wall_T), top=Side("wall", **wall_T),
            nu=nu, energy=energy, kappa=kappa, scheme=scheme, steady_tol=tol, max_steps=max_steps,
            u_ref=1.0, precision="fp64", backend=backend,
        )
        t0 = time.perf_counter()
        s = MACSolver(case, backend=be, dtype=np.float64)
        h = 1.0 / n
        xf, xc = np.arange(n + 1) * h, (np.arange(n) + 0.5) * h
        su = sources(xf[:, None], xc[None, :], nu, kappa)["u"]
        sv = sources(xc[:, None], xf[None, :], nu, kappa)["v"]
        sT = sources(xc[:, None], xc[None, :], nu, kappa)["T"] if energy else None
        s.set_sources(su, sv, sT)
        while not s.finished:
            s.step()
            if s.step_count % 10 == 0:
                s.residuals()
        secs.append(time.perf_counter() - t0)
        its.append(s.step_count)
        U = be.to_numpy(s.U)[G:G + n + 1, G:G + n]
        V = be.to_numpy(s.V)[G:G + n, G:G + n + 1]
        P = be.to_numpy(s.P)
        eu = U - exact(xf[:, None], xc[None, :])["u"]
        ev = V - exact(xc[:, None], xf[None, :])["v"]
        pe = exact(xc[:, None], xc[None, :])["p"]
        ep = (P - P.mean()) - (pe - pe.mean())
        hs.append(h)
        for k, e in (("u", eu[1:-1]), ("v", ev[:, 1:-1]), ("p", ep)):
            errs[k].append(_l2(e))
            errs_inf[k].append(float(np.abs(e).max()))
        if energy:
            eT = be.to_numpy(s.T)[G:G + n, G:G + n] - exact(xc[:, None], xc[None, :])["T"]
            errs["T"].append(_l2(eT))
            errs_inf["T"].append(float(np.abs(eT).max()))
    return MMSResult("MAC projection (structured)", list(grids), hs, errs, errs_inf, iterations=its, seconds=secs)


def run_simplec(grids: Sequence[int] = (12, 24, 48), mu: float = 0.5, k: float = 0.5, energy: bool = True,
                triangulate: bool = False, warp: float = 0.0, scheme: str = "second-order",
                backend: str = "numpy", tol: float = 1e-8, max_iter: int = 20000,
                pressure_gradient: str = "green-gauss") -> MMSResult:
    """Steady MMS on the unstructured SIMPLEC solver (quads or triangles).

    ``warp`` distorts the mesh smoothly (non-orthogonal, skewed cells) so the
    grid sequence remains a systematic refinement.
    """
    from ..hardware.backend import get_backend
    from ..mesh.unstructured import cartesian_mesh_2d
    from ..unstructured.case import BoundarySpec, UnstructuredCase
    from ..unstructured.simple import SimpleSolver

    be = get_backend(backend)
    hs, errs, errs_inf, its, secs = [], {"u": [], "v": [], "p": []}, {"u": [], "v": [], "p": []}, [], []
    if energy:
        errs["T"], errs_inf["T"] = [], []
    for n in grids:
        mesh = cartesian_mesh_2d(n, n, triangulate=triangulate, warp=warp)
        wall = BoundarySpec("wall", thermal="temperature", value=0.0) if energy else BoundarySpec("wall")
        case = UnstructuredCase(name=f"MMS SIMPLEC n={n}", boundaries={s: wall for s in ("left", "right", "bottom", "top")},
                                density=1.0, viscosity=mu, energy=energy, conductivity=k, cp=1.0,
                                scheme=scheme, max_iter=max_iter, tol=tol, backend=backend, relax_u=0.8,
                                pressure_gradient=pressure_gradient)
        t0 = time.perf_counter()
        s = SimpleSolver(mesh, case, backend=be)
        xc = s.cell_centres()
        src = sources(xc[:, 0], xc[:, 1], mu, k)
        s.set_sources(np.column_stack([src["u"], src["v"]]), src["T"] if energy else None)
        while not s.finished:
            s.step()
        secs.append(time.perf_counter() - t0)
        its.append(s.iteration)
        f = s.fields()
        ex = exact(xc[:, 0], xc[:, 1])
        vol = s.fv.geometry.cell_volume
        hs.append(float(np.sqrt(vol.mean())))
        pe = ex["p"] - np.sum(ex["p"] * vol) / vol.sum()
        ph = f["pressure"] - np.sum(f["pressure"] * vol) / vol.sum()
        for key, e in (("u", f["x-velocity"] - ex["u"]), ("v", f["y-velocity"] - ex["v"]), ("p", ph - pe)):
            errs[key].append(_l2(e, vol))
            errs_inf[key].append(float(np.abs(e).max()))
        if energy:
            e = f["temperature"] - ex["T"]
            errs["T"].append(_l2(e, vol))
            errs_inf["T"].append(float(np.abs(e).max()))
    label = "SIMPLEC (unstructured, " + ("warped " if warp else "") + ("triangles" if triangulate else "quads") + ")"
    return MMSResult(label, list(grids), hs, errs, errs_inf, iterations=its, seconds=secs)
