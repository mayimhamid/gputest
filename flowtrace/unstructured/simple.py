r"""Pressure-based segregated solver (SIMPLE / SIMPLEC) on unstructured meshes.

This mirrors the algorithm behind Fluent's default pressure-based solver:

* collocated cell-centred finite volumes on arbitrary polyhedra;
* momentum: implicit first-order upwind + deferred-correction bounded
  second-order upwind, central diffusion with explicit non-orthogonal
  correction (over-relaxed decomposition), least-squares gradients;
* Rhie-Chow face-flux interpolation to suppress pressure checkerboarding;
* pressure-correction equation solved with AMG-preconditioned CG;
* SIMPLE (alpha_p = 0.3) or SIMPLEC (consistent d-coefficients, alpha_p = 1);
* steady or transient (implicit Euler / BDF2 with inner iterations per time
  step and Choi's Rhie-Chow correction);
* energy with P-1 radiation, passive species transport;
* Fluent-style scaled residuals (momentum/energy: sum|r| / sum|a_P phi_P|,
  continuity: sum|mass imbalance| / max of the first five iterations).

All per-iteration work is vectorised over faces/cells and runs on NumPy or
CuPy; sparse assembly writes directly into a fixed CSR pattern.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict

import numpy as np

from ..hardware.backend import get_backend
from ..linalg.amg import make_amg
from ..linalg.krylov import bicgstab, pcg
from ..mesh.unstructured import UnstructuredMesh
from .case import FLUENT_TO_KIND, BoundarySpec, UnstructuredCase
from .fvmesh import FVMesh

WALL, INLET, OUTLET, PRESSURE, SYMMETRY = range(5)


def rotation_velocity(x: np.ndarray, rot: dict, dim: int) -> np.ndarray:
    """omega * axis x (x - origin) for points x (n, dim); rot = {"omega", "origin", "axis"}."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, dim))
    w = float(rot.get("omega", 0.0))
    origin = np.asarray((list(rot.get("origin", [0.0, 0.0, 0.0])) + [0.0, 0.0, 0.0])[:3], dtype=float)
    ax = np.asarray(rot.get("axis", [0.0, 0.0, 1.0]), dtype=float)
    ax = ax / np.linalg.norm(ax)
    r = np.zeros((x.shape[0], 3))
    r[:, :dim] = x - origin[:dim]
    return (w * np.cross(ax[None, :], r))[:, :dim]
def auto_relax_p(non_orthogonality_deg: float) -> float:
    """Initial SIMPLEC pressure under-relaxation (the stability guard lowers it if needed).

    SIMPLEC needs none on orthogonal meshes. On non-orthogonal ones the skewed part of the
    p' Laplacian is neglected and accumulates without it (measured: warped 96^2/128^2 meshes
    diverged at 1.0 and converge at 0.8, which costs 2-6 % iterations).

    The worst angle alone cannot set a safe value: warped triangles (43 deg) converge at 0.8,
    uniformly sheared quads (45 deg) diverge at 0.5 and converge at 0.33, because their
    skew does not alternate in sign from face to face.
    """
    return 1.0 if non_orthogonality_deg <= 5.0 else 0.8


_KIND_CODE = {"wall": WALL, "inlet": INLET, "outlet": OUTLET, "pressure": PRESSURE, "symmetry": SYMMETRY}
T_ADIABATIC, T_FIXED, T_FLUX = range(3)


class SimpleSolver:
    def __init__(self, mesh: UnstructuredMesh, case: UnstructuredCase, backend=None):
        self.case = case
        self.mesh = mesh
        self.be = backend if backend is not None else get_backend(case.backend)
        self.xp = xp = self.be.xp
        self.dtype = np.dtype(np.float32 if case.precision == "fp32" else np.float64)
        t0 = time.perf_counter()
        if case.periodic and not mesh.periodic:  # zone pairs given on the case: pair them on (a copy of) the mesh
            from ..mesh.unstructured import _shallow, make_periodic

            mesh = _shallow(mesh)
            for za, zb in case.periodic:
                mesh = make_periodic(mesh, za, zb)
            self.mesh = mesh
        self.fv = fv = FVMesh(mesh, self.be, self.dtype)
        self.dim = dim = mesh.dim
        self.nc = mesh.n_cells
        self.energy = case.energy
        self.timers: Dict[str, float] = defaultdict(float)
        self.iteration = 0
        self.step_count = 0
        self.converged = False
        self._res_max5: Dict[str, float] = {}
        self.history = []
        self._amg = None
        self._amg_age = 10 ** 9
        self.amg_rebuild = 10
        self.last_linear = {}
        self._gp = self._gU = self._gT = None  # lagged gradients for boundary reconstruction
        self.external_energy = False  # True when a conjugate (fluid+solid) solver owns the energy equation
        self.relax_p = float(case.relax_p) if case.relax_p is not None else auto_relax_p(fv.non_orthogonality_deg)
        self._build_boundaries()
        U0 = np.zeros((self.nc, dim))
        iv = list(case.init_velocity) + [0.0] * 3
        U0[:] = iv[:dim]
        self.U = xp.asarray(U0, dtype=self.dtype)
        self.p = xp.zeros(self.nc, dtype=self.dtype)
        self.T = xp.full(self.nc, case.init_T, dtype=self.dtype) if self.energy else None
        ub = self._velocity_bvals(self.U)
        self.F = case.density * xp.sum(fv.face_interp_vec(self.U) * fv.S, axis=1)
        self.Fb = self._boundary_flux_fixed(ub)
        self.mom_source = None  # (nc, dim) per unit volume
        self.energy_source = None  # (nc,) volumetric heat source per unit volume
        from .turbulence import make_turbulence

        self.turb = make_turbulence(self)
        self.Y = {name: xp.zeros(self.nc, dtype=self.dtype) for name in case.species}
        self._build_species_boundaries()
        self.rad = None
        if case.radiation == "p1":
            from .radiation import P1Model

            self.rad = P1Model(self)
        # transient bookkeeping
        self.time = 0.0
        self.time_step = 0
        self.inner = 0
        self._old: Dict[str, tuple] = {}
        self._F_old: list = []
        if case.unsteady:
            self._store_old()
        # a user-set relaxation is left alone; transient runs restart every time step from a larger
        # residual than the previous step's converged inner iterations, which a growth test would misread
        self.guard_enabled = case.relax_p is None and not case.unsteady
        self._reset_guard()
        self.setup_seconds = time.perf_counter() - t0

    def initialize(self, fields: dict) -> None:
        """Initialise from cell values with Fluent field names (e.g. an interpolated
        .ip file): x-/y-/z-velocity, pressure, temperature. Face mass fluxes are
        rebuilt from the velocity so the first iteration starts consistently."""
        xp, dim = self.xp, self.dim
        U = self.be.to_numpy(self.U).astype(np.float64)
        for k, name in enumerate(("x-velocity", "y-velocity", "z-velocity")[:dim]):
            if name in fields:
                U[:, k] = np.asarray(fields[name], dtype=np.float64)
        self.U = xp.asarray(U, dtype=self.dtype)
        if "pressure" in fields:
            p = np.asarray(fields["pressure"], dtype=np.float64)
            if self.singular_p:
                p = p - p.mean()
            self.p = xp.asarray(p, dtype=self.dtype)
        if self.energy and "temperature" in fields:
            self.T = xp.asarray(np.asarray(fields["temperature"], dtype=np.float64), dtype=self.dtype)
        ub = self._velocity_bvals(self.U)
        self.F = self.case.density * xp.sum(self.fv.face_interp_vec(self.U) * self.fv.S, axis=1)
        self.Fb = self._boundary_flux_fixed(ub)
        self._gp = self._gU = self._gT = None
        self._reset_guard()

    def set_sources(self, momentum=None, energy=None) -> None:
        """Volumetric sources: momentum (nc, dim) [force/volume], energy (nc,) [heat/volume].

        Used for manufactured-solution verification and spatially varying body
        forces; a uniform body force is ``case.body_force``.
        """
        xp = self.xp
        if momentum is not None:
            m = np.asarray(momentum, dtype=np.float64).reshape(self.nc, self.dim)
            self.mom_source = xp.asarray(m * self.be.to_numpy(self.fv.vol)[:, None], dtype=self.dtype)
        else:
            self.mom_source = None
        if energy is not None:
            if not self.energy:
                raise ValueError("energy source given but the energy equation is off")
            e = np.asarray(energy, dtype=np.float64).reshape(self.nc)
            self.energy_source = xp.asarray(e * self.be.to_numpy(self.fv.vol) / self.case.cp, dtype=self.dtype)
        else:
            self.energy_source = None

    # ------------------------------------------------------------------
    def _build_boundaries(self):
        fv, case, xp, dim = self.fv, self.case, self.xp, self.dim
        n = fv.n_bf
        kind = np.zeros(n, dtype=np.int64)
        ub = np.zeros((n, dim))
        pb = np.zeros(n)
        tk = np.zeros(n, dtype=np.int64)
        tv = np.zeros(n)
        nb = self.be.to_numpy(fv.nb)
        xfb = self.be.to_numpy(fv.xfb).astype(np.float64)
        Pb_h = self.be.to_numpy(fv.Pb)
        self._build_mrf()
        self.zone_specs = {}
        for z in self.mesh.boundary_zones():
            spec = case.boundaries.get(z.name)
            if spec is None:
                spec = BoundarySpec(kind=FLUENT_TO_KIND.get(z.type, "wall"))
                if spec.kind == "inlet":
                    spec.normal_speed = 1.0
            self.zone_specs[z.name] = spec
            m = fv.bzone == z.id
            if spec.kind == "periodic" or z.type in ("periodic", "periodic-shadow"):
                if m.any():
                    raise ValueError(f"zone {z.name!r} is periodic but not paired (case.periodic / make_periodic)")
                continue
            kind[m] = _KIND_CODE[spec.kind]
            if spec.kind in ("wall", "inlet"):
                if spec.velocity is not None:
                    v = (list(spec.velocity) + [0.0, 0.0, 0.0])[:dim]
                    ub[m] = v
                elif spec.normal_speed is not None:
                    ub[m] = -spec.normal_speed * nb[m]  # area vectors point outwards
                elif spec.kind == "wall" and spec.rotation is not None:
                    ub[m] = rotation_velocity(xfb[m], spec.rotation, dim)
                elif spec.kind == "wall" and self.mrf_zones:
                    # as in Fluent, a wall bounding a moving reference frame moves with it
                    ub[m] = self._frame_at(xfb[m], Pb_h[m])
            pb[m] = spec.pressure
            if spec.kind == "wall":
                tk[m] = {"adiabatic": T_ADIABATIC, "temperature": T_FIXED, "flux": T_FLUX}[spec.thermal]
            elif spec.kind in ("inlet", "pressure"):
                tk[m] = T_FIXED
            tv[m] = spec.value
        self.bkind_h = kind
        d = lambda a: xp.asarray(a, dtype=self.dtype)  # noqa: E731
        self.bkind = xp.asarray(kind)
        self.ub_fixed = d(ub)
        self.pb_fixed = d(pb)
        self.tkind = xp.asarray(tk)
        self.tval = d(tv)
        self.m_wall = d(kind == WALL)
        self.m_inlet = d(kind == INLET)
        self.m_pres = d((kind == OUTLET) | (kind == PRESSURE))
        self.m_sym = d(kind == SYMMETRY)
        self.m_dirvel = d((kind == WALL) | (kind == INLET) | (kind == SYMMETRY))
        self.singular_p = not bool(((kind == OUTLET) | (kind == PRESSURE)).any())

    def _build_mrf(self):
        """Moving reference frames (absolute-velocity formulation, as Fluent's MRF).

        In an MRF cell zone rotating with Omega about ``origin``: convection uses the
        relative mass flux F - rho (Omega x r)_f . S_f, continuity the absolute flux,
        and the momentum equation gets the Coriolis-type source -rho Omega x U.
        """
        c, fv, xp, dim = self.case, self.fv, self.xp, self.dim
        self.mrf_zones = {}
        self._frame_zone = np.full(self.nc, -1, dtype=np.int64)
        self.mrf_flux = self.mrf_flux_b = None
        self.mrf_omega_cells = None
        if not c.mrf:
            return
        rots = []
        for i, (zname, rot) in enumerate(c.mrf.items()):
            z = self.mesh.zone_by_name(zname)
            if z is None or z.kind != "cell":
                raise ValueError(f"MRF zone {zname!r} is not a cell zone of this mesh")
            self._frame_zone[self.mesh.cell_zone == z.id] = i
            self.mrf_zones[zname] = rot
            rots.append(rot)
        self._rots = rots
        h = self.be.to_numpy
        P, N, Pb = h(fv.P), h(fv.N), h(fv.Pb)
        xf, xfb = h(fv.xf).astype(np.float64), h(fv.xfb).astype(np.float64)
        S, Sb = h(fv.S).astype(np.float64), h(fv.Sb).astype(np.float64)
        # frame of a face: the owner's, else the neighbour's (an MRF interface should be a
        # surface of revolution, where the frame velocity is tangential either way)
        zf = np.where(self._frame_zone[P] >= 0, self._frame_zone[P], self._frame_zone[N])
        Ff = np.zeros(P.size)
        for i, rot in enumerate(rots):
            m = zf == i
            Ff[m] = np.einsum("fd,fd->f", rotation_velocity(xf[m], rot, dim), S[m])
        zb = self._frame_zone[Pb]
        Fbb = np.zeros(Pb.size)
        for i, rot in enumerate(rots):
            m = zb == i
            Fbb[m] = np.einsum("fd,fd->f", rotation_velocity(xfb[m], rot, dim), Sb[m])
        rho = c.density
        self.mrf_flux = xp.asarray(rho * Ff, dtype=self.dtype)
        self.mrf_flux_b = xp.asarray(rho * Fbb, dtype=self.dtype)
        om = np.zeros((self.nc, 3))
        for i, rot in enumerate(rots):
            ax = np.asarray(rot.get("axis", [0.0, 0.0, 1.0]), dtype=float)
            om[self._frame_zone == i] = float(rot["omega"]) * ax / np.linalg.norm(ax)
        self.mrf_omega_cells = xp.asarray(om, dtype=self.dtype)

    def _frame_at(self, x, cells):
        """Frame velocity at points x whose adjacent cells are ``cells`` (0 outside MRF zones)."""
        out = np.zeros((x.shape[0], self.dim))
        z = self._frame_zone[cells]
        for i, rot in enumerate(getattr(self, "_rots", [])):
            m = z == i
            out[m] = rotation_velocity(x[m], rot, self.dim)
        return out

    def _conv(self):
        """Mass fluxes that convect momentum/scalars (relative to MRF frames)."""
        if self.mrf_flux is None:
            return self.F, self.Fb
        return self.F - self.mrf_flux, self.Fb - self.mrf_flux_b

    def _velocity_bvals(self, U, gU=None):
        """Boundary face velocities; zero-gradient values are reconstructed with the
        (lagged) cell gradient along the face, which keeps skewed boundary cells
        second-order accurate."""
        fv, xp = self.fv, self.xp
        UP = U[fv.Pb]
        if gU is not None:
            UP = UP + xp.stack([xp.sum(g[fv.Pb] * fv.db_t, axis=1) for g in gU], axis=1)
        un = xp.sum(UP * fv.nb, axis=1)
        sym = UP - un[:, None] * fv.nb
        fixed = (self.m_wall + self.m_inlet)[:, None]
        return fixed * self.ub_fixed + self.m_sym[:, None] * sym + self.m_pres[:, None] * UP

    def _pressure_bvals(self, p, gp=None):
        """Wall/inlet/symmetry face pressure: linear extrapolation from the owner
        cell with the (lagged) gradient; fixed value on pressure boundaries."""
        fv = self.fv
        pP = p[fv.Pb]
        if gp is not None:
            pP = pP + self.xp.sum(gp[fv.Pb] * fv.db, axis=1)
        return self.m_pres * self.pb_fixed + (1.0 - self.m_pres) * pP

    def _boundary_flux_fixed(self, ub):
        fv = self.fv
        flux = self.case.density * self.xp.sum(ub * fv.Sb, axis=1)
        return flux * (self.m_inlet + self.m_pres)

    # ------------------------------------------------------------------
    @staticmethod
    def _split_gamma(gamma):
        """Diffusivity as (interior-face, boundary-face) values; scalars broadcast."""
        if isinstance(gamma, tuple):
            return gamma
        return gamma, gamma

    def _coefficients(self, gamma, dirichlet):
        """Implicit part of div(F phi) - div(gamma grad phi): (diag, aPN, aNP, Db).

        ``gamma`` is a scalar or a tuple (interior-face values, boundary-face values).
        """
        fv, xp = self.fv, self.xp
        F, Fb = self._conv()
        gf, gb = self._split_gamma(gamma)
        D = gf * fv.alpha
        aPN = -(D + xp.maximum(-F, 0.0))  # coefficient of phi_N in P's row
        aNP = -(D + xp.maximum(F, 0.0))  # coefficient of phi_P in N's row
        Db = gb * fv.alpha_b * dirichlet
        diag = fv.scatter(D + xp.maximum(F, 0.0), D + xp.maximum(-F, 0.0), Db + xp.maximum(Fb, 0.0))
        return diag, aPN, aNP, Db

    def _rhs(self, gamma, phi, grad, phi_b, Db, extra=None, scheme=None):
        """Boundary sources + explicit non-orthogonal and second-order corrections."""
        fv, xp = self.fv, self.xp
        F, Fb = self._conv()
        gface, _ = self._split_gamma(gamma)
        rhs = fv.Cb_d @ ((Db + xp.maximum(-Fb, 0.0)) * phi_b)
        # non-orthogonal diffusion flux into P: gamma (grad phi)_f . T_f, outward from P
        gf = fv.face_interp_vec(grad)
        rhs = rhs + fv.Cdiff_d @ (gface * xp.sum(gf * fv.Tvec, axis=1))
        if (scheme or self.case.scheme) == "second-order":
            P, N = fv.P, fv.N
            phP, phN = phi[P], phi[N]
            hoP = phP + xp.sum(grad[P] * fv.dP_f, axis=1)
            hoN = phN + xp.sum(grad[N] * fv.dN_f, axis=1)
            up = F >= 0
            ho = xp.clip(xp.where(up, hoP, hoN), xp.minimum(phP, phN), xp.maximum(phP, phN))
            rhs = rhs - fv.Cdiff_d @ (F * (ho - xp.where(up, phP, phN)))
        if extra is not None:
            rhs = rhs + extra
        return rhs

    def _residual(self, A, diag, x, b):
        xp = self.xp
        r = float(xp.sum(xp.abs(b - A @ x)))
        s = float(xp.sum(xp.abs(diag * x)))
        return r / s if s > 0 else r

    def _relax_solve(self, diag, aPN, aNP, rhs, phi, alpha, tol=0.1, maxiter=40):
        dr = diag / alpha
        Ar = self.fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - alpha) / alpha) * diag * phi
        x, _ = self._solve(Ar, b, phi, dr, tol=tol, maxiter=maxiter)
        return x

    # ---------------- transient ----------------
    def _time_coeffs(self):
        """(c0, c1, c2) of d(rho phi)/dt ~ rho V/dt (c0 phi - c1 phi^n - c2 phi^(n-1)), or None if steady."""
        if not self.case.unsteady:
            return None
        if self.case.time_scheme == "bdf2" and self.time_step >= 1:
            return 1.5, 2.0, -0.5
        return 1.0, 1.0, 0.0

    def _time_terms(self, name, phi, diag, rhs):
        tc = self._time_coeffs()
        if tc is None or name is None:
            return diag, rhs
        c0, c1, c2 = tc
        rv = self.case.density * self.fv.vol / self.case.dt
        old, older = self._old.get(name, (phi, phi))
        return diag + c0 * rv, rhs + rv * (c1 * old + (c2 * older if c2 else 0.0))

    def _store_old(self):
        cur = {f"U{k}": self.U[:, k].copy() for k in range(self.dim)}
        if self.energy and self.T is not None:
            cur["T"] = self.T.copy()
        if self.turb is not None:
            cur["k"] = self.turb.k.copy()
            cur["w"] = self.turb.w.copy()
        for name, Y in getattr(self, "Y", {}).items():
            cur[f"Y:{name}"] = Y.copy()
        for name, v in cur.items():
            prev = self._old.get(name, (v, v))[0]
            self._old[name] = (v, prev)
        Fprev = self._F_old[0] if self._F_old else self.F.copy()
        self._F_old = [self.F.copy(), Fprev]

    def _advance_time(self):
        self.time += self.case.dt
        self.time_step += 1
        self.inner = 0
        self._store_old()

    @property
    def t(self):
        return self.time if self.case.unsteady else self.iteration

    @property
    def dt(self):
        return self.case.dt if self.case.unsteady else None

    # ---------------- species ----------------
    def _build_species_boundaries(self):
        """Per-species boundary values and Dirichlet masks (inlets, openings, walls with a value)."""
        xp, fv = self.xp, self.fv
        self.Yb, self.Ydir = {}, {}
        for name in self.case.species:
            val = np.zeros(fv.n_bf)
            fixed = np.zeros(fv.n_bf)
            for z in self.mesh.boundary_zones():
                m = fv.bzone == z.id
                spec = self.zone_specs.get(z.name)
                if spec is None or not m.any():
                    continue
                v = (spec.species or {}).get(name)
                if spec.kind in ("inlet", "pressure"):
                    val[m] = v or 0.0
                    fixed[m] = 1.0 if spec.kind == "inlet" else 0.0
                elif spec.kind == "outlet":
                    val[m] = v or 0.0  # backflow value
                elif spec.kind == "wall" and v is not None:
                    val[m] = v
                    fixed[m] = 1.0
            self.Yb[name] = xp.asarray(val, dtype=self.dtype)
            self.Ydir[name] = xp.asarray(fixed, dtype=self.dtype)

    def _species(self) -> dict:
        c, fv, xp = self.case, self.fv, self.xp
        res = {}
        rho = c.density
        if self.turb is not None:
            mut = self.turb.mut
            gam = (rho * c.species_diffusivity + fv.face_interp(mut) / c.schmidt_turbulent,
                   rho * c.species_diffusivity + mut[fv.Pb] / c.schmidt_turbulent * (1.0 - self.m_wall))
        else:
            gam = rho * c.species_diffusivity
        inflow = (self.m_inlet > 0) | ((self.m_pres > 0) & (self.Fb < 0))
        for name, Y in self.Y.items():
            fixed = (self.Ydir[name] > 0) | inflow
            Yb = xp.where(fixed, self.Yb[name], Y[fv.Pb])
            gY = fv.gradient(Y, Yb)
            diag, aPN, aNP, Db = self._coefficients(gam, self.Ydir[name])
            rhs = self._rhs(gam, Y, gY, Yb, Db)
            diag, rhs = self._time_terms(f"Y:{name}", Y, diag, rhs)
            A = fv.matrix(diag, aPN, aNP)
            res[name] = self._residual(A, diag, Y, rhs)
            self.Y[name] = self._relax_solve(diag, aPN, aNP, rhs, Y, 1.0, tol=1e-3, maxiter=60)
        return res

    def _wall_temperature(self):
        """Boundary-face temperatures: fixed values, else the adjacent cell (adiabatic/flux/outflow)."""
        TP = self.T[self.fv.Pb]
        return self.xp.where(self.tkind == T_FIXED, self.tval, TP)

    def _solve(self, A, b, x0, diag, tol=0.1, maxiter=30):
        inv = 1.0 / diag
        if self.fv.nat is not None:  # the whole Krylov loop in C++ (same algorithm and stopping rules)
            return self.fv.nat.bicgstab_jacobi(A, b, x0, inv, tol, maxiter, relative_to_initial=True, lib=self.fv._lib)
        if self.fv.cu is not None:
            return self.fv.cu.bicgstab_jacobi(A, b, x0, inv, tol, maxiter)
        x, info = bicgstab(lambda v: A @ v, b, self.xp, M=lambda v: inv * v, x0=x0, tol=tol, maxiter=maxiter,
                           relative_to_initial=True)
        return x, info

    # ------------------------------------------------------------------
    def step(self) -> float:
        if self.finished:
            return 0.0
        c, fv, xp = self.case, self.fv, self.xp
        rho, mu = c.density, c.viscosity
        dim = self.dim
        tm = self.timers
        t0 = time.perf_counter()
        U, p = self.U, self.p
        ub = self._velocity_bvals(U, self._gU)
        pb = self._pressure_bvals(p, self._gp)
        gp_ls = fv.gradient(p, pb)
        gp = gp_ls
        if c.pressure_gradient == "green-gauss":
            # face-pressure force: the discrete adjoint of the face-flux divergence,
            # which keeps the saddle-point system consistent on triangles
            gp = fv.gradient_gg(p, pb, gp_ls)
        gU = [fv.gradient(U[:, k], ub[:, k]) for k in range(dim)]
        self._gp, self._gU = gp, gU
        t1 = time.perf_counter()
        tm["gradients"] += t1 - t0

        # ---------------- momentum ----------------
        src = -gp * fv.vol[:, None]
        if self.energy and any(c.buoyancy[:dim]):
            bv = xp.asarray(np.asarray(c.buoyancy[:dim]), dtype=self.dtype)
            src = src + (fv.vol * (self.T - c.t_ref))[:, None] * bv[None, :]
        drive = np.asarray(c.body_force[:dim], dtype=np.float64) - np.asarray(
            (list(c.periodic_pressure_gradient) + [0.0] * 3)[:dim], dtype=np.float64)  # mean -dp/dx drives periodic flow
        if np.any(drive):
            bf = xp.asarray(drive, dtype=self.dtype)
            src = src + fv.vol[:, None] * bf[None, :]
        if self.mom_source is not None:
            src = src + self.mom_source
        if self.mrf_omega_cells is not None:  # -rho Omega x U (absolute-velocity MRF)
            om = self.mrf_omega_cells
            if dim == 2:
                cr = xp.stack([-om[:, 2] * U[:, 1], om[:, 2] * U[:, 0]], axis=1)
            else:
                cr = xp.stack([om[:, 1] * U[:, 2] - om[:, 2] * U[:, 1], om[:, 2] * U[:, 0] - om[:, 0] * U[:, 2],
                               om[:, 0] * U[:, 1] - om[:, 1] * U[:, 0]], axis=1)
            src = src - rho * fv.vol[:, None] * cr
        gm = mu
        if self.turb is not None:
            gm = self.turb.momentum_viscosity(U)  # mu + mu_t on faces, wall-function mu_w on walls
            src = src + self.turb.transpose_stress(gU)
        res = {}
        Ustar = xp.empty_like(U)
        au = c.relax_u
        diag, aPN, aNP, Db = self._coefficients(gm, self.m_dirvel)
        rhs_all = [self._rhs(gm, U[:, k], gU[k], ub[:, k], Db, src[:, k]) for k in range(dim)]
        tcoef = self._time_coeffs()
        if tcoef is not None:
            rv = rho * fv.vol / c.dt
            diag = diag + tcoef[0] * rv
            for k in range(dim):
                old, older = self._old[f"U{k}"]
                rhs_all[k] = rhs_all[k] + rv * (tcoef[1] * old + tcoef[2] * older)
        A = fv.matrix(diag, aPN, aNP)
        # Fluent-style scaled momentum residuals; the denominators are floored so
        # that a velocity component that is identically ~0 (2-D flow in a 3-D
        # mesh) does not divide round-off by round-off.
        num = [float(xp.sum(xp.abs(rhs_all[k] - A @ U[:, k]))) for k in range(dim)]
        den = [float(xp.sum(xp.abs(diag * U[:, k]))) for k in range(dim)]
        floor = 1e-3 * max(max(den), 1e-300)
        for k, name in enumerate(("x-velocity", "y-velocity", "z-velocity")[:dim]):
            res[name] = num[k] / max(den[k], floor)
        diag_r = diag / au
        Ar = fv.matrix(diag_r, aPN, aNP)
        its = []
        for k in range(dim):
            b = rhs_all[k] + ((1.0 - au) / au) * diag * U[:, k]
            x, info = self._solve(Ar, b, U[:, k], diag_r)
            Ustar[:, k] = x
            its.append(info.iterations)
        t2 = time.perf_counter()
        tm["momentum"] += t2 - t1

        # ---------------- Rhie-Chow fluxes ----------------
        if c.algorithm == "SIMPLEC":
            sum_nb = fv.scatter(-aPN, -aNP)
            denom = diag_r - sum_nb
            denom = xp.where(denom > 1e-12 * diag_r, denom, diag_r)
        else:
            denom = diag_r
        dP = fv.vol / denom
        dF = fv.face_interp(dP)
        gpf = fv.face_interp_vec(gp)
        P, N = fv.P, fv.N
        # Rhie-Chow: compact minus interpolated pressure gradient, both projected on
        # the orthogonal part alpha*d of the area vector. (Projecting the
        # interpolated gradient on S instead leaves a spurious grad(p).T mass flux
        # on non-orthogonal faces, which MMS shows as a non-converging pressure.)
        Fst = rho * (xp.sum(fv.face_interp_vec(Ustar) * fv.S, axis=1)
                     - dF * fv.alpha * (p[N] - p[P] - xp.sum(gpf * fv.d, axis=1)))
        if tcoef is not None:
            # Choi's correction: keeps the converged face flux independent of the time step
            r = rho * dF / c.dt
            Un = xp.stack([self._old[f"U{k}"][0] for k in range(dim)], axis=1)
            corr = tcoef[1] * (self._F_old[0] - rho * xp.sum(fv.face_interp_vec(Un) * fv.S, axis=1))
            if tcoef[2]:
                Unm = xp.stack([self._old[f"U{k}"][1] for k in range(dim)], axis=1)
                corr = corr + tcoef[2] * (self._F_old[1] - rho * xp.sum(fv.face_interp_vec(Unm) * fv.S, axis=1))
            Fst = Fst + r * corr
        Pb = fv.Pb
        Fb_in = rho * xp.sum(self.ub_fixed * fv.Sb, axis=1) * self.m_inlet
        Fb_pr = rho * (xp.sum(Ustar[Pb] * fv.Sb, axis=1)
                       - dP[Pb] * fv.alpha_b * (self.pb_fixed - p[Pb] - xp.sum(gp[Pb] * fv.db, axis=1)))
        Fbst = Fb_in + Fb_pr * self.m_pres

        # ---------------- pressure correction ----------------
        cf = rho * dF * fv.alpha
        cb = rho * dP[Pb] * fv.alpha_b * self.m_pres
        diag_p = fv.scatter(cf, cf, cb)
        Ap = fv.matrix(diag_p, -cf, -cf)
        mass = fv.net_outflow(Fst, Fbst)
        res["continuity"] = float(xp.sum(xp.abs(mass)))
        t3 = time.perf_counter()
        tm["assembly"] += t3 - t2
        if self._amg is None or self._amg_age >= self.amg_rebuild:
            Ah = Ap.get() if self.be.is_gpu else Ap
            self._amg = make_amg(Ah, self.be, singular=self.singular_p)
            self._amg_age = 0
        self._amg_age += 1
        if self.singular_p:
            def project(v):
                return v - v.mean()
        else:
            project = None
        pp, pinfo = pcg(lambda v: Ap @ v, -mass, xp, M=self._amg, tol=c.pressure_tol, maxiter=200, project=project)
        t4 = time.perf_counter()
        tm["pressure"] += t4 - t3

        # ---------------- corrections ----------------
        self.F = Fst + cf * (pp[P] - pp[N])
        self.Fb = Fbst + cb * pp[Pb]
        self.p = p + self.relax_p * pp
        if self.singular_p:
            self.p = self.p - self.p.mean()
        ppb = (1.0 - self.m_pres) * pp[Pb]
        gpp = fv.gradient(pp, ppb)
        self.U = Ustar - dP[:, None] * gpp

        # ---------------- energy ----------------
        if self.energy and not self.external_energy:
            res["energy"] = self._energy()
            if self.rad is not None:
                res["p1"] = self.rad.solve(self.T, self._wall_temperature())
        t5 = time.perf_counter()
        tm["correct+energy"] += t5 - t4

        # ---------------- turbulence ----------------
        if self.turb is not None:
            ubc = self._velocity_bvals(self.U, gU)
            gUc = [fv.gradient(self.U[:, k], ubc[:, k]) for k in range(dim)]
            res.update(self.turb.solve(gUc))
            tm["turbulence"] += time.perf_counter() - t5
        if self.Y:
            t6 = time.perf_counter()
            res.update(self._species())
            tm["species"] += time.perf_counter() - t6

        self.iteration += 1
        self.step_count = self.iteration
        self.last_linear = {"momentum_iters": its, "pressure_iters": pinfo.iterations,
                            "pressure_residual": pinfo.residual}
        prev = getattr(self, "_last_res", {}), getattr(self, "_last_raw", {})
        self._last_raw = dict(res)  # continuity: un-normalised sum |mass imbalance|
        if self.iteration <= 5:
            self._res_max5["continuity"] = max(self._res_max5.get("continuity", 0.0), res["continuity"], 1e-300)
        # Fluent normalises continuity by its largest value in the first five
        # iterations. Flows that start without any imbalance (periodic, warm
        # starts) would be normalised by round-off, so floor the denominator at a
        # small fraction of the total face mass flux.
        flux_scale = float(xp.sum(xp.abs(self.F))) + float(xp.sum(xp.abs(self.Fb)))
        res["continuity"] /= max(self._res_max5["continuity"], 1e-3 * flux_scale)
        self._last_res = res
        if self._stability_guard(self._last_raw["continuity"], flux_scale):
            self._last_res, self._last_raw = prev  # report the restored state, not the diverged one
            return 1.0
        if c.unsteady:
            self.inner += 1
            if self.inner >= c.inner_iterations or max(res.values()) < c.inner_tol:
                self._advance_time()
        elif self.iteration > 5 and max(res.values()) < c.tol:
            self.converged = True
        return 1.0

    # divergence guard: roll back and lower the pressure under-relaxation
    GUARD_GROWTH = 20.0  # continuity above GUARD_GROWTH x its running minimum = diverging
    GUARD_ROUNDOFF = 1e-6  # ... where that minimum is floored at this fraction of the total face mass flux
    GUARD_EVERY = 20  # snapshot interval (iterations), taken only while the residual is healthy
    GUARD_FACTOR = 0.6
    GUARD_MIN_RELAX = 0.1

    def _reset_guard(self):
        self.guard_events = []
        self._guard_min = float("inf")
        self._guard_snap = (self.state_dict(), float("inf")) if self.guard_enabled else None

    def _stability_guard(self, raw_continuity: float, flux_scale: float) -> bool:
        """Detect a diverging pressure-velocity coupling and recover from it.

        When the raw continuity residual becomes non-finite or grows ``GUARD_GROWTH``
        times above its minimum so far (floored at ``GUARD_ROUNDOFF`` x the face mass
        flux, so that round-off noise in periodic or converged flows is not mistaken
        for growth), restore the fields of the last healthy snapshot and multiply the
        pressure under-relaxation by ``GUARD_FACTOR``. The iteration counter is not
        rewound: discarded iterations count as work, and ``max_iter`` bounds the total.
        Returns True on a rollback. Only active when ``case.relax_p`` is None.
        """
        if not self.guard_enabled or self.external_energy:
            # a conjugate solver owns the solid temperatures, which a rollback here would not restore
            return False
        r = raw_continuity
        ref = max(self._guard_min, self.GUARD_ROUNDOFF * flux_scale)
        bad = not math.isfinite(r) or (self.iteration > 10 and r > self.GUARD_GROWTH * ref)
        if bad and self._guard_snap is not None and self.relax_p * self.GUARD_FACTOR >= self.GUARD_MIN_RELAX:
            snap, snap_r = self._guard_snap
            at = self.iteration
            self.load_state(snap)
            self.iteration = self.step_count = at
            self.relax_p *= self.GUARD_FACTOR
            self._amg_age = 10 ** 9
            self._guard_min = snap_r
            self.guard_events.append({"detected_at": at, "restored_to": int(snap["iteration"]),
                                      "relax_p": round(self.relax_p, 4), "continuity": r})
            return True
        if math.isfinite(r) and self.iteration > 5:  # past the start-up transient
            self._guard_min = min(self._guard_min, r)
            if self.iteration % self.GUARD_EVERY == 0 and r <= 2.0 * self._guard_min:
                self._guard_snap = (self.state_dict(), r)
        return False

    def _energy(self) -> float:
        c, fv, xp = self.case, self.fv, self.xp
        gam = c.conductivity / c.cp
        if self.turb is not None:
            gam = self.turb.energy_diffusivity(gam)  # k/cp + mu_t/Pr_t
        T = self.T
        TP = T[fv.Pb]
        fixed = self.tkind == T_FIXED
        flux = self.tkind == T_FLUX
        out = self.bkind == OUTLET
        if self._gT is not None:
            TP = TP + xp.sum(self._gT[fv.Pb] * fv.db_t, axis=1)
        Tb = xp.where(fixed, self.tval, TP)
        Tb = xp.where(out & (self.Fb < 0), self.tval, Tb)  # backflow enters at the outlet's T
        gT = fv.gradient(T, Tb)
        self._gT = gT
        q_in = fv.Cb_d @ (flux * self.tval * fv.Sb_mag / c.cp)
        if self.energy_source is not None:
            q_in = q_in + self.energy_source
        diag, aPN, aNP, Db = self._coefficients(gam, fixed.astype(self.dtype))
        rhs = self._rhs(gam, T, gT, Tb, Db, q_in)
        if self.rad is not None:
            dd, rr = self.rad.energy_terms(T)
            diag = diag + dd
            rhs = rhs + rr
        diag, rhs = self._time_terms("T", T, diag, rhs)
        A = fv.matrix(diag, aPN, aNP)
        r = self._residual(A, diag, T, rhs)
        aT = c.relax_T
        dr = diag / aT
        Ar = fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - aT) / aT) * diag * T
        self.T, _ = self._solve(Ar, b, T, dr, tol=1e-3, maxiter=60)
        return r

    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        if self.case.unsteady:
            return self.time >= self.case.t_end - 1e-9 * self.case.dt or self.iteration >= self.case.max_iter
        return self.converged or self.iteration >= self.case.max_iter

    def residuals(self) -> dict:
        """scaled: Fluent-style (continuity normalised by the largest of the first
        five iterations); raw: continuity as the absolute sum |mass imbalance|."""
        r = dict(getattr(self, "_last_res", {}))
        raw = dict(getattr(self, "_last_raw", r))
        return {"raw": raw, "scaled": r, "normalized": r}

    def describe(self) -> dict:
        amg = self._amg
        return {
            "solver": f"pressure-based {self.case.algorithm}",
            "turbulence": self.case.turbulence,
            "backend": self.be.name, "device": self.be.device, "precision": str(self.dtype),
            "cells": self.nc, "faces": self.mesh.n_faces, "dim": self.dim,
            "convection": self.case.scheme,
            "relax_p": self.relax_p,
            "stability_rollbacks": list(self.guard_events),
            "pressure_solver": "AMG-PCG (" + ("pyamg" if amg is not None and not hasattr(amg, "levels")
                                               else "FlowTrace SA-AMG") + ")",
            "max_non_orthogonality_deg": round(self.fv.non_orthogonality_deg, 2),
            "setup_seconds": round(self.setup_seconds, 3),
        }

    def fields(self, names=None) -> Dict[str, np.ndarray]:
        h = self.be.to_numpy
        U = h(self.U).astype(np.float64)
        out = {"x-velocity": U[:, 0], "y-velocity": U[:, 1]}
        if self.dim == 3:
            out["z-velocity"] = U[:, 2]
        out["velocity-magnitude"] = np.linalg.norm(U, axis=1)
        out["pressure"] = h(self.p).astype(np.float64)
        if self.energy:
            out["temperature"] = h(self.T).astype(np.float64)
        if self.turb is not None:
            out.update(self.turb.fields())
        if self.rad is not None:
            out["incident-radiation"] = h(self.rad.G).astype(np.float64)
        for name, Y in self.Y.items():
            out[name] = h(Y).astype(np.float64)
        if names:
            out = {k: v for k, v in out.items() if k in names}
        return out

    def cell_centres(self) -> np.ndarray:
        return self.fv.geometry.cell_centre

    def mass_balance(self) -> dict:
        """Net mass flow through every boundary zone (should sum to ~0)."""
        Fb = self.be.to_numpy(self.Fb)
        out = {}
        for z in self.mesh.boundary_zones():
            m = self.fv.bzone == z.id
            if m.any():
                out[z.name] = float(Fb[m].sum())
        return out

    def periodic_mass_flow(self) -> dict:
        """Mass flow entering through zone_a (= leaving through zone_b) for each periodic pair."""
        F = self.be.to_numpy(self.F)
        return {f"{p['zones'][0]}->{p['zones'][1]}": -float(F[p["faces"]].sum()) for p in self.fv.periodic}

    def _wall_viscosity(self):
        """Wall viscosity per boundary face, as the momentum equation uses it."""
        if self.turb is not None:
            return self._split_gamma(self.turb.momentum_viscosity(self.U))[1] * self.xp.ones(self.fv.n_bf, self.dtype)
        return self.xp.full(self.fv.n_bf, self.case.viscosity, dtype=self.dtype)

    def wall_report(self) -> dict:
        """Per wall zone: area, forces (pressure + viscous), Cd/Cl, y+, skin friction, heat flows.

        Cd/Cl use ``case.ref_velocity`` and ``case.ref_area``. The forces equal
        :meth:`wall_forces`; heat flows are conduction (``heat_flow_conv``) and P-1
        radiation (``heat_flow_rad``) into the fluid.
        """
        xp, fv, c = self.xp, self.fv, self.case
        h = self.be.to_numpy
        rho, mu = c.density, c.viscosity
        Pb, nb = fv.Pb, fv.nb
        rel = self.U[Pb] - self.ub_fixed
        rel_t = rel - xp.sum(rel * nb, axis=1)[:, None] * nb
        muw = self._wall_viscosity()
        tau_vec = muw[:, None] * rel_t / fv.yb[:, None]
        tau = xp.sqrt(xp.sum(tau_vec * tau_vec, axis=1))
        yplus = fv.yb * xp.sqrt(tau / rho) * rho / mu
        forces = self.wall_forces()
        qconv = None
        if self.energy:
            kc = c.conductivity
            qconv = h(kc * fv.alpha_b * (self._wall_temperature() - self.T[Pb]) * (self.tkind == T_FIXED)
                      + xp.where(self.tkind == T_FLUX, self.tval * fv.Sb_mag, 0.0))
        qrad = h(self.rad.wall_flux(self._wall_temperature())) if self.rad is not None else None
        hp = {"tau": h(tau), "yp": h(yplus), "A": h(fv.Sb_mag)}
        qdyn = 0.5 * rho * c.ref_velocity ** 2
        out = {}
        for z in self.mesh.boundary_zones():
            if z.name not in forces:
                continue
            m = fv.bzone == z.id
            A = hp["A"][m]
            f = forces[z.name]
            Ft = np.asarray(f["total"])
            rep = {"area": float(A.sum()), "force_pressure": f["pressure"], "force_viscous": f["viscous"],
                   "force_total": f["total"], "cd": float(Ft[0] / (qdyn * c.ref_area)),
                   "cl": float(Ft[1] / (qdyn * c.ref_area)),
                   "yplus_avg": float((hp["yp"][m] * A).sum() / A.sum()), "yplus_max": float(hp["yp"][m].max()),
                   "cf_avg": float((hp["tau"][m] * A).sum() / A.sum() / qdyn)}
            if qconv is not None:
                rep["heat_flow_conv"] = float(qconv[m].sum())
            if qrad is not None:
                rep["heat_flow_rad"] = float(qrad[m].sum())
            out[z.name] = rep
        return out

    def wall_profile(self, zone: str) -> dict:
        """Face-wise distributions on one wall zone (x, y, Cf, Cp, y+), sorted by x, for plotting."""
        xp, fv, c = self.xp, self.fv, self.case
        h = self.be.to_numpy
        z = self.mesh.zone_by_name(zone)
        m = fv.bzone == z.id
        rho, mu = c.density, c.viscosity
        Pb, nb = fv.Pb, fv.nb
        rel = self.U[Pb] - self.ub_fixed
        rel_t = rel - xp.sum(rel * nb, axis=1)[:, None] * nb
        tau = self._wall_viscosity() * xp.sqrt(xp.sum(rel_t * rel_t, axis=1)) / fv.yb
        qdyn = 0.5 * rho * c.ref_velocity ** 2
        xfb = h(fv.xfb)[m]
        order = np.argsort(xfb[:, 0])
        return {"x": xfb[order, 0], "y": xfb[order, 1], "cf": h(tau)[m][order] / qdyn,
                "cp": h(self._pressure_bvals(self.p, self._gp))[m][order] / qdyn,
                "yplus": h(fv.yb * xp.sqrt(tau / rho) * rho / mu)[m][order]}

    def wall_forces(self) -> dict:
        """Force exerted by the fluid on every wall zone (Fluent: Report > Forces), per unit depth in 2-D.

        Built from the discrete terms of the momentum equation, so it balances the
        solution exactly: the face pressure ``p_f S_f`` (the reconstructed boundary
        pressure) plus the wall shear ``mu_w alpha_b (u_P - u_w)_t``, where ``mu_w`` is
        the wall viscosity the momentum equation uses (laminar ``mu``, or the
        wall-function / SST value with turbulence). Returns
        ``{zone: {"pressure": [..], "viscous": [..], "total": [..]}}``.
        """
        h = self.be.to_numpy
        fv = self.fv
        pb = h(self._pressure_bvals(self.p, self._gp)).astype(np.float64)
        if self.turb is not None:
            mu_b = h(self._split_gamma(self.turb.momentum_viscosity(self.U))[1]).astype(np.float64)
            mu_b = np.broadcast_to(mu_b, pb.shape)
        else:
            mu_b = np.full(pb.shape, float(self.case.viscosity))
        Sb, nb = h(fv.Sb).astype(np.float64), h(fv.nb).astype(np.float64)
        alpha_b = h(fv.alpha_b).astype(np.float64)
        du = h(self.U).astype(np.float64)[h(fv.Pb)] - h(self.ub_fixed).astype(np.float64)
        du_t = du - np.einsum("fd,fd->f", du, nb)[:, None] * nb
        f_p = pb[:, None] * Sb  # S points out of the fluid, i.e. into the wall
        f_v = (mu_b * alpha_b)[:, None] * du_t
        out = {}
        for z in self.mesh.boundary_zones():
            m = (fv.bzone == z.id) & (self.bkind_h == WALL)
            if not m.any():
                continue
            fp, fvis = f_p[m].sum(axis=0), f_v[m].sum(axis=0)
            out[z.name] = {"pressure": fp.tolist(), "viscous": fvis.tolist(), "total": (fp + fvis).tolist()}
        return out

    def heat_flow(self) -> dict:
        """Heat flow into the domain through every boundary zone (conduction + enthalpy flux).

        At steady state (no volumetric sources) the values sum to ~0; Fluent's
        *Report > Fluxes > Total Heat Transfer Rate* reports the same quantity
        with the opposite sign convention for outflow.
        """
        if not self.energy:
            return {}
        h = self.be.to_numpy
        c, fv = self.case, self.fv
        T = h(self.T).astype(np.float64)
        Pb = h(fv.Pb)
        TP = T[Pb]
        tk, tv = h(self.tkind), h(self.tval).astype(np.float64)
        kind = self.bkind_h
        Fb = h(self.Fb).astype(np.float64)
        alpha_b = h(fv.alpha_b).astype(np.float64)
        Sb = h(fv.Sb_mag).astype(np.float64)
        Tb = np.where(tk == T_FIXED, tv, TP)
        Tb = np.where((kind == OUTLET) & (Fb < 0), tv, Tb)
        cond = c.conductivity * alpha_b * (Tb - TP) * (tk == T_FIXED)
        cond = cond + np.where(tk == T_FLUX, tv * Sb, 0.0)
        conv = -c.cp * Fb * Tb  # Fb > 0 is outflow
        q = cond + conv
        out = {}
        for z in self.mesh.boundary_zones():
            out[z.name] = float(q[fv.bzone == z.id].sum())
        return out

    def state_dict(self) -> dict:
        h = self.be.to_numpy
        st = {"U": h(self.U), "p": h(self.p), "F": h(self.F), "Fb": h(self.Fb), "iteration": self.iteration}
        if self.energy:
            st["T"] = h(self.T)
        if self.turb is not None:
            st.update(self.turb.state())
        st["time"], st["time_step"] = self.time, self.time_step
        st["res_continuity_norm"] = self._res_max5.get("continuity", 0.0)
        if self.rad is not None:
            st["G"] = h(self.rad.G)
        for name, Y in self.Y.items():
            st[f"Y:{name}"] = h(Y)
        return st

    def load_state(self, st: dict):
        xp = self.xp
        self.U = xp.asarray(st["U"], dtype=self.dtype)
        self.p = xp.asarray(st["p"], dtype=self.dtype)
        self.F = xp.asarray(st["F"], dtype=self.dtype)
        self.Fb = xp.asarray(st["Fb"], dtype=self.dtype)
        if self.energy and "T" in st:
            self.T = xp.asarray(st["T"], dtype=self.dtype)
        if self.rad is not None and "G" in st:
            self.rad.G = xp.asarray(st["G"], dtype=self.dtype)
        for name in self.Y:
            if f"Y:{name}" in st:
                self.Y[name] = xp.asarray(st[f"Y:{name}"], dtype=self.dtype)
        self.iteration = int(st.get("iteration", 0))
        self._gp = self._gU = self._gT = None
        if self.turb is not None:
            self.turb.load(st)
        if float(st.get("res_continuity_norm", 0.0)) > 0:
            self._res_max5["continuity"] = float(st["res_continuity_norm"])
        self.time = float(st.get("time", 0.0))
        self.time_step = int(st.get("time_step", 0))
        if self.case.unsteady:
            self._old, self._F_old = {}, []
            self._store_old()
