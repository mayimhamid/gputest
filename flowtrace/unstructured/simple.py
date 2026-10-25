r"""Pressure-based segregated solver (SIMPLE / SIMPLEC) on unstructured meshes.

This mirrors the algorithm behind Fluent's default pressure-based solver:

* collocated cell-centred finite volumes on arbitrary polyhedra;
* momentum: implicit first-order upwind + deferred-correction bounded
  second-order upwind, central diffusion with explicit non-orthogonal
  correction (over-relaxed decomposition), least-squares gradients;
* Rhie-Chow face-flux interpolation to suppress pressure checkerboarding;
* pressure-correction equation solved with AMG-preconditioned CG;
* SIMPLE (alpha_p = 0.3) or SIMPLEC (consistent d-coefficients, alpha_p = 1);
* Fluent-style scaled residuals (momentum/energy: sum|r| / sum|a_P phi_P|,
  continuity: sum|mass imbalance| / max of the first five iterations).

All per-iteration work is vectorised over faces/cells and runs on NumPy or
CuPy; sparse assembly writes directly into a fixed CSR pattern.
"""

from __future__ import annotations

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
        self.fv = fv = FVMesh(mesh, self.be, self.dtype)
        self.setup_seconds = time.perf_counter() - t0
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

    def _solve(self, A, b, x0, diag, tol=0.1, maxiter=30):
        inv = 1.0 / diag
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
        gp = fv.gradient(p, pb)
        if c.pressure_gradient == "green-gauss":
            # face-pressure force: the discrete adjoint of the face-flux divergence,
            # which keeps the saddle-point system consistent on triangles
            gp = fv.gradient_gg(p, pb, gp)
        gU = [fv.gradient(U[:, k], ub[:, k]) for k in range(dim)]
        self._gp, self._gU = gp, gU
        t1 = time.perf_counter()
        tm["gradients"] += t1 - t0

        # ---------------- momentum ----------------
        src = -gp * fv.vol[:, None]
        if self.energy and any(c.buoyancy[:dim]):
            bv = xp.asarray(np.asarray(c.buoyancy[:dim]), dtype=self.dtype)
            src = src + (fv.vol * (self.T - c.t_ref))[:, None] * bv[None, :]
        if any(c.body_force[:dim]):
            bf = xp.asarray(np.asarray(c.body_force[:dim], dtype=np.float64), dtype=self.dtype)
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
        self.p = p + c.relax_p * pp
        if self.singular_p:
            self.p = self.p - self.p.mean()
        ppb = (1.0 - self.m_pres) * pp[Pb]
        gpp = fv.gradient(pp, ppb)
        self.U = Ustar - dP[:, None] * gpp

        # ---------------- energy ----------------
        if self.energy and not self.external_energy:
            res["energy"] = self._energy()
        t5 = time.perf_counter()
        tm["correct+energy"] += t5 - t4

        # ---------------- turbulence ----------------
        if self.turb is not None:
            ubc = self._velocity_bvals(self.U, gU)
            gUc = [fv.gradient(self.U[:, k], ubc[:, k]) for k in range(dim)]
            res.update(self.turb.solve(gUc))
            tm["turbulence"] += time.perf_counter() - t5

        self.iteration += 1
        self.step_count = self.iteration
        self.last_linear = {"momentum_iters": its, "pressure_iters": pinfo.iterations,
                            "pressure_residual": pinfo.residual}
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
        if self.iteration > 5 and max(res.values()) < c.tol:
            self.converged = True
        return 1.0

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
            out[z.name] = float(Fb[self.fv.bzone == z.id].sum())
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
        return st

    def load_state(self, st: dict):
        xp = self.xp
        self.U = xp.asarray(st["U"], dtype=self.dtype)
        self.p = xp.asarray(st["p"], dtype=self.dtype)
        self.F = xp.asarray(st["F"], dtype=self.dtype)
        self.Fb = xp.asarray(st["Fb"], dtype=self.dtype)
        if self.energy and "T" in st:
            self.T = xp.asarray(st["T"], dtype=self.dtype)
        self.iteration = int(st.get("iteration", 0))
        self._gp = self._gU = self._gT = None
        if self.turb is not None:
            self.turb.load(st)
