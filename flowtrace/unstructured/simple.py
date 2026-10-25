r"""Pressure-based segregated solver (SIMPLE / SIMPLEC) on unstructured meshes.

This mirrors the algorithm behind Fluent's default pressure-based solver:

* collocated cell-centred finite volumes on arbitrary polyhedra;
* momentum: implicit first-order upwind + deferred-correction bounded
  second-order upwind, central diffusion with explicit non-orthogonal
  correction (over-relaxed decomposition), least-squares gradients;
* Rhie-Chow face-flux interpolation to suppress pressure checkerboarding
  (with Choi's correction for transient runs);
* pressure-correction equation solved with AMG-preconditioned CG;
* SIMPLE (alpha_p = 0.3) or SIMPLEC (consistent d-coefficients, alpha_p = 1);
* steady or transient (implicit Euler / BDF2 with inner iterations per step);
* k-omega SST turbulence, energy with turbulent conductivity, P-1 radiation;
* translational periodic zones driven by a mean pressure gradient;
* Fluent-style scaled residuals (momentum/energy: sum|r| / sum|a_P phi_P|,
  continuity: sum|mass imbalance| / max of the first five iterations);
* wall reports: y+, wall shear, skin friction, pressure/viscous forces, heat flux.

All per-iteration work is vectorised over faces/cells and runs on NumPy or
CuPy; sparse assembly writes directly into a fixed CSR pattern.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from ..hardware.backend import get_backend
from ..linalg.amg import make_amg
from ..linalg.krylov import bicgstab, pcg
from ..mesh.unstructured import UnstructuredMesh
from .case import FLUENT_TO_KIND, BoundarySpec, UnstructuredCase
from .fvmesh import FVMesh

WALL, INLET, OUTLET, PRESSURE, SYMMETRY = range(5)
_KIND_CODE = {"wall": WALL, "inlet": INLET, "outlet": OUTLET, "pressure": PRESSURE, "symmetry": SYMMETRY}
T_ADIABATIC, T_FIXED, T_FLUX = range(3)


class SimpleSolver:
    WALL = WALL

    def __init__(self, mesh: UnstructuredMesh, case: UnstructuredCase, backend=None):
        self.case = case
        self.mesh = mesh
        self.be = backend if backend is not None else get_backend(case.backend)
        self.xp = xp = self.be.xp
        self.dtype = np.dtype(np.float32 if case.precision == "fp32" else np.float64)
        t0 = time.perf_counter()
        self.fv = fv = FVMesh(mesh, self.be, self.dtype, periodic=case.periodic)
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
        self.mu_b = xp.full(fv.n_bf, case.viscosity, dtype=self.dtype)
        self.turb = None
        if case.turbulent:
            from .turbulence import SSTModel

            self.turb = SSTModel(self)
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
        self._old = {}
        self._F_old = []
        if case.unsteady:
            self._store_old()
        self.setup_seconds = time.perf_counter() - t0

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
        self.zone_specs = {}
        periodic_zones = {z for pair in case.periodic for z in pair}
        for z in self.mesh.boundary_zones():
            spec = case.boundaries.get(z.name)
            if spec is None:
                spec = BoundarySpec(kind=FLUENT_TO_KIND.get(z.type, "wall"))
                if spec.kind == "inlet":
                    spec.normal_speed = 1.0
            if z.name in periodic_zones:
                spec = BoundarySpec(kind="periodic")
            self.zone_specs[z.name] = spec
            m = fv.bzone == z.id
            if spec.kind == "periodic":
                if m.any():
                    raise ValueError(f"zone {z.name!r} is 'periodic' but not paired in case.periodic")
                continue
            kind[m] = _KIND_CODE[spec.kind]
            if spec.kind in ("wall", "inlet"):
                if spec.velocity is not None:
                    v = (list(spec.velocity) + [0.0, 0.0, 0.0])[:dim]
                    ub[m] = v
                elif spec.normal_speed is not None:
                    ub[m] = -spec.normal_speed * nb[m]  # area vectors point outwards
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
            gam = rho * c.species_diffusivity + fv.face_interp(mut) / c.schmidt_turbulent
            gam_b = rho * c.species_diffusivity + mut[fv.Pb] / c.schmidt_turbulent * (1.0 - self.m_wall)
        else:
            gam, gam_b = rho * c.species_diffusivity, None
        inflow = (self.m_inlet > 0) | ((self.m_pres > 0) & (self.Fb < 0))
        for name, Y in self.Y.items():
            fixed = (self.Ydir[name] > 0) | inflow
            Yb = xp.where(fixed, self.Yb[name], Y[fv.Pb])
            gY = fv.gradient(Y, Yb)
            diag, aPN, aNP, Db = self._coefficients(gam, self.Ydir[name], gamma_b=gam_b)
            rhs = self._rhs(gam, Y, gY, Yb, Db)
            diag, rhs = self._time_terms(f"Y:{name}", Y, diag, rhs)
            A = fv.matrix(diag, aPN, aNP)
            res[name] = self._residual(A, diag, Y, rhs)
            self.Y[name] = self._relax_solve(diag, aPN, aNP, rhs, Y, 1.0, tol=1e-3, maxiter=60)
        return res

    def _velocity_bvals(self, U):
        fv, xp = self.fv, self.xp
        UP = U[fv.Pb]
        un = xp.sum(UP * fv.nb, axis=1)
        sym = UP - un[:, None] * fv.nb
        fixed = (self.m_wall + self.m_inlet)[:, None]
        return fixed * self.ub_fixed + self.m_sym[:, None] * sym + self.m_pres[:, None] * UP

    def _pressure_bvals(self, p):
        return self.m_pres * self.pb_fixed + (1.0 - self.m_pres) * p[self.fv.Pb]

    def _boundary_flux_fixed(self, ub):
        fv = self.fv
        flux = self.case.density * self.xp.sum(ub * fv.Sb, axis=1)
        return flux * (self.m_inlet + self.m_pres)

    # ------------------------------------------------------------------
    # discretisation helpers (shared with the turbulence / radiation models)
    # ------------------------------------------------------------------
    def _coefficients(self, gamma, dirichlet, gamma_b=None):
        """Implicit part of div(F phi) - div(gamma grad phi): (diag, aPN, aNP, Db).

        ``gamma`` is a scalar or an interior-face array; ``gamma_b`` the
        boundary-face diffusivity (defaults to ``gamma`` when it is a scalar).
        """
        fv, xp = self.fv, self.xp
        F, Fb = self.F, self.Fb
        D = gamma * fv.alpha
        aPN = -(D + xp.maximum(-F, 0.0))  # coefficient of phi_N in P's row
        aNP = -(D + xp.maximum(F, 0.0))  # coefficient of phi_P in N's row
        gb = gamma if gamma_b is None else gamma_b
        Db = gb * fv.alpha_b * dirichlet
        diag = fv.scatter(D + xp.maximum(F, 0.0), D + xp.maximum(-F, 0.0), Db + xp.maximum(Fb, 0.0))
        return diag, aPN, aNP, Db

    def _rhs(self, gamma, phi, grad, phi_b, Db, extra=None, second_order=None):
        """Boundary sources + explicit non-orthogonal and second-order corrections."""
        fv, xp = self.fv, self.xp
        F = self.F
        rhs = fv.Cb_d @ ((Db + xp.maximum(-self.Fb, 0.0)) * phi_b)
        # non-orthogonal diffusion flux into P: gamma (grad phi)_f . T_f, outward from P
        gf = fv.face_interp_vec(grad)
        rhs = rhs + fv.Cdiff_d @ (gamma * xp.sum(gf * fv.Tvec, axis=1))
        if second_order is None:
            second_order = self.case.scheme == "second-order"
        if second_order:
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

    def _residual(self, A, diag, x, b, mask=None):
        xp = self.xp
        rr = xp.abs(b - A @ x)
        ss = xp.abs(diag * x)
        if mask is not None:
            rr, ss = rr * mask, ss * mask
        r = float(xp.sum(rr))
        s = float(xp.sum(ss))
        return r / s if s > 0 else r

    def _solve(self, A, b, x0, diag, tol=0.1, maxiter=30):
        inv = 1.0 / diag
        x, info = bicgstab(lambda v: A @ v, b, self.xp, M=lambda v: inv * v, x0=x0, tol=tol, maxiter=maxiter,
                           relative_to_initial=True)
        return x, info

    def _relax_solve(self, diag, aPN, aNP, rhs, phi, alpha, tol=0.1, maxiter=40):
        dr = diag / alpha
        Ar = self.fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - alpha) / alpha) * diag * phi
        x, _ = self._solve(Ar, b, phi, dr, tol=tol, maxiter=maxiter)
        return x

    # ---------------- transient ----------------
    def _time_coeffs(self):
        if not self.case.unsteady:
            return None
        if self.case.time_scheme == "bdf2" and self.time_step >= 1:
            return 1.5, 2.0, -0.5
        return 1.0, 1.0, 0.0

    def _time_terms(self, name, phi, diag, rhs):
        tc = self._time_coeffs()
        if tc is None:
            return diag, rhs
        c0, c1, c2 = tc
        rv = self.case.density * self.fv.vol / self.case.dt
        old, older = self._old.get(name, (phi, phi))
        return diag + c0 * rv, rhs + rv * (c1 * old + (c2 * older if c2 else 0.0))

    def _store_old(self):
        xp = self.xp
        cur = {f"U{k}": self.U[:, k].copy() for k in range(self.dim)}
        if self.energy:
            cur["T"] = self.T.copy()
        if self.turb is not None:
            cur["k"] = self.turb.k.copy()
            cur["omega"] = self.turb.om.copy()
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
        ub = self._velocity_bvals(U)
        pb = self._pressure_bvals(p)
        gp = fv.gradient(p, pb)
        gU = [fv.gradient(U[:, k], ub[:, k]) for k in range(dim)]
        t1 = time.perf_counter()
        tm["gradients"] += t1 - t0

        # ---------------- effective viscosity ----------------
        if self.turb is not None:
            mut = self.turb.mut
            mu_f = mu + fv.face_interp(mut)
            self.mu_b = self.turb.wall_viscosity(U)
            mu_b = self.mu_b * self.m_wall + (mu + mut[fv.Pb]) * (1.0 - self.m_wall)
        else:
            mu_f = mu
            mu_b = mu

        # ---------------- momentum ----------------
        src = -gp * fv.vol[:, None]
        dpdx = [float(v) for v in (list(c.periodic_pressure_gradient) + [0.0] * 3)[:dim]]
        if any(dpdx):
            src = src - xp.asarray(np.asarray(dpdx), dtype=self.dtype)[None, :] * fv.vol[:, None]
        if self.energy and any(c.buoyancy[:dim]):
            bv = xp.asarray(np.asarray(c.buoyancy[:dim]), dtype=self.dtype)
            src = src + (fv.vol * (self.T - c.t_ref))[:, None] * bv[None, :]
        if self.turb is not None:
            # explicit part of div(mu_eff (grad u)^T), non-zero when mu_eff varies
            gUf = [fv.face_interp_vec(g) for g in gU]
            for i in range(dim):
                flux = sum(gUf[j][:, i] * fv.S[:, j] for j in range(dim)) * mu_f
                src[:, i] = src[:, i] + fv.Cdiff_d @ flux
        res = {}
        Ustar = xp.empty_like(U)
        au = c.relax_u
        diag, aPN, aNP, Db = self._coefficients(mu_f, self.m_dirvel, gamma_b=mu_b)
        rhs_all = [self._rhs(mu_f, U[:, k], gU[k], ub[:, k], Db, src[:, k]) for k in range(dim)]
        tcoef = self._time_coeffs()
        if tcoef is not None:
            c0 = tcoef[0]
            rv = rho * fv.vol / c.dt
            diag = diag + c0 * rv
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
        Fst = rho * (xp.sum(fv.face_interp_vec(Ustar) * fv.S, axis=1)
                     - dF * (fv.alpha * (p[N] - p[P]) - xp.sum(gpf * fv.S, axis=1)))
        if tcoef is not None:
            # Choi's correction: keeps the converged flux independent of dt
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
                       - dP[Pb] * (fv.alpha_b * (self.pb_fixed - p[Pb]) - xp.sum(gp[Pb] * fv.Sb, axis=1)))
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
        pp, pinfo = pcg(lambda v: Ap @ v, -mass, xp, M=self._amg, tol=1e-3, maxiter=200, project=project)
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

        # ---------------- energy & radiation ----------------
        if self.energy:
            res["energy"] = self._energy()
            if self.rad is not None:
                res["p1"] = self.rad.solve(self.T, self._wall_temperature())
        t5 = time.perf_counter()
        tm["energy+radiation"] += t5 - t4

        # ---------------- turbulence ----------------
        if self.turb is not None:
            ubn = self._velocity_bvals(self.U)
            gUn = [fv.gradient(self.U[:, k], ubn[:, k]) for k in range(dim)]
            res.update(self.turb.step(gUn))
            tm["turbulence"] += time.perf_counter() - t5
        if self.Y:
            t6 = time.perf_counter()
            res.update(self._species())
            tm["species"] += time.perf_counter() - t6

        self.iteration += 1
        self.step_count = self.iteration
        self.last_linear = {"momentum_iters": its, "pressure_iters": pinfo.iterations,
                            "pressure_residual": pinfo.residual}
        if self.iteration <= 5 or "continuity" not in self._res_max5:  # (also right after a restart)
            self._res_max5["continuity"] = max(self._res_max5.get("continuity", 0.0), res["continuity"], 1e-300)
        # Fluent normalises by the largest of the first five iterations; floor it by a
        # small fraction of the total face mass flux so that a start from an already
        # mass-conserving field (e.g. periodic flows) does not scale round-off to O(1).
        flux_scale = 1e-6 * float(xp.sum(xp.abs(self.F)) + xp.sum(xp.abs(self.Fb)))
        res["continuity"] /= max(self._res_max5["continuity"], flux_scale, 1e-300)
        self._last_res = res
        if c.unsteady:
            self.inner += 1
            if self.inner >= c.inner_iterations or max(res.values()) < c.inner_tol:
                self._advance_time()
        elif self.iteration > 5 and max(res.values()) < c.tol:
            self.converged = True
        return 1.0

    def _wall_temperature(self):
        """Boundary-face temperatures: fixed values, else the adjacent cell (adiabatic/flux/outflow)."""
        xp = self.xp
        TP = self.T[self.fv.Pb]
        return xp.where(self.tkind == T_FIXED, self.tval, TP)

    def _energy(self) -> float:
        c, fv, xp = self.case, self.fv, self.xp
        k_cond = c.conductivity / c.cp
        if self.turb is not None:
            mut = self.turb.mut
            gam = k_cond + fv.face_interp(mut) / c.prandtl_turbulent
            gam_b = k_cond + mut[fv.Pb] / c.prandtl_turbulent * (1.0 - self.m_wall)
        else:
            gam, gam_b = k_cond, None
        T = self.T
        TP = T[fv.Pb]
        fixed = self.tkind == T_FIXED
        flux = self.tkind == T_FLUX
        out = self.bkind == OUTLET
        Tb = xp.where(fixed, self.tval, TP)
        Tb = xp.where(out & (self.Fb < 0), self.tval, Tb)  # backflow enters at the outlet's T
        gT = fv.gradient(T, Tb)
        q_in = fv.Cb_d @ (flux * self.tval * fv.Sb_mag / c.cp)
        diag, aPN, aNP, Db = self._coefficients(gam, fixed.astype(self.dtype), gamma_b=gam_b)
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
        r = dict(getattr(self, "_last_res", {}))
        return {"raw": r, "scaled": r, "normalized": r}

    def describe(self) -> dict:
        amg = self._amg
        c = self.case
        d = {
            "solver": f"pressure-based {c.algorithm}" + (" transient " + c.time_scheme.upper() if c.unsteady else " steady"),
            "backend": self.be.name, "device": self.be.device, "precision": str(self.dtype),
            "cells": self.nc, "faces": self.mesh.n_faces, "dim": self.dim,
            "convection": c.scheme, "viscous": c.turbulence, "radiation": c.radiation,
            "pressure_solver": "AMG-PCG (" + ("pyamg" if amg is not None and not hasattr(amg, "levels")
                                               else "FlowTrace SA-AMG") + ")",
            "max_non_orthogonality_deg": round(self.fv.non_orthogonality_deg, 2),
            "setup_seconds": round(self.setup_seconds, 3),
        }
        if self.fv.periodic:
            d["periodic"] = [p["zones"] for p in self.fv.periodic]
        return d

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

    def wall_report(self) -> dict:
        """Per wall zone: area, forces (pressure + viscous), y+, skin friction, heat flux."""
        xp, fv, c = self.xp, self.fv, self.case
        h = self.be.to_numpy
        rho, mu = c.density, c.viscosity
        U = self.U
        Pb = fv.Pb
        nb = fv.nb
        rel = U[Pb] - self.ub_fixed
        rel_t = rel - xp.sum(rel * nb, axis=1)[:, None] * nb
        ut = xp.sqrt(xp.sum(rel_t * rel_t, axis=1))
        if self.turb is not None:
            muw = self.mu_b
        else:
            muw = xp.full(fv.n_bf, mu, dtype=self.dtype)
        tau_vec = muw[:, None] * rel_t / fv.yb[:, None]  # shear exerted by the fluid on the wall
        tau = xp.sqrt(xp.sum(tau_vec * tau_vec, axis=1))
        yplus = fv.yb * xp.sqrt(tau / rho) * rho / mu
        pbv = self._pressure_bvals(self.p)
        fp = pbv[:, None] * fv.Sb
        fvisc = tau_vec * fv.Sb_mag[:, None]
        qconv = None
        if self.energy:
            Tb = self._wall_temperature()
            kc = c.conductivity
            qconv = kc * (Tb - self.T[Pb]) / fv.yb * fv.Sb_mag  # into the fluid
        qrad = self.rad.wall_flux(self._wall_temperature()) if self.rad is not None else None
        hp = {k: h(v) for k, v in (("fp", fp), ("fv", fvisc), ("tau", tau), ("yp", yplus), ("A", fv.Sb_mag))}
        hq = h(qconv) if qconv is not None else None
        hr = h(qrad) if qrad is not None else None
        qdyn = 0.5 * rho * c.ref_velocity ** 2
        out = {}
        for z in self.mesh.boundary_zones():
            spec = self.zone_specs.get(z.name)
            if spec is None or spec.kind != "wall":
                continue
            m = fv.bzone == z.id
            if not m.any():
                continue
            A = hp["A"][m]
            Fp = hp["fp"][m].sum(axis=0)
            Fv = hp["fv"][m].sum(axis=0)
            Ft = Fp + Fv
            rep = {
                "area": float(A.sum()),
                "force_pressure": Fp.tolist(), "force_viscous": Fv.tolist(), "force_total": Ft.tolist(),
                "cd": float(Ft[0] / (qdyn * c.ref_area)), "cl": float(Ft[1] / (qdyn * c.ref_area)),
                "yplus_avg": float((hp["yp"][m] * A).sum() / A.sum()), "yplus_max": float(hp["yp"][m].max()),
                "cf_avg": float((hp["tau"][m] * A).sum() / A.sum() / qdyn),
            }
            if hq is not None:
                rep["heat_flow_conv"] = float(hq[m].sum())
            if hr is not None:
                rep["heat_flow_rad"] = float(hr[m].sum())
            out[z.name] = rep
        return out

    def wall_profile(self, zone: str) -> dict:
        """Face-wise distributions on one wall zone (x, y+, Cf, Cp) for plotting."""
        xp, fv, c = self.xp, self.fv, self.case
        h = self.be.to_numpy
        z = self.mesh.zone_by_name(zone)
        m = fv.bzone == z.id
        rho, mu = c.density, c.viscosity
        Pb = fv.Pb
        rel = self.U[Pb] - self.ub_fixed
        nb = fv.nb
        rel_t = rel - xp.sum(rel * nb, axis=1)[:, None] * nb
        ut = xp.sqrt(xp.sum(rel_t * rel_t, axis=1))
        muw = self.mu_b if self.turb is not None else xp.full(fv.n_bf, mu, dtype=self.dtype)
        tau = muw * ut / fv.yb
        qdyn = 0.5 * rho * c.ref_velocity ** 2
        xfb = h(fv.xfb)[m]
        order = np.argsort(xfb[:, 0])
        return {"x": xfb[order, 0], "y": xfb[order, 1], "cf": h(tau)[m][order] / qdyn,
                "cp": h(self._pressure_bvals(self.p))[m][order] / qdyn,
                "yplus": h(fv.yb * xp.sqrt(tau / rho) * rho / mu)[m][order]}

    def state_dict(self) -> dict:
        h = self.be.to_numpy
        st = {"U": h(self.U), "p": h(self.p), "F": h(self.F), "Fb": h(self.Fb), "iteration": self.iteration,
              "time": self.time, "time_step": self.time_step,
              "res_continuity_norm": self._res_max5.get("continuity", 0.0)}
        if self.energy:
            st["T"] = h(self.T)
        if self.turb is not None:
            st["k"] = h(self.turb.k)
            st["omega"] = h(self.turb.om)
            st["mut"] = h(self.turb.mut)
        if self.rad is not None:
            st["G"] = h(self.rad.G)
        for name, Y in self.Y.items():
            st[f"Y:{name}"] = h(Y)
        return st

    def load_state(self, st: dict):
        xp = self.xp
        dev = lambda a: xp.asarray(a, dtype=self.dtype)  # noqa: E731
        self.U = dev(st["U"])
        self.p = dev(st["p"])
        self.F = dev(st["F"])
        self.Fb = dev(st["Fb"])
        if self.energy and "T" in st:
            self.T = dev(st["T"])
        if self.turb is not None and "k" in st:
            self.turb.k, self.turb.om, self.turb.mut = dev(st["k"]), dev(st["omega"]), dev(st["mut"])
        if self.rad is not None and "G" in st:
            self.rad.G = dev(st["G"])
        for name in self.Y:
            if f"Y:{name}" in st:
                self.Y[name] = dev(st[f"Y:{name}"])
        self.iteration = int(st.get("iteration", 0))
        if float(st.get("res_continuity_norm", 0.0)) > 0:
            self._res_max5["continuity"] = float(st["res_continuity_norm"])
        self.time = float(st.get("time", 0.0))
        self.time_step = int(st.get("time_step", 0))
        if self.case.unsteady:
            self._old = {}
            self._F_old = []
            self._store_old()
