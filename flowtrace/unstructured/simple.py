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
    def _coefficients(self, gamma, dirichlet):
        """Implicit part of div(F phi) - div(gamma grad phi): (diag, aPN, aNP, Db)."""
        fv, xp = self.fv, self.xp
        F, Fb = self.F, self.Fb
        D = gamma * fv.alpha
        aPN = -(D + xp.maximum(-F, 0.0))  # coefficient of phi_N in P's row
        aNP = -(D + xp.maximum(F, 0.0))  # coefficient of phi_P in N's row
        Db = gamma * fv.alpha_b * dirichlet
        diag = fv.scatter(D + xp.maximum(F, 0.0), D + xp.maximum(-F, 0.0), Db + xp.maximum(Fb, 0.0))
        return diag, aPN, aNP, Db

    def _rhs(self, gamma, phi, grad, phi_b, Db, extra=None):
        """Boundary sources + explicit non-orthogonal and second-order corrections."""
        fv, xp = self.fv, self.xp
        F = self.F
        rhs = fv.Cb_d @ ((Db + xp.maximum(-self.Fb, 0.0)) * phi_b)
        # non-orthogonal diffusion flux into P: gamma (grad phi)_f . T_f, outward from P
        gf = fv.face_interp_vec(grad)
        rhs = rhs + fv.Cdiff_d @ (gamma * xp.sum(gf * fv.Tvec, axis=1))
        if self.case.scheme == "second-order":
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
        ub = self._velocity_bvals(U)
        pb = self._pressure_bvals(p)
        gp = fv.gradient(p, pb)
        gU = [fv.gradient(U[:, k], ub[:, k]) for k in range(dim)]
        t1 = time.perf_counter()
        tm["gradients"] += t1 - t0

        # ---------------- momentum ----------------
        src = -gp * fv.vol[:, None]
        if self.energy and any(c.buoyancy[:dim]):
            bv = xp.asarray(np.asarray(c.buoyancy[:dim]), dtype=self.dtype)
            src = src + (fv.vol * (self.T - c.t_ref))[:, None] * bv[None, :]
        res = {}
        Ustar = xp.empty_like(U)
        au = c.relax_u
        diag, aPN, aNP, Db = self._coefficients(mu, self.m_dirvel)
        rhs_all = [self._rhs(mu, U[:, k], gU[k], ub[:, k], Db, src[:, k]) for k in range(dim)]
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

        # ---------------- energy ----------------
        if self.energy:
            res["energy"] = self._energy()
        t5 = time.perf_counter()
        tm["correct+energy"] += t5 - t4

        self.iteration += 1
        self.step_count = self.iteration
        self.last_linear = {"momentum_iters": its, "pressure_iters": pinfo.iterations,
                            "pressure_residual": pinfo.residual}
        if self.iteration <= 5:
            self._res_max5["continuity"] = max(self._res_max5.get("continuity", 0.0), res["continuity"], 1e-300)
        res["continuity"] /= self._res_max5["continuity"]
        self._last_res = res
        if self.iteration > 5 and max(res.values()) < c.tol:
            self.converged = True
        return 1.0

    def _energy(self) -> float:
        c, fv, xp = self.case, self.fv, self.xp
        gam = c.conductivity / c.cp
        T = self.T
        TP = T[fv.Pb]
        fixed = self.tkind == T_FIXED
        flux = self.tkind == T_FLUX
        out = self.bkind == OUTLET
        Tb = xp.where(fixed, self.tval, TP)
        Tb = xp.where(out & (self.Fb < 0), self.tval, Tb)  # backflow enters at the outlet's T
        gT = fv.gradient(T, Tb)
        q_in = fv.Cb_d @ (flux * self.tval * fv.Sb_mag / c.cp)
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
        r = dict(getattr(self, "_last_res", {}))
        return {"raw": r, "scaled": r, "normalized": r}

    def describe(self) -> dict:
        amg = self._amg
        return {
            "solver": f"pressure-based {self.case.algorithm}",
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

    def state_dict(self) -> dict:
        h = self.be.to_numpy
        st = {"U": h(self.U), "p": h(self.p), "F": h(self.F), "Fb": h(self.Fb), "iteration": self.iteration}
        if self.energy:
            st["T"] = h(self.T)
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
