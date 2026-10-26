r"""Density-based finite-volume solver for compressible flow (Euler / laminar Navier-Stokes).

Conservative variables Q = (rho, rho u, rho E) on cell centres of any polygonal / polyhedral
mesh (2-D, 3-D, or 2-D axisymmetric with x the axis and y the radius, as in Fluent).

* inviscid flux: HLLC (Toro), wave speeds from Roe averages (Einfeldt);
* second order: MUSCL reconstruction of the primitive variables (rho, u, p) with
  least-squares gradients and a Venkatakrishnan (or Barth-Jespersen) limiter, falling back
  to first order on any face where the reconstruction would lose positivity;
* viscous flux (optional): constant or Sutherland viscosity, face gradients from averaged cell
  gradients with a directional correction, Fourier heat flux with a constant Prandtl number;
* time: transient SSP-RK3 with a global time step, or steady-state with a local time step,
  either explicit (SSP-RK3) or matrix-free implicit Euler: the first-order Rusanov Jacobian
  is applied with Jacobi sweeps, so every operation is a face or cell loop (vectorised here,
  one kernel per loop on the GPU) and the CFL number can grow to O(10-100);
* boundary conditions by ghost states: pressure inlet (p0, T0), pressure outlet, slip and
  no-slip walls (adiabatic or isothermal), symmetry/axis, characteristic far field, velocity
  and supersonic inlets.

Runs unchanged on NumPy or CuPy arrays (``backend.xp``).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict

import numpy as np

from ..hardware.backend import get_backend
from ..mesh.unstructured import UnstructuredMesh
from ..unstructured.fvmesh import FVMesh
from .case import FLUENT_TO_COMP, CompBC, CompressibleCase

WALL, SLIP, P_IN, P_OUT, FARFIELD, V_IN, SUP_IN = range(7)
_CODES = {"wall": WALL, "slip-wall": SLIP, "symmetry": SLIP, "pressure-inlet": P_IN, "pressure-outlet": P_OUT,
          "far-field": FARFIELD, "velocity-inlet": V_IN, "supersonic-inlet": SUP_IN}


def _scatter_max(xp, out, idx, vals):
    if xp is np:
        np.maximum.at(out, idx, vals)
    else:
        import cupyx  # type: ignore

        cupyx.scatter_max(out, idx, vals)


def _scatter_min(xp, out, idx, vals):
    if xp is np:
        np.minimum.at(out, idx, vals)
    else:
        import cupyx  # type: ignore

        cupyx.scatter_min(out, idx, vals)


class CompressibleSolver:
    def __init__(self, mesh: UnstructuredMesh, case: CompressibleCase, backend=None):
        t0 = time.perf_counter()
        self.mesh, self.case = mesh, case
        self.be = backend if backend is not None else get_backend(case.backend)
        xp = self.xp = self.be.xp
        self.dtype = np.dtype(np.float32 if case.precision == "fp32" else np.float64)
        self.fv = fv = FVMesh(mesh, self.be, self.dtype)
        self.dim = dim = mesh.dim
        if case.axisymmetric and dim != 2:
            raise ValueError("axisymmetric cases need a 2-D mesh (x = axis, y = radius)")
        self.nc, self.nvar = mesh.n_cells, dim + 2
        g = case.gas
        self.gamma, self.R = g.gamma, g.R
        h = self.be.to_numpy
        # geometry: unit normals and (axisymmetric: radius-weighted) areas and volumes
        A = xp.sqrt(xp.sum(fv.S * fv.S, axis=1))
        Ab = xp.sqrt(xp.sum(fv.Sb * fv.Sb, axis=1))
        self.n_i, self.n_b = fv.S / A[:, None], fv.Sb / Ab[:, None]
        self.vol_planar = fv.vol
        if case.axisymmetric:
            ylo = float(h(fv.xc)[:, 1].min())
            if ylo < -1e-12 * max(1.0, float(h(fv.xc)[:, 1].max())):
                raise ValueError("axisymmetric mesh must lie in y >= 0 (y is the radius)")
            self.A_i = A * xp.maximum(fv.xf[:, 1], 0.0)
            self.A_b = Ab * xp.maximum(fv.xfb[:, 1], 0.0)
            self.V = fv.vol * fv.xc[:, 1]
        else:
            self.A_i, self.A_b, self.V = A, Ab, fv.vol
        # cell size for the limiter threshold and the viscous spectral radius
        self.h = fv.vol ** (1.0 / dim)
        self.d_len = xp.sqrt(xp.sum(fv.d * fv.d, axis=1))
        self.db_len = xp.maximum(xp.abs(xp.sum(fv.db * self.n_b, axis=1)), 1e-300)
        self._build_boundaries()
        # initial field
        u0 = (list(case.init_velocity) + [0.0] * 3)[:dim]
        W = np.zeros((self.nc, self.nvar))
        W[:, 0] = case.init_p / (self.R * case.init_T)
        W[:, 1:1 + dim] = u0
        W[:, -1] = case.init_p
        self.Q = self.cons(xp.asarray(W, dtype=self.dtype))
        self.iteration = 0
        self.time = 0.0
        self.cfl = case.cfl
        self.converged = False
        self.timers: Dict[str, float] = defaultdict(float)
        self._res0: Dict[str, float] = {}
        self._last_res: Dict[str, float] = {}
        self._prev_rmax = math.inf
        self.history = []
        self.setup_seconds = time.perf_counter() - t0

    def initialize(self, rho, velocity, p) -> None:
        """Set the field from cell values (arrays or scalars) of density, velocity (nc, dim) and pressure."""
        xp, d = self.xp, self.dim
        W = np.zeros((self.nc, self.nvar))
        W[:, 0] = np.broadcast_to(np.asarray(rho, dtype=float), (self.nc,))
        W[:, 1:1 + d] = np.broadcast_to(np.asarray(velocity, dtype=float), (self.nc, d))
        W[:, -1] = np.broadcast_to(np.asarray(p, dtype=float), (self.nc,))
        self.Q = self.cons(xp.asarray(W, dtype=self.dtype))

    # ------------------------------------------------------------------ state conversion
    def prim(self, Q):
        xp, g, d = self.xp, self.gamma, self.dim
        rho = Q[:, 0]
        u = Q[:, 1:1 + d] / rho[:, None]
        p = (g - 1.0) * (Q[:, -1] - 0.5 * rho * xp.sum(u * u, axis=1))
        return xp.concatenate([rho[:, None], u, p[:, None]], axis=1)

    def cons(self, W):
        xp, g, d = self.xp, self.gamma, self.dim
        rho, u, p = W[:, 0], W[:, 1:1 + d], W[:, -1]
        E = p / (g - 1.0) + 0.5 * rho * xp.sum(u * u, axis=1)
        return xp.concatenate([rho[:, None], rho[:, None] * u, E[:, None]], axis=1)

    # ------------------------------------------------------------------ boundaries
    def _build_boundaries(self):
        fv, xp, case, d = self.fv, self.xp, self.case, self.dim
        nb = fv.n_bf
        code = np.full(nb, WALL, dtype=np.int64)
        p0 = np.zeros(nb)
        T0 = np.zeros(nb)
        pb = np.zeros(nb)
        Tb = np.zeros(nb)
        mach = np.zeros(nb)
        dirv = np.zeros((nb, d))
        vel = np.zeros((nb, d))
        wallT = np.full(nb, np.nan)
        has_dir = np.zeros(nb, dtype=bool)
        self.zone_specs = {}
        for z in self.mesh.boundary_zones():
            spec = case.boundaries.get(z.name) or CompBC(kind=FLUENT_TO_COMP.get(z.type, "wall"))
            self.zone_specs[z.name] = spec
            m = fv.bzone == z.id
            if not m.any():
                continue
            code[m] = _CODES[spec.kind]
            if spec.kind == "wall" and not case.viscous:
                code[m] = SLIP
            p0[m], T0[m], pb[m], Tb[m], mach[m] = spec.p0, spec.T0, spec.p, spec.T, spec.mach
            if spec.direction is not None:
                v = np.asarray((list(spec.direction) + [0.0] * 3)[:d], dtype=float)
                dirv[m] = v / np.linalg.norm(v)
                has_dir[m] = True
            if spec.velocity is not None:
                vel[m] = (list(spec.velocity) + [0.0] * 3)[:d]
            if spec.wall_T is not None:
                wallT[m] = spec.wall_T
        n_h = self.be.to_numpy(self.n_b).astype(float)
        dirv[~has_dir] = -n_h[~has_dir]  # default inflow direction: normal into the domain
        a = lambda v: xp.asarray(v, dtype=self.dtype)  # noqa: E731
        self.bcode = xp.asarray(code)
        self.bc_p0, self.bc_T0, self.bc_p, self.bc_T, self.bc_mach = a(p0), a(T0), a(pb), a(Tb), a(mach)
        self.bc_dir, self.bc_vel = a(dirv), a(vel)
        self.bc_wallT = a(wallT)
        self.is_iso = xp.asarray(~np.isnan(wallT))

    def _ghost(self, WP):
        """Ghost (outer) primitive state on every boundary face from the inner state WP."""
        xp, g, R, d = self.xp, self.gamma, self.R, self.dim
        n = self.n_b
        code = self.bcode
        rho, u, p = WP[:, 0], WP[:, 1:1 + d], WP[:, -1]
        un = xp.sum(u * n, axis=1)
        c = xp.sqrt(g * p / rho)
        out = WP.copy()

        def put(mask, r, uu, pp):
            m = mask[:, None]
            new = xp.concatenate([r[:, None], uu, pp[:, None]], axis=1)
            return xp.where(m, new, out)

        # slip wall / symmetry / axis: mirror the normal velocity
        um = u - 2.0 * un[:, None] * n
        out = put(code == SLIP, rho, um, p)
        # no-slip wall: reflect the velocity; isothermal walls fix T at the face
        Tw = xp.where(self.is_iso, self.bc_wallT, p / (rho * R))
        Tg = xp.where(self.is_iso, xp.maximum(2.0 * Tw - p / (rho * R), 0.5 * Tw), p / (rho * R))
        out = put(code == WALL, p / (R * Tg), -u, p)
        # pressure outlet: static p when subsonic, extrapolate when supersonic; backflow at T
        sup = un >= c
        p_o = xp.where(sup, p, self.bc_p)
        r_o = xp.where(un < 0, p_o / (R * self.bc_T), rho * (p_o / p) ** (1.0 / g))  # isentropic to p_out
        out = put(code == P_OUT, r_o, u, p_o)
        # pressure inlet: total conditions, static pressure from the interior (subsonic)
        p_in = xp.minimum(p, self.bc_p0 * (1.0 - 1e-9))
        M2 = 2.0 / (g - 1.0) * ((self.bc_p0 / p_in) ** ((g - 1.0) / g) - 1.0)
        sup_in = M2 >= 1.0
        p_in = xp.where(sup_in, xp.minimum(self.bc_p, self.bc_p0 * (1.0 - 1e-9)), p_in)
        M2 = 2.0 / (g - 1.0) * ((self.bc_p0 / p_in) ** ((g - 1.0) / g) - 1.0)
        T_in = self.bc_T0 / (1.0 + 0.5 * (g - 1.0) * M2)
        vmag = xp.sqrt(xp.maximum(M2, 0.0) * g * R * T_in)
        out = put(code == P_IN, p_in / (R * T_in), vmag[:, None] * self.bc_dir, p_in)
        # velocity inlet: u and T imposed, p from the interior
        out = put(code == V_IN, p / (R * self.bc_T), self.bc_vel, p)
        # supersonic inlet: everything imposed
        out = put(code == SUP_IN, self.bc_p / (R * self.bc_T), self.bc_vel, self.bc_p)
        # characteristic far field (Riemann invariants normal to the face)
        c_inf = xp.sqrt(g * R * self.bc_T)
        u_inf = (self.bc_mach * c_inf)[:, None] * self.bc_dir
        r_inf = self.bc_p / (R * self.bc_T)
        un_inf = xp.sum(u_inf * n, axis=1)
        Rp = un + 2.0 * c / (g - 1.0)
        Rm = un_inf - 2.0 * c_inf / (g - 1.0)
        un_b = 0.5 * (Rp + Rm)
        c_b = 0.25 * (g - 1.0) * (Rp - Rm)
        inflow = un_b < 0
        s_b = xp.where(inflow, self.bc_p / r_inf ** g, p / rho ** g)
        ut_b = xp.where(inflow[:, None], u_inf - un_inf[:, None] * n, u - un[:, None] * n)
        r_b = (xp.maximum(c_b, 1e-12) ** 2 / (g * s_b)) ** (1.0 / (g - 1.0))
        p_b = r_b * c_b ** 2 / g
        u_b = ut_b + un_b[:, None] * n
        supersonic = xp.abs(un_inf) >= c_inf
        r_ff = xp.where(supersonic, xp.where(un_inf < 0, r_inf, rho), r_b)
        p_ff = xp.where(supersonic, xp.where(un_inf < 0, self.bc_p, p), p_b)
        u_ff = xp.where(supersonic[:, None], xp.where((un_inf < 0)[:, None], u_inf, u), u_b)
        out = put(code == FARFIELD, r_ff, u_ff, p_ff)
        return out

    # ------------------------------------------------------------------ fluxes
    def _euler_flux(self, W, n):
        """Physical Euler flux F(W).n per unit area."""
        xp, g, d = self.xp, self.gamma, self.dim
        rho, u, p = W[:, 0], W[:, 1:1 + d], W[:, -1]
        un = xp.sum(u * n, axis=1)
        E = p / (g - 1.0) + 0.5 * rho * xp.sum(u * u, axis=1)
        return xp.concatenate([(rho * un)[:, None], rho[:, None] * u * un[:, None] + p[:, None] * n,
                               ((E + p) * un)[:, None]], axis=1)

    def _hllc(self, WL, WR, n):
        """HLLC flux per unit area (Toro 2009, ch. 10) with Einfeldt / Roe wave-speed estimates."""
        xp, g, d = self.xp, self.gamma, self.dim
        rL, uL, pL = WL[:, 0], WL[:, 1:1 + d], WL[:, -1]
        rR, uR, pR = WR[:, 0], WR[:, 1:1 + d], WR[:, -1]
        unL, unR = xp.sum(uL * n, axis=1), xp.sum(uR * n, axis=1)
        cL, cR = xp.sqrt(g * pL / rL), xp.sqrt(g * pR / rR)
        EL = pL / (g - 1.0) + 0.5 * rL * xp.sum(uL * uL, axis=1)
        ER = pR / (g - 1.0) + 0.5 * rR * xp.sum(uR * uR, axis=1)
        sl, sr = xp.sqrt(rL), xp.sqrt(rR)
        w = sl / (sl + sr)
        u_roe = w[:, None] * uL + (1.0 - w)[:, None] * uR
        H_roe = w * (EL + pL) / rL + (1.0 - w) * (ER + pR) / rR
        c_roe = xp.sqrt(xp.maximum((g - 1.0) * (H_roe - 0.5 * xp.sum(u_roe * u_roe, axis=1)), 1e-30))
        un_roe = xp.sum(u_roe * n, axis=1)
        SL = xp.minimum(unL - cL, un_roe - c_roe)
        SR = xp.maximum(unR + cR, un_roe + c_roe)
        den = rL * (SL - unL) - rR * (SR - unR)
        SM = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / xp.where(den == 0, 1e-300, den)
        FL = self._euler_flux(WL, n)
        FR = self._euler_flux(WR, n)

        def star(r, u, p, un, E, S):
            coef = r * (S - un) / xp.where(S == SM, 1e-300, S - SM)
            mom = coef[:, None] * (u + (SM - un)[:, None] * n)
            ene = coef * (E / r + (SM - un) * (SM + p / (r * xp.where(S == un, 1e-300, S - un))))
            return xp.concatenate([coef[:, None], mom, ene[:, None]], axis=1)

        QL = xp.concatenate([rL[:, None], rL[:, None] * uL, EL[:, None]], axis=1)
        QR = xp.concatenate([rR[:, None], rR[:, None] * uR, ER[:, None]], axis=1)
        FsL = FL + SL[:, None] * (star(rL, uL, pL, unL, EL, SL) - QL)
        FsR = FR + SR[:, None] * (star(rR, uR, pR, unR, ER, SR) - QR)
        return xp.where((SL >= 0)[:, None], FL, xp.where((SM >= 0)[:, None], FsL,
                                                          xp.where((SR > 0)[:, None], FsR, FR)))

    def _viscosity(self, T):
        g = self.case.gas
        if not g.sutherland:
            return g.mu + 0.0 * T
        return g.mu * (T / g.t_ref) ** 1.5 * (g.t_ref + g.s_const) / (T + g.s_const)

    # ------------------------------------------------------------------ residual
    def _gradients(self, W, Wface_b):
        """Least-squares gradients (nc, nvar, dim) of the primitive variables."""
        xp = self.xp
        return xp.stack([self.fv.gradient(W[:, k], Wface_b[:, k]) for k in range(self.nvar)], axis=1)

    def _limiter(self, W, Wface_b, grad):
        xp, fv = self.xp, self.fv
        P, N, Pb = fv.P, fv.N, fv.Pb
        Wmax, Wmin = W.copy(), W.copy()
        for idx, other in ((P, W[N]), (N, W[P]), (Pb, Wface_b)):
            _scatter_max(xp, Wmax, idx, other)
            _scatter_min(xp, Wmin, idx, other)
        phi = xp.ones_like(W)
        dW_range = xp.maximum(xp.max(Wmax, axis=0) - xp.min(Wmin, axis=0), 1e-30)
        for cells, dx in ((P, fv.dP_f), (N, fv.dN_f), (Pb, fv.db)):
            d2 = xp.einsum("fkd,fd->fk", grad[cells], dx)
            d1 = xp.where(d2 > 0, Wmax[cells] - W[cells], Wmin[cells] - W[cells])
            if self.case.limiter == "barth":
                ph = xp.where(xp.abs(d2) > 1e-300, xp.minimum(1.0, d1 / xp.where(d2 == 0, 1e-300, d2)), 1.0)
            else:  # Venkatakrishnan, threshold scaled with the variable's global range
                eps2 = (self.case.venkat_k * self.h[cells] / self._L) ** 3
                eps2 = eps2[:, None] * dW_range[None, :] ** 2
                num = (d1 ** 2 + eps2) * d2 + 2.0 * d2 ** 2 * d1
                den = d2 * (d1 ** 2 + 2.0 * d2 ** 2 + d1 * d2 + eps2)
                ph = xp.where(xp.abs(d2) > 1e-300, num / xp.where(den == 0, 1e-300, den), 1.0)
            _scatter_min(xp, phi, cells, xp.clip(ph, 0.0, 1.0))
        return phi

    def residual(self, Q):
        """Net outflow R(Q) per cell and equation (residual = rate of change x V with a minus sign)."""
        xp, fv = self.xp, self.fv
        W = self.prim(Q)
        P, N, Pb = fv.P, fv.N, fv.Pb
        WP_b = W[Pb]
        Wg = self._ghost(WP_b)
        Wface_b = 0.5 * (WP_b + Wg)
        grad = None
        if self.case.order == 2 or self.case.viscous:
            grad = self._gradients(W, Wface_b)
        if self.case.order == 2:
            phi = self._limiter(W, Wface_b, grad) if self.case.limiter != "none" else xp.ones_like(W)
            WL = W[P] + phi[P] * xp.einsum("fkd,fd->fk", grad[P], fv.dP_f)
            WR = W[N] + phi[N] * xp.einsum("fkd,fd->fk", grad[N], fv.dN_f)
            WbL = WP_b + phi[Pb] * xp.einsum("fkd,fd->fk", grad[Pb], fv.db)
            bad = (WL[:, 0] <= 0) | (WL[:, -1] <= 0) | (WR[:, 0] <= 0) | (WR[:, -1] <= 0)
            WL = xp.where(bad[:, None], W[P], WL)
            WR = xp.where(bad[:, None], W[N], WR)
            badb = (WbL[:, 0] <= 0) | (WbL[:, -1] <= 0)
            WbL = xp.where(badb[:, None], WP_b, WbL)
            Wgb = self._ghost(WbL)
        else:
            WL, WR, WbL, Wgb = W[P], W[N], WP_b, Wg
        F = self._hllc(WL, WR, self.n_i) * self.A_i[:, None]
        Fb = self._hllc(WbL, Wgb, self.n_b) * self.A_b[:, None]
        if self.case.viscous:
            Fv, Fvb = self._viscous_fluxes(W, grad, Wface_b, WP_b, Wg)
            F = F - Fv * self.A_i[:, None]
            Fb = Fb - Fvb * self.A_b[:, None]
        R = fv.Cdiff_d @ F + fv.Cb_d @ Fb
        if self.case.axisymmetric:
            # pressure (and hoop stress) on the two wedge faces: a radial source per radian
            src = W[:, -1]
            if self.case.viscous:
                T = W[:, -1] / (W[:, 0] * self.R)
                mu = self._viscosity(T)
                r = xp.maximum(fv.xc[:, 1], 1e-30)
                div = grad[:, 1, 0] + grad[:, 2, 1] + W[:, 2] / r
                src = src - 2.0 * mu * (W[:, 2] / r - div / 3.0)
            R[:, 2] = R[:, 2] - src * self.vol_planar
        return R, W

    def _viscous_fluxes(self, W, grad, Wface_b, WP_b, Wg):
        """Viscous flux tau.n and energy flux (tau.u + k grad T).n per unit area."""
        xp, fv, d = self.xp, self.fv, self.dim
        R = self.R
        P, N, Pb = fv.P, fv.N, fv.Pb
        T = W[:, -1] / (W[:, 0] * R)
        gT = (grad[:, -1, :] - (R * T)[:, None] * grad[:, 0, :]) / (W[:, 0] * R)[:, None]
        gu = grad[:, 1:1 + d, :]  # (nc, d, d): gu[c, i, j] = d u_i / d x_j

        def face_grad(gc_P, gc_N, phiP, phiN, dvec, dlen2):
            gavg = 0.5 * (gc_P + gc_N)
            corr = (phiN - phiP - xp.sum(gavg * dvec, axis=-1)) / dlen2
            return gavg + corr[..., None] * dvec

        dl2 = xp.sum(fv.d * fv.d, axis=1)
        gu_f = face_grad(gu[P], gu[N], W[P, 1:1 + d], W[N, 1:1 + d], fv.d[:, None, :], dl2[:, None])
        gT_f = face_grad(gT[P], gT[N], T[P], T[N], fv.d, dl2)
        u_f = 0.5 * (W[P, 1:1 + d] + W[N, 1:1 + d])
        T_f = 0.5 * (T[P] + T[N])
        r_f = fv.xf[:, 1] if self.case.axisymmetric else None
        Fv = self._tau_flux(gu_f, gT_f, u_f, T_f, self.n_i, r_f)
        # boundary faces: one-sided gradients towards the face value
        Tb_g = Wg[:, -1] / (Wg[:, 0] * R)
        Tf_b = 0.5 * (T[Pb] + Tb_g)
        uf_b = 0.5 * (WP_b[:, 1:1 + d] + Wg[:, 1:1 + d])
        dbl2 = xp.sum(fv.db * fv.db, axis=1)
        gu_b = gu[Pb] + ((uf_b - W[Pb, 1:1 + d] - xp.sum(gu[Pb] * fv.db[:, None, :], axis=-1)) / dbl2[:, None])[..., None] \
            * fv.db[:, None, :]
        gT_b = gT[Pb] + ((Tf_b - T[Pb] - xp.sum(gT[Pb] * fv.db, axis=1)) / dbl2)[:, None] * fv.db
        adiabatic = (self.bcode == WALL) & ~self.is_iso
        gT_b = xp.where(adiabatic[:, None], gT_b - xp.sum(gT_b * self.n_b, axis=1)[:, None] * self.n_b, gT_b)
        noflux = (self.bcode == SLIP) | (self.bcode == FARFIELD) | (self.bcode == P_OUT)
        rb = fv.xfb[:, 1] if self.case.axisymmetric else None
        Fvb = self._tau_flux(gu_b, gT_b, uf_b, Tf_b, self.n_b, rb)
        Fvb = xp.where(noflux[:, None], 0.0 * Fvb, Fvb)  # zero viscous stress on symmetry, far field, outlets
        return Fv, Fvb

    def _tau_flux(self, gu, gT, u, T, n, r=None):
        xp, d = self.xp, self.dim
        mu = self._viscosity(T)
        k = mu * self.case.gas.cp / self.case.gas.prandtl
        div = xp.einsum("fii->f", gu)
        if r is not None:  # axisymmetric: + v / r
            div = div + xp.where(r > 1e-30, u[:, 1] / xp.where(r > 1e-30, r, 1.0), gu[:, 1, 1])
        tau = mu[:, None, None] * (gu + xp.swapaxes(gu, 1, 2))
        idx = xp.arange(d)
        tau[:, idx, idx] -= (2.0 / 3.0) * (mu * div)[:, None]
        tn = xp.einsum("fij,fj->fi", tau, n)
        en = xp.sum(tn * u, axis=1) + k * xp.sum(gT * n, axis=1)
        zero = xp.zeros_like(en)
        return xp.concatenate([zero[:, None], tn, en[:, None]], axis=1)

    # ------------------------------------------------------------------ time step
    def _spectral(self, W):
        """Face spectral radii (|u.n| + c) A (+ viscous part) on interior and boundary faces."""
        xp, fv, g, d = self.xp, self.fv, self.gamma, self.dim
        P, N, Pb = fv.P, fv.N, fv.Pb
        c = xp.sqrt(g * W[:, -1] / W[:, 0])
        un_P = xp.abs(xp.sum(W[P, 1:1 + d] * self.n_i, axis=1))
        un_N = xp.abs(xp.sum(W[N, 1:1 + d] * self.n_i, axis=1))
        lam = xp.maximum(un_P + c[P], un_N + c[N]) * self.A_i
        lamb = (xp.abs(xp.sum(W[Pb, 1:1 + d] * self.n_b, axis=1)) + c[Pb]) * self.A_b
        if self.case.viscous:
            T = W[:, -1] / (W[:, 0] * self.R)
            nu = self._viscosity(T) / W[:, 0] * max(4.0 / 3.0, g / self.case.gas.prandtl)
            lam = lam + 2.0 * xp.maximum(nu[P], nu[N]) * self.A_i / self.d_len
            lamb = lamb + 2.0 * nu[Pb] * self.A_b / self.db_len
        return lam, lamb

    def _local_dt(self, W, cfl):
        fv = self.fv
        lam, lamb = self._spectral(W)
        Lc = fv.Cown_d @ lam + fv.Cnei_d @ lam + fv.Cb_d @ lamb
        return cfl * self.V / Lc, lam, lamb, Lc

    # ------------------------------------------------------------------ iteration
    @property
    def _L(self):
        if not hasattr(self, "_Lref"):
            lo, hi = self.mesh.bounding_box()
            self._Lref = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) or 1.0
        return self._Lref

    def _positive(self, Q, dQ):
        """Largest relaxation in (0, 1] keeping rho and p positive (scaled update)."""
        xp = self.xp
        Qn = Q + dQ
        Wn = self.prim(Qn)
        ok = (Wn[:, 0] > 0) & (Wn[:, -1] > 0)
        if bool(xp.all(ok)):
            return dQ
        for a in (0.5, 0.25, 0.1, 0.03):
            Wn = self.prim(Q + a * dQ)
            okc = (Wn[:, 0] > 0) & (Wn[:, -1] > 0)
            dQ = xp.where(ok[:, None], dQ, a * dQ)
            ok = ok | okc
            if bool(xp.all(ok)):
                return dQ
        return xp.where(ok[:, None], dQ, 0.0 * dQ)

    def step(self) -> None:
        if self.finished:
            return
        c = self.case
        t0 = time.perf_counter()
        if c.time == "transient":
            self._step_rk3(global_dt=True)
        elif c.time == "steady-explicit":
            self._step_rk3(global_dt=False)
        else:
            self._step_implicit()
        self.timers["iteration"] += time.perf_counter() - t0
        self.iteration += 1

    def _step_rk3(self, global_dt: bool):
        xp, c = self.xp, self.case
        Q0 = self.Q
        W = self.prim(Q0)
        dt, _, _, _ = self._local_dt(W, c.cfl)
        if global_dt:
            dtg = float(xp.min(dt)) if c.dt is None else float(c.dt)
            dtg = min(dtg, c.t_end - self.time) if c.t_end else dtg
            dt = xp.full(self.nc, dtg, dtype=self.dtype)
        R, _ = self.residual(Q0)
        self._record(R)
        Q1 = Q0 - R * (dt / self.V)[:, None]
        R1, _ = self.residual(Q1)
        Q2 = 0.75 * Q0 + 0.25 * (Q1 - R1 * (dt / self.V)[:, None])
        R2, _ = self.residual(Q2)
        self.Q = Q0 / 3.0 + (2.0 / 3.0) * (Q2 - R2 * (dt / self.V)[:, None])
        if global_dt:
            self.time += float(dt[0])

    def _step_implicit(self):
        """Implicit Euler, first-order Rusanov Jacobian, Jacobi sweeps (matrix-free)."""
        fv, c = self.fv, self.case
        P, N = fv.P, fv.N
        Q = self.Q
        R, W = self.residual(Q)
        rmax = self._record(R)
        # CFL ramp: grow while the residual falls, back off when it rises
        if rmax < self._prev_rmax:
            self.cfl = min(c.cfl_max, self.cfl * c.cfl_growth)
        elif rmax > 1.5 * self._prev_rmax:
            self.cfl = max(c.cfl, 0.5 * self.cfl)
        self._prev_rmax = rmax
        dt, lam, lamb, Lc = self._local_dt(W, self.cfl)
        D = self.V / dt + 0.5 * Lc
        dQ = -R / D[:, None]
        n = self.n_i
        A = self.A_i[:, None]
        WP, WN = W[P], W[N]
        FP0, FN0 = self._euler_flux(WP, n), self._euler_flux(WN, n)
        lamA = lam[:, None]
        for _ in range(c.jacobi_sweeps):
            dQP, dQN = dQ[P], dQ[N]
            WPn = self.prim(Q[P] + dQP)
            WNn = self.prim(Q[N] + dQN)
            dFN = self._euler_flux(WNn, n) - FN0
            dFP = self._euler_flux(WPn, n) - FP0
            offP = 0.5 * (dFN * A - lamA * dQN)  # neighbour N in P's row (normal out of P)
            offN = 0.5 * (-dFP * A - lamA * dQP)  # neighbour P in N's row (normal out of N is -n)
            dQ = (-R - fv.Cown_d @ offP - fv.Cnei_d @ offN) / D[:, None]
        self.Q = Q + self._positive(Q, dQ)

    def _record(self, R) -> float:
        """Scaled residuals: RMS per equation, divided by its largest value over the first 5 iterations."""
        xp = self.xp
        rms = xp.sqrt(xp.mean(R * R, axis=0))
        names = ["continuity"] + ["x-momentum", "y-momentum", "z-momentum"][:self.dim] + ["energy"]
        vals = [float(v) for v in self.be.to_numpy(rms)]
        res = {}
        for nm, v in zip(names, vals):
            if self.iteration < 5 or nm not in self._res0:
                self._res0[nm] = max(self._res0.get(nm, 0.0), v, 1e-300)
            res[nm] = v / self._res0[nm]
        self._last_res = res
        rmax = max(res.values())
        if not math.isfinite(rmax):
            raise FloatingPointError("compressible solver diverged (non-finite residual)")
        if self.case.time != "transient" and self.iteration > 5 and rmax < self.case.tol:
            self.converged = True
        return rmax

    # ------------------------------------------------------------------ reporting
    @property
    def finished(self) -> bool:
        c = self.case
        if c.time == "transient":
            return self.time >= c.t_end * (1 - 1e-12) or self.iteration >= c.max_iter
        return self.converged or self.iteration >= c.max_iter

    @property
    def step_count(self) -> int:
        return self.iteration

    @property
    def t(self):
        return self.time if self.case.time == "transient" else self.iteration

    def residuals(self) -> dict:
        r = dict(self._last_res)
        return {"raw": r, "scaled": r, "normalized": r}

    def fields(self, names=None) -> Dict[str, np.ndarray]:
        h = self.be.to_numpy
        W = h(self.prim(self.Q)).astype(np.float64)
        d, g, R = self.dim, self.gamma, self.R
        out = {"density": W[:, 0], "x-velocity": W[:, 1], "y-velocity": W[:, 2]}
        if d == 3:
            out["z-velocity"] = W[:, 3]
        out["pressure"] = W[:, -1]
        out["temperature"] = W[:, -1] / (W[:, 0] * R)
        vm = np.linalg.norm(W[:, 1:1 + d], axis=1)
        out["velocity-magnitude"] = vm
        out["mach-number"] = vm / np.sqrt(g * W[:, -1] / W[:, 0])
        if names:
            out = {k: v for k, v in out.items() if k in names}
        return out

    def cell_centres(self) -> np.ndarray:
        return self.fv.geometry.cell_centre

    def mass_balance(self) -> dict:
        """Mass flow out of every boundary zone (kg/s per radian in axisymmetric cases)."""
        fv = self.fv
        W = self.prim(self.Q)
        WP = W[fv.Pb]
        Fb = self._hllc(WP, self._ghost(WP), self.n_b)[:, 0] * self.A_b
        Fh = self.be.to_numpy(Fb)
        out = {}
        for z in self.mesh.boundary_zones():
            m = fv.bzone == z.id
            if m.any():
                out[z.name] = float(Fh[m].sum())
        return out

    def describe(self) -> dict:
        c = self.case
        return {"solver": "density-based " + ("Navier-Stokes" if c.viscous else "Euler")
                + (" axisymmetric" if c.axisymmetric else ""), "flux": "HLLC",
                "order": c.order, "limiter": c.limiter, "time": c.time, "backend": self.be.name,
                "device": self.be.device, "precision": str(self.dtype), "cells": self.nc, "dim": self.dim,
                "setup_seconds": round(self.setup_seconds, 3)}
