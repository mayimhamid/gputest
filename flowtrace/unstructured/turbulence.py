r"""Menter k-omega SST turbulence model (2003 form, as in Fluent / NASA TMR "SST-2003").

    d(rho k)/dt + div(rho u k) = P~k - beta* rho k w + div((mu + sigma_k mu_t) grad k)
    d(rho w)/dt + div(rho u w) = gamma rho/mu_t P~k - beta rho w^2 + div((mu + sigma_w mu_t) grad w)
                                 + 2 (1 - F1) rho sigma_w2 / w  grad k . grad w
    mu_t = rho a1 k / max(a1 w, S F2),   P~k = min(mu_t S^2, 10 beta* rho k w)

Near-wall treatment is y+-insensitive (in the spirit of Fluent's enhanced /
OpenFOAM's automatic wall treatment):

* wall shear from Spalding's single-formula law of the wall, which reduces to
  the viscous sublayer for y+ < 5 and to the log law beyond;
* omega fixed in wall-adjacent cells to sqrt(w_vis^2 + w_log^2) with
  w_vis = 6 nu / (beta1 y^2), w_log = u_tau / (sqrt(beta*) kappa y);
* k: Dirichlet k = 0 at walls with y+ < 11.25 (resolved), zero flux and
  log-layer production in the first cell otherwise.
"""

from __future__ import annotations

import numpy as np

SIGMA_K1, SIGMA_W1, BETA1 = 0.85, 0.5, 0.075
SIGMA_K2, SIGMA_W2, BETA2 = 1.0, 0.856, 0.0828
BETA_STAR, A1, KAPPA, B_LOG, C1 = 0.09, 0.31, 0.41, 5.2, 10.0
GAMMA1 = BETA1 / BETA_STAR - SIGMA_W1 * KAPPA ** 2 / np.sqrt(BETA_STAR)
GAMMA2 = BETA2 / BETA_STAR - SIGMA_W2 * KAPPA ** 2 / np.sqrt(BETA_STAR)
YPLUS_LOG = 11.25


def spalding_utau(xp, ut, y, nu, iters: int = 60):
    """Friction velocity from Spalding's law, vectorised bisection in log space."""
    ut = xp.maximum(ut, 1e-300)
    u_lam = xp.sqrt(nu * ut / y)
    lo = xp.log(xp.maximum(u_lam * 1e-4, 1e-300))
    hi = xp.log(10.0 * xp.maximum(ut, u_lam))
    E = np.exp(-KAPPA * B_LOG)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        utau = xp.exp(mid)
        up = ut / utau
        ku = xp.minimum(KAPPA * up, 300.0)
        g = up + E * (xp.exp(ku) - 1.0 - ku - 0.5 * ku ** 2 - ku ** 3 / 6.0)
        f = y * utau / nu - g  # increasing in utau
        neg = f < 0
        lo = xp.where(neg, mid, lo)
        hi = xp.where(neg, hi, mid)
    return xp.exp(0.5 * (lo + hi))


class SSTModel:
    def __init__(self, solver):
        s = self.s = solver
        fv, xp, c = s.fv, s.xp, s.case
        self.xp = xp
        rho, mu = c.density, c.viscosity
        wall_h = s.bkind_h == s.WALL
        y = np.maximum(fv.wall_distance(wall_h), 1e-12)
        self.y = xp.asarray(np.minimum(y, 1e6), dtype=s.dtype)
        self.has_walls = bool(wall_h.any())
        self.wf = xp.asarray(np.nonzero(wall_h)[0])
        self.wc = fv.Pb[self.wf]
        self.yw = fv.yb[self.wf]
        # inlet / backflow turbulence per boundary face
        nb = s.fv.n_bf
        kb = np.zeros(nb)
        wb = np.zeros(nb)
        ub = s.be.to_numpy(s.ub_fixed)
        speed = np.linalg.norm(ub, axis=1)
        for z in s.mesh.boundary_zones():
            m = s.fv.bzone == z.id
            if not m.any():
                continue
            spec = s.zone_specs.get(z.name)
            I = spec.turb_intensity if spec is not None and spec.turb_intensity is not None else c.turb_intensity
            R = spec.turb_viscosity_ratio if spec is not None and spec.turb_viscosity_ratio is not None \
                else c.turb_viscosity_ratio
            U = np.where(speed[m] > 0, speed[m], c.ref_velocity)
            kk = 1.5 * (I * U) ** 2
            kb[m] = kk
            wb[m] = rho * kk / (mu * R)
        self.k_in = xp.asarray(np.maximum(kb, 1e-14), dtype=s.dtype)
        self.w_in = xp.asarray(np.maximum(wb, 1e-10), dtype=s.dtype)
        k0 = 1.5 * (c.turb_intensity * c.ref_velocity) ** 2
        w0 = rho * k0 / (mu * c.turb_viscosity_ratio)
        self.k = xp.full(s.nc, k0, dtype=s.dtype)
        self.om = xp.full(s.nc, w0, dtype=s.dtype)
        self.mut = rho * self.k / self.om
        self.F1 = xp.ones(s.nc, dtype=s.dtype)
        self.utau = xp.zeros(int(self.wf.shape[0]), dtype=s.dtype)
        self.yplus = xp.zeros(int(self.wf.shape[0]), dtype=s.dtype)
        self.k_min, self.w_min = 1e-14, 1e-10 * w0 + 1e-12

    # ------------------------------------------------------------------
    def wall_viscosity(self, U):
        """Effective wall viscosity on every boundary face (mu elsewhere) from Spalding's law."""
        s, fv, xp, c = self.s, self.s.fv, self.xp, self.s.case
        mu, rho = c.viscosity, c.density
        mub = xp.full(fv.n_bf, mu, dtype=s.dtype)
        if not self.has_walls:
            return mub
        nb = fv.nb[self.wf]
        rel = U[self.wc] - s.ub_fixed[self.wf]
        rel_t = rel - xp.sum(rel * nb, axis=1)[:, None] * nb
        ut = xp.sqrt(xp.sum(rel_t * rel_t, axis=1))
        utau = spalding_utau(xp, ut, self.yw, mu / rho)
        self.utau = utau
        self.yplus = self.yw * utau * rho / mu
        muw = xp.where(ut > 1e-30, rho * utau ** 2 * self.yw / xp.maximum(ut, 1e-30), mu)
        mub[self.wf] = xp.maximum(muw, mu)
        return mub

    def face_mut(self):
        return self.s.fv.face_interp(self.mut)

    # ------------------------------------------------------------------
    def _bvals(self, phi, inflow_val):
        """Boundary face values: inlet value at inlets/backflow, zero-gradient elsewhere."""
        s, xp = self.s, self.xp
        P = phi[s.fv.Pb]
        use_in = (s.m_inlet > 0) | ((s.m_pres > 0) & (s.Fb < 0))
        return xp.where(use_in, inflow_val, P)

    def step(self, gU):
        s, fv, xp, c = self.s, self.s.fv, self.xp, self.s.case
        rho, mu = c.density, c.viscosity
        nu = mu / rho
        V = fv.vol
        k, om, y = self.k, self.om, self.y
        dim = s.dim
        # strain-rate magnitude S = sqrt(2 Sij Sij)
        S2 = xp.zeros_like(k)
        for i in range(dim):
            for j in range(dim):
                sij = 0.5 * (gU[i][:, j] + gU[j][:, i])
                S2 = S2 + 2.0 * sij * sij
        S = xp.sqrt(S2)
        kb = self._bvals(k, self.k_in)
        wb = self._bvals(om, self.w_in)
        if self.has_walls:
            kb[self.wf] = xp.where(self.yplus < YPLUS_LOG, 0.0, k[self.wc])
        gk = fv.gradient(k, kb)
        gw = fv.gradient(om, wb)
        kgw = xp.sum(gk * gw, axis=1)
        sk_ = xp.sqrt(xp.maximum(k, 0.0))
        CD = xp.maximum(2.0 * rho * SIGMA_W2 * kgw / om, 1e-10)
        arg1 = xp.minimum(xp.maximum(sk_ / (BETA_STAR * om * y), 500.0 * nu / (y * y * om)),
                          4.0 * rho * SIGMA_W2 * k / (CD * y * y))
        F1 = xp.tanh(arg1 ** 4)
        arg2 = xp.maximum(2.0 * sk_ / (BETA_STAR * om * y), 500.0 * nu / (y * y * om))
        F2 = xp.tanh(arg2 ** 2)
        self.F1 = F1
        blend = lambda a, b: F1 * a + (1.0 - F1) * b  # noqa: E731
        sig_k, sig_w = blend(SIGMA_K1, SIGMA_K2), blend(SIGMA_W1, SIGMA_W2)
        beta, gamma = blend(BETA1, BETA2), blend(GAMMA1, GAMMA2)
        mut = self.mut
        Pk = xp.minimum(mut * S2, 10.0 * BETA_STAR * rho * k * om)

        # wall treatment
        res = {}
        Pb = fv.Pb
        dir_k = s.m_inlet.copy()
        wall_cells_log = None
        if self.has_walls:
            low = self.yplus < YPLUS_LOG
            dir_k[self.wf] = low.astype(s.dtype)
            hi_f = ~low
            if bool(hi_f.any()):
                cells = self.wc[hi_f]
                tau = rho * self.utau[hi_f] ** 2
                Pk_log = tau * (BETA_STAR ** 0.25) * xp.sqrt(xp.maximum(k[cells], 0.0)) / (KAPPA * self.yw[hi_f])
                Pk[cells] = Pk_log
                wall_cells_log = cells

        # ---------------- k ----------------
        gam_c = mu + sig_k * mut
        gam_f = fv.face_interp(gam_c)
        gam_b = mu + (sig_k * mut)[Pb] * (1.0 - s.m_wall)
        diag, aPN, aNP, Db = s._coefficients(gam_f, dir_k, gamma_b=gam_b)
        so = c.turb_scheme == "second-order"
        rhs = s._rhs(gam_f, k, gk, kb, Db, Pk * V, second_order=so)
        diag = diag + BETA_STAR * rho * om * V
        diag, rhs = s._time_terms("k", k, diag, rhs)
        A = fv.matrix(diag, aPN, aNP)
        res["k"] = s._residual(A, diag, k, rhs)
        k_new = s._relax_solve(diag, aPN, aNP, rhs, k, c.relax_turb)
        k_new = xp.maximum(k_new, self.k_min)

        # ---------------- omega ----------------
        prod_w = gamma * rho * xp.minimum(S2, (C1 / A1) * BETA_STAR * om * xp.maximum(A1 * om, F2 * S))
        cdk = (1.0 - F1) * 2.0 * rho * SIGMA_W2 * kgw / om
        gam_c = mu + sig_w * mut
        gam_f = fv.face_interp(gam_c)
        gam_b = mu + (sig_w * mut)[Pb] * (1.0 - s.m_wall)
        dir_w = s.m_inlet.copy()
        diag, aPN, aNP, Db = s._coefficients(gam_f, dir_w, gamma_b=gam_b)
        rhs = s._rhs(gam_f, om, gw, wb, Db, (prod_w + xp.maximum(cdk, 0.0)) * V, second_order=so)
        diag = diag + (beta * rho * om + xp.maximum(-cdk, 0.0) / om) * V
        diag, rhs = s._time_terms("omega", om, diag, rhs)
        A = fv.matrix(diag, aPN, aNP)
        if self.has_walls:
            free = xp.ones(s.nc, dtype=s.dtype)
            free[self.wc] = 0.0  # omega is imposed there; exclude from the residual
            res["omega"] = s._residual(A, diag, om, rhs, mask=free)
        else:
            res["omega"] = s._residual(A, diag, om, rhs)
        if self.has_walls:
            yc = y[self.wc]
            w_vis = 6.0 * nu / (BETA1 * yc * yc)
            w_log = self.utau / (np.sqrt(BETA_STAR) * KAPPA * yc)
            w_wall = xp.sqrt(w_vis ** 2 + w_log ** 2)
            big = 1e10 * float(xp.max(diag))
            diag = diag.copy()
            rhs = rhs.copy()
            diag[self.wc] = big
            rhs[self.wc] = big * w_wall
        om_new = s._relax_solve(diag, aPN, aNP, rhs, om, c.relax_turb)
        om_new = xp.maximum(om_new, self.w_min)

        self.k, self.om = k_new, om_new
        sk_ = xp.sqrt(k_new)
        arg2 = xp.maximum(2.0 * sk_ / (BETA_STAR * om_new * y), 500.0 * nu / (y * y * om_new))
        F2 = xp.tanh(arg2 ** 2)
        self.mut = rho * A1 * k_new / xp.maximum(A1 * om_new, S * F2)
        self._wall_cells_log = wall_cells_log
        return res

    def fields(self):
        h = self.s.be.to_numpy
        return {"turb-kinetic-energy": h(self.k).astype(np.float64),
                "specific-diss-rate": h(self.om).astype(np.float64),
                "viscosity-ratio": h(self.mut).astype(np.float64) / self.s.case.viscosity,
                "wall-distance": h(self.y).astype(np.float64)}
