r"""Menter k-omega SST (2003) for the density-based solver.

Loosely coupled: after every mean-flow iteration, k and omega take one implicit step with
the same local time step, first-order upwind convection from the mean-flow mass fluxes,
central diffusion, and point-implicit destruction; the linear system is relaxed with Jacobi
sweeps, so every operation is a face or cell loop (NumPy or CuPy). Low-Reynolds wall
treatment (k = 0 at walls, omega = 60 nu / (beta1 y^2) in the wall cells), so the first cell
should sit at y+ ~ 1, as in Fluent's density-based SST without wall functions.

In axisymmetric cases the strain rate includes the hoop component v / r and all fluxes use
the radius-weighted face areas and volumes of the mean-flow solver.
"""

from __future__ import annotations

import math
import types

import numpy as np

from ..unstructured.turbulence import KAPPA, PR_T, TurbulenceModel

SK1, SW1, B1 = 0.85, 0.5, 0.075
SK2, SW2, B2 = 1.0, 0.856, 0.0828
BSTAR, A1 = 0.09, 0.31
G1 = B1 / BSTAR - SW1 * KAPPA ** 2 / math.sqrt(BSTAR)
G2 = B2 / BSTAR - SW2 * KAPPA ** 2 / math.sqrt(BSTAR)


class CompressibleSST:
    prandtl_t = PR_T

    def __init__(self, solver, sweeps: int = 6):
        s = self.s = solver
        xp = self.xp = s.xp
        self.sweeps = sweeps
        fv = s.fv
        self.y = xp.asarray(TurbulenceModel._wall_distance(types.SimpleNamespace(s=s)), dtype=s.dtype)
        # inlet / far-field turbulence from intensity and viscosity ratio of each zone
        nb = fv.n_bf
        kin, win = np.zeros(nb), np.zeros(nb)
        W0 = s.be.to_numpy(s.prim(s.Q))
        rho0 = float(W0[0, 0])
        mu0 = float(s._viscosity(np.array([W0[0, -1] / (W0[0, 0] * s.R)]))[0])
        for z in s.mesh.boundary_zones():
            m = fv.bzone == z.id
            spec = s.zone_specs[z.name]
            T = spec.T0 if spec.kind == "pressure-inlet" else spec.T
            c = math.sqrt(s.gamma * s.R * T)
            if spec.velocity is not None:
                U = float(np.linalg.norm(spec.velocity))
            elif spec.kind == "far-field":
                U = spec.mach * c
            else:
                U = max(float(np.linalg.norm(W0[:, 1:1 + s.dim], axis=1).max()), 0.1 * c)
            k = 1.5 * (spec.turb_intensity * U) ** 2
            kin[m] = k
            win[m] = rho0 * k / (mu0 * spec.turb_visc_ratio)
        self.k_in, self.w_in = xp.asarray(kin, dtype=s.dtype), xp.asarray(win, dtype=s.dtype)
        from .solver import WALL

        self.wall_b = s.bcode == WALL
        wb = s.be.to_numpy(self.wall_b)
        wcells = np.unique(s.be.to_numpy(fv.Pb)[wb])
        self.wall_cell = xp.asarray(np.isin(np.arange(s.nc), wcells))
        # initial field: the free-stream values of the first inflow zone
        inflow = wb == 0
        k0 = float(kin[inflow & (kin > 0)].max()) if (kin[inflow] > 0).any() else 1e-6
        w0 = float(win[inflow & (win > 0)].max()) if (win[inflow] > 0).any() else 1.0
        self.k = xp.full(s.nc, k0, dtype=s.dtype)
        self.w = xp.full(s.nc, w0, dtype=s.dtype)
        self.k_min, self.w_min = 1e-10 * k0 + 1e-14, 1e-6 * w0
        self.mut = xp.zeros(s.nc, dtype=s.dtype)
        self.F1 = xp.ones(s.nc, dtype=s.dtype)

    # ------------------------------------------------------------------
    def _strain(self, grad):
        """|S| = sqrt(2 S_ij S_ij) (and |Omega|) from the primitive gradients, incl. v / r."""
        s, xp, d = self.s, self.xp, self.s.dim
        gu = grad[:, 1:1 + d, :]
        Sij = 0.5 * (gu + xp.swapaxes(gu, 1, 2))
        div = xp.einsum("cii->c", gu)
        S2 = xp.sum(Sij * Sij, axis=(1, 2))
        if s.case.axisymmetric:
            r = xp.maximum(s.fv.xc[:, 1], 1e-30)
            hoop = s.W_last[:, 2] / r
            S2 = S2 + hoop ** 2
            div = div + hoop
        # production uses the deviatoric strain (compressible form of P_k = mu_t S^2 - 2/3 rho k div u)
        return xp.sqrt(2.0 * xp.maximum(S2, 0.0)), div

    def _face_values(self, phi, phi_in, wall_value):
        """Boundary-face values: inflow -> inlet value, wall -> wall_value (None = zero gradient)."""
        xp, s = self.xp, self.s
        inner = phi[s.fv.Pb]
        inflow = s.mdot_b < 0
        out = xp.where(inflow & ~self.wall_b, phi_in, inner)
        if wall_value is not None:
            out = xp.where(self.wall_b, wall_value, out)
        return out

    def update(self, W, grad, dt):
        """Advance k and omega one implicit step; returns their scaled RMS updates."""
        s, xp, fv = self.s, self.xp, self.s.fv
        rho = W[:, 0]
        T = W[:, -1] / (rho * s.R)
        mu = s._viscosity(T)
        nu = mu / rho
        k, w, y = self.k, self.w, self.y
        kb = self._face_values(k, self.k_in, 0.0)
        wb = self._face_values(w, self.w_in, None)
        gk = fv.gradient(k, kb)
        gw = fv.gradient(w, wb)
        kw = xp.sum(gk * gw, axis=1)
        cd = xp.maximum(2.0 * rho * SW2 * kw / w, 1e-10)
        sk = xp.sqrt(k)
        a = xp.maximum(sk / (BSTAR * w * y), 500.0 * nu / (y * y * w))
        F1 = xp.tanh(xp.minimum(a, 4.0 * rho * SW2 * k / (cd * y * y)) ** 4)
        F2 = xp.tanh(xp.maximum(2.0 * sk / (BSTAR * w * y), 500.0 * nu / (y * y * w)) ** 2)
        S, div = self._strain(grad)
        bl = lambda u, v: F1 * u + (1.0 - F1) * v  # noqa: E731
        Pk = self.mut * S * S - (2.0 / 3.0) * rho * k * div
        Pk = xp.clip(Pk, 0.0, 10.0 * BSTAR * rho * k * w)
        cdt = (1.0 - F1) * 2.0 * rho * SW2 * kw / w
        Pw = bl(G1, G2) * rho * S * S + xp.maximum(cdt, 0.0)
        k_new = self._step(k, kb, rho, mu + bl(SK1, SK2) * self.mut, Pk, BSTAR * rho * w, dt, wall_dirichlet=True)
        w_new = self._step(w, wb, rho, mu + bl(SW1, SW2) * self.mut, Pw,
                           bl(B1, B2) * rho * w + xp.maximum(-cdt, 0.0) / w, dt, wall_dirichlet=False)
        w_wall = 60.0 * nu / (B1 * y * y)
        w_new = xp.where(self.wall_cell, w_wall, w_new)
        k_new = xp.maximum(k_new, self.k_min)
        w_new = xp.maximum(w_new, self.w_min)
        rk = float(xp.sqrt(xp.mean((k_new - k) ** 2)) / (xp.max(k_new) + 1e-30))
        rw = float(xp.sqrt(xp.mean(((w_new - w) / w_new) ** 2)))
        self.k, self.w, self.F1 = k_new, w_new, F1
        self.mut = rho * A1 * k_new / xp.maximum(A1 * w_new, S * F2)
        return {"k": rk, "omega": rw}

    def _step(self, phi, phib, rho, gamma, Sc, Sp, dt, wall_dirichlet):
        """Implicit Euler for rho*phi: V rho (phi' - phi)/dt + conv + diff = V (Sc - Sp phi')."""
        s, xp, fv = self.s, self.xp, self.s.fv
        P, N, Pb = fv.P, fv.N, fv.Pb
        m, mb = s.mdot_i, s.mdot_b
        Ai, Ab = s.A_i, s.A_b
        g_f = 0.5 * (gamma[P] + gamma[N]) * Ai / s.d_len
        g_b = xp.where(self.wall_b if wall_dirichlet else xp.zeros_like(self.wall_b), gamma[Pb] * Ab / s.db_len, 0.0)
        g_b = g_b + xp.where(~self.wall_b & (mb < 0), gamma[Pb] * Ab / s.db_len, 0.0)
        mp, mn = xp.maximum(m, 0.0), xp.maximum(-m, 0.0)
        # explicit residual (outflow positive)
        flux = m * xp.where(m >= 0, phi[P], phi[N]) - g_f * (phi[N] - phi[P])
        fb = mb * xp.where(mb >= 0, phi[Pb], phib) - g_b * (phib - phi[Pb])
        R = fv.Cdiff_d @ flux + fv.Cb_d @ fb - s.V * (Sc - Sp * phi)
        # diagonal and off-diagonal (upwind + diffusion) coefficients of the delta form
        D = s.V * rho / dt + s.V * Sp + fv.Cown_d @ (mp + g_f) + fv.Cnei_d @ (mn + g_f) \
            + fv.Cb_d @ (xp.maximum(mb, 0.0) + g_b)
        aPN = mn + g_f  # coefficient of dphi_N in P's row (with a minus sign)
        aNP = mp + g_f
        dphi = -R / D
        for _ in range(self.sweeps):
            dphi = (-R + fv.Cown_d @ (aPN * dphi[N]) + fv.Cnei_d @ (aNP * dphi[P])) / D
        return phi + dphi

    def fields(self):
        h = self.s.be.to_numpy
        return {"turb-kinetic-energy": h(self.k).astype(np.float64),
                "specific-diss-rate": h(self.w).astype(np.float64),
                "viscosity-turb": h(self.mut).astype(np.float64)}
