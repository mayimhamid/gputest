<<<<<<< HEAD
r"""RANS turbulence models for the pressure-based unstructured solver.

Models
------
``k-omega-sst``
    Menter's SST model in its 2003 form (Menter, Kuntz & Langtry, *Turbulence,
    Heat and Mass Transfer 4*, 2003), the form used by Fluent's default
    *SST k-omega* model and NASA's Turbulence Modeling Resource (TMR):

    .. math::
        \nabla\cdot(\rho U k) = \nabla\cdot((\mu + \sigma_k \mu_t)\nabla k) + \tilde P_k - \beta^* \rho k \omega

        \nabla\cdot(\rho U \omega) = \nabla\cdot((\mu + \sigma_\omega \mu_t)\nabla \omega) + \gamma \rho S^2
            - \beta \rho \omega^2 + 2 (1 - F_1) \rho \sigma_{\omega 2} \frac{\nabla k\cdot\nabla\omega}{\omega}

    with :math:`\mu_t = \rho a_1 k / \max(a_1 \omega, S F_2)` and
    :math:`\tilde P_k = \min(\mu_t S^2, 10 \beta^* \rho k \omega)`.

``k-epsilon-realizable``
    Shih et al. (*Computers & Fluids* 24, 1995) realizable k-epsilon with the
    variable :math:`C_\mu` used by Fluent (A0 = 4.04).

Wall treatment
--------------
* SST: y+-insensitive ("automatic"). The near-wall cell value of omega blends
  the viscous-sublayer and log-layer solutions,
  :math:`\omega_P = \sqrt{\omega_{vis}^2 + \omega_{log}^2}`, and the wall
  shear stress comes from Spalding's single-formula law of the wall, which is
  exactly the laminar stress for y+ < 5 and the log law for y+ > 30. The model
  is therefore accurate on meshes that resolve the sublayer (y+ ~ 1) and
  consistent on coarser ones.
* Realizable k-epsilon: scalable standard wall functions (Launder & Spalding
  with y* limited to 11.225, as Fluent's *Scalable Wall Functions*): wall
  shear from the log law, wall-cell production from the wall shear, and the
  near-wall epsilon fixed to its equilibrium value.

Boundary conditions
-------------------
Inlets (and backflow at outlets) use the turbulence intensity ``I`` and the
viscosity ratio ``mu_t/mu`` (Fluent's defaults 5 % and 10), or explicit k
and omega/epsilon values; outlets and symmetry planes are zero-gradient.
=======
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
>>>>>>> 3d528151e15014be7c185d9c110063ec4634bd8d
"""

from __future__ import annotations

<<<<<<< HEAD
import math
from typing import Dict, Optional

import numpy as np

KAPPA = 0.41
E_WALL = 9.793
B_SPALDING = 5.2
CMU = 0.09
YSTAR_LIM = 11.225
PR_T = 0.85  # turbulent Prandtl number (energy)

MODELS = ("laminar", "k-omega-sst", "k-epsilon-realizable", "k-epsilon-standard")


def spalding_uplus(re_y: np.ndarray, iters: int = 30) -> np.ndarray:
    """Solve Spalding's law for u+ given Re_y = U_t y / nu (vectorised Newton).

    Spalding (1961): y+ = u+ + e^{-kB} (e^{k u+} - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6),
    with y+ = Re_y / u+ for the unknown friction velocity.
    """
    re_y = np.maximum(np.asarray(re_y, dtype=float), 0.0)
    k, eb = KAPPA, math.exp(-KAPPA * B_SPALDING)
    # start from the laminar or the log-law estimate
    up = np.where(re_y < 120.0, np.sqrt(np.maximum(re_y, 1e-300)),
                  np.log(np.maximum(E_WALL * np.sqrt(np.maximum(re_y, 1.0)), 1.0)) / k + 1.0)
    for _ in range(iters):
        ku = np.minimum(k * up, 60.0)
        ex = np.exp(ku)
        g = up + eb * (ex - 1 - ku - ku ** 2 / 2 - ku ** 3 / 6)  # y+(u+)
        dg = 1 + eb * k * (ex - 1 - ku - ku ** 2 / 2)
        f = re_y / np.maximum(up, 1e-300) - g
        df = -re_y / np.maximum(up, 1e-300) ** 2 - dg
        step = f / df
        up = np.maximum(up - step, 0.5 * up)
        if np.all(np.abs(step) <= 1e-10 * np.maximum(up, 1e-12)):
            break
    return np.where(re_y > 0, up, 0.0)


def strain_invariants(xp, gU, dim):
    """|S| = sqrt(2 S_ij S_ij), |Omega| = sqrt(2 W_ij W_ij), sum S_ij S_ij, S_ij S_jk S_ki."""
    S2 = 0.0
    W2 = 0.0
    Sij = [[0.5 * (gU[i][:, j] + gU[j][:, i]) for j in range(dim)] for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            S2 = S2 + Sij[i][j] ** 2
            W2 = W2 + (0.5 * (gU[i][:, j] - gU[j][:, i])) ** 2
    S3 = 0.0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                S3 = S3 + Sij[i][j] * Sij[j][k] * Sij[k][i]
    return xp.sqrt(2.0 * S2), xp.sqrt(2.0 * W2), S2, W2, S3


class TurbulenceModel:
    """Shared machinery: wall distance, BCs, scalar transport, wall shear."""

    name = "base"
    second = "omega"  # name of the second transported variable
    default_relax = 0.8

    def __init__(self, solver):
        from .simple import INLET, OUTLET, PRESSURE, WALL

        self.s = s = solver
        fv, xp = s.fv, s.xp
        self.xp = xp
        c = s.case
        self.rho, self.mu = c.density, c.viscosity
        self.nu = self.mu / self.rho
        kind = s.bkind_h
        self.wall_b = kind == WALL
        self.inlet_b = kind == INLET
        self.open_b = (kind == OUTLET) | (kind == PRESSURE)
        h = s.be.to_numpy
        Pb = h(fv.Pb)
        nb = h(fv.nb).astype(np.float64)
        db = h(fv.db).astype(np.float64)
        # wall-normal distance of the owner cell of every wall face
        self.yb = np.abs(np.einsum("fd,fd->f", db, nb))
        self.y = self._wall_distance()
        nc = s.nc
        # wall-adjacent cells and their (smallest) wall distance
        wf = np.nonzero(self.wall_b)[0]
        self.wall_faces = wf
        self.wall_cells = np.unique(Pb[wf])
        ywc = np.full(nc, np.inf)
        np.minimum.at(ywc, Pb[wf], self.yb[wf])
        self.y_wall_cell = ywc
        self.is_wall_cell = np.isfinite(ywc)
        if self.wall_cells.size:
            self.y[self.wall_cells] = ywc[self.wall_cells]
        self.y_d = xp.asarray(self.y, dtype=s.dtype)
        # inlet / backflow values
        spd = np.linalg.norm(h(s.ub_fixed).astype(np.float64), axis=1)
        u_ref = float(spd[self.inlet_b].max()) if self.inlet_b.any() else float(c.turb_reference_velocity or 1.0)
        self.u_ref = max(u_ref, 1e-6)
        kin = np.zeros(s.fv.n_bf)
        sin = np.zeros(s.fv.n_bf)
        for z in s.mesh.boundary_zones():
            spec = s.zone_specs[z.name]
            m = h(fv.bzone) == z.id
            I = spec.turb_intensity if spec.turb_intensity is not None else c.turb_intensity
            r = spec.turb_viscosity_ratio if spec.turb_viscosity_ratio is not None else c.turb_viscosity_ratio
            U = np.where(spd[m] > 0, spd[m], self.u_ref)
            k = 1.5 * (I * U) ** 2 if spec.k is None else np.full(m.sum(), spec.k)
            kin[m] = np.maximum(k, 1e-12 * self.u_ref ** 2)
            sin[m] = self.second_from(kin[m], r, getattr(spec, self.second))
        self.kb_in = kin
        self.sb_in = sin
        # initial field
        init = c.turb_init or {}
        k0 = init.get("k", 1.5 * (c.turb_intensity * self.u_ref) ** 2)
        s0 = init.get(self.second, float(self.second_from(np.array([k0]), c.turb_viscosity_ratio, None)[0]))
        self.k = xp.full(nc, k0, dtype=s.dtype)
        self.w = xp.full(nc, s0, dtype=s.dtype)  # omega or epsilon
        self.mut = xp.zeros(nc, dtype=s.dtype)
        self.k_min = 1e-14 * self.u_ref ** 2
        self.w_min = 1e-14
        self.relax_k = c.relax_k if c.relax_k is not None else self.default_relax
        self.relax_w = c.relax_w if c.relax_w is not None else self.default_relax
        self._mu_w = None  # effective wall viscosity per boundary face
        self.yplus = np.zeros(s.fv.n_bf)
        self._update_mut(None)

    # ------------------------------------------------------------------
    def second_from(self, k, ratio, explicit):
        raise NotImplementedError

    def _wall_distance(self) -> np.ndarray:
        """Distance from every cell centre to the nearest wall face (exact for 2-D segments)."""
        s = self.s
        mesh, geo = s.mesh, s.fv.geometry
        xc = geo.cell_centre
        wall_zone_ids = [z.id for z in mesh.boundary_zones() if s.zone_specs[z.name].kind == "wall"]
        wf = np.nonzero(np.isin(mesh.face_zone, wall_zone_ids))[0]
        if wf.size == 0:
            return np.full(s.nc, 1e30)
        from scipy.spatial import cKDTree

        fc = geo.face_centre[wf]
        kq = min(8, wf.size)
        _, idx = cKDTree(fc).query(xc, k=kq)
        idx = np.atleast_2d(idx).reshape(xc.shape[0], kq)
        best = np.full(xc.shape[0], np.inf)
        if mesh.dim == 2:
            a = mesh.nodes[mesh.face_nodes[mesh.face_ptr[wf]]]
            b = mesh.nodes[mesh.face_nodes[mesh.face_ptr[wf] + 1]]
            for j in range(kq):
                f = idx[:, j]
                ab = b[f] - a[f]
                t = np.clip(np.einsum("cd,cd->c", xc - a[f], ab) / np.maximum(np.einsum("cd,cd->c", ab, ab), 1e-300), 0, 1)
                d = np.linalg.norm(xc - (a[f] + t[:, None] * ab), axis=1)
                best = np.minimum(best, d)
        else:
            n = geo.face_area_vec[wf]
            area = np.linalg.norm(n, axis=1)
            n = n / area[:, None]
            rad = np.sqrt(area / np.pi) * 1.2
            for j in range(kq):
                f = idx[:, j]
                v = xc - fc[f]
                dn = np.abs(np.einsum("cd,cd->c", v, n[f]))
                dt = np.linalg.norm(v - np.einsum("cd,cd->c", v, n[f])[:, None] * n[f], axis=1)
                d = np.where(dt <= rad[f], dn, np.sqrt(dn ** 2 + (dt - rad[f]) ** 2))
                best = np.minimum(best, d)
        return np.maximum(best, 1e-300)

    # ------------------------------------------------------------------
    def wall_shear_viscosity(self, U) -> np.ndarray:
        """Effective viscosity mu_w on every boundary face (walls only differ from mu)."""
        raise NotImplementedError

    def momentum_viscosity(self, U):
        """(interior-face, boundary-face) effective viscosity for the momentum equations."""
        s, xp = self.s, self.xp
        fv = s.fv
        mut_f = fv.face_interp(self.mut)
        mub = self.mu + self.mut[fv.Pb]
        mu_w = self.wall_shear_viscosity(U)
        mub = xp.where(xp.asarray(self.wall_b), xp.asarray(mu_w, dtype=s.dtype), mub)
        return self.mu + mut_f, mub

    def energy_diffusivity(self, k_over_cp: float):
        fv = self.s.fv
        mut_f = fv.face_interp(self.mut)
        gb = k_over_cp + self.xp.where(self.xp.asarray(self.wall_b), 0.0, self.mut[fv.Pb] / PR_T)
        return k_over_cp + mut_f / PR_T, gb

    def transpose_stress(self, gU):
        """Explicit div(mu_t (grad U)^T) per cell (the part not in the Laplacian)."""
        s, xp = self.s, self.xp
        fv, dim = s.fv, s.dim
        mut_f = fv.face_interp(self.mut)
        out = xp.zeros((s.nc, dim), dtype=s.dtype)
        for k in range(dim):
            gT = xp.stack([gU[j][:, k] for j in range(dim)], axis=1)  # d u_j / d x_k
            out[:, k] = fv.Cdiff_d @ (mut_f * xp.sum(fv.face_interp_vec(gT) * fv.S, axis=1))
        return out

    # ------------------------------------------------------------------
    def _bvals(self, phi, inflow_vals, wall_zero_grad=True):
        """Boundary values: fixed at inlets and outlet backflow, zero-gradient elsewhere."""
        s, xp = self.s, self.xp
        fv = s.fv
        fixed = xp.asarray(self.inlet_b) | (xp.asarray(self.open_b) & (s.Fb < 0))
        vals = xp.where(fixed, xp.asarray(inflow_vals, dtype=s.dtype), phi[fv.Pb])
        return vals, fixed.astype(s.dtype)

    def _transport(self, phi, inflow_vals, sigma_mut, Sc, Sp, relax, fixed_cells=None, fixed_vals=None,
                   floor=0.0):
        s, xp = self.s, self.xp
        fv = s.fv
        phib, fixed = self._bvals(phi, inflow_vals)
        grad = fv.gradient(phi, phib)
        gam_c = self.mu + sigma_mut * self.mut  # cell diffusivity
        gf = fv.face_interp(gam_c)
        gb = gam_c[fv.Pb]
        diag, aPN, aNP, Db = s._coefficients((gf, gb), fixed)
        rhs = s._rhs((gf, gb), phi, grad, phib, Db, Sc * fv.vol, scheme=s.case.turb_scheme)
        diag = diag + Sp * fv.vol
        if fixed_cells is not None and fixed_cells.size:
            big = 1e10 * float(xp.max(diag))
            fc = xp.asarray(fixed_cells)
            diag[fc] = diag[fc] + big
            rhs[fc] = rhs[fc] + big * xp.asarray(fixed_vals, dtype=s.dtype)
        A = fv.matrix(diag, aPN, aNP)
        # residual over the cells the equation actually controls: not the fixed near-wall
        # cells, and not cells held at the lower bound (e.g. free-stream k decaying to the
        # floor upstream of a flat plate), whose clipped residual can never vanish
        mask = phi > floor * (1.0 + 1e-9)
        if fixed_cells is not None and fixed_cells.size:
            mask[xp.asarray(fixed_cells)] = False
        r = xp.abs(rhs - A @ phi)[mask]
        den = xp.abs(diag * phi)[mask]
        res = float(xp.sum(r)) / max(float(xp.sum(den)), 1e-300)
        dr = diag / relax
        Ar = fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - relax) / relax) * diag * phi
        new, _ = s._solve(Ar, b, phi, dr, tol=1e-3, maxiter=60)
        new = xp.maximum(new, floor)
        return new, res

    def _gradient(self, phi, inflow_vals):
        b, _ = self._bvals(phi, inflow_vals)
        return self.s.fv.gradient(phi, b)

    # ------------------------------------------------------------------
    def fields(self) -> Dict[str, np.ndarray]:
        h = self.s.be.to_numpy
        out = {"turb-kinetic-energy": h(self.k).astype(np.float64),
               self.field_name: h(self.w).astype(np.float64),
               "viscosity-turb": h(self.mut).astype(np.float64),
               "viscosity-ratio": h(self.mut).astype(np.float64) / self.mu,
               "wall-distance": self.y.copy()}
        yp = np.zeros(self.s.nc)
        Pb = h(self.s.fv.Pb)
        if self.wall_faces.size:
            np.maximum.at(yp, Pb[self.wall_faces], self.yplus[self.wall_faces])
        out["y-plus"] = yp
        return out

    def state(self) -> dict:
        h = self.s.be.to_numpy
        return {"k": h(self.k), "w": h(self.w), "mut": h(self.mut)}

    def load(self, st: dict):
        xp, dt = self.xp, self.s.dtype
        if "k" in st:
            self.k = xp.asarray(st["k"], dtype=dt)
            self.w = xp.asarray(st["w"], dtype=dt)
            self.mut = xp.asarray(st["mut"], dtype=dt)

    def wall_shear(self) -> np.ndarray:
        """Wall shear stress magnitude on every boundary face (0 off walls)."""
        h = self.s.be.to_numpy
        fv = self.s.fv
        U = h(self.s.U).astype(np.float64)
        Pb = h(fv.Pb)
        nb = h(fv.nb).astype(np.float64)
        UP = U[Pb]
        ut = UP - np.einsum("fd,fd->f", UP, nb)[:, None] * nb
        ub = h(self.s.ub_fixed).astype(np.float64)
        ut = ut - (ub - np.einsum("fd,fd->f", ub, nb)[:, None] * nb)
        mag = np.linalg.norm(ut, axis=1)
        mu_w = self.wall_shear_viscosity(self.s.U)
        return np.where(self.wall_b, mu_w * mag / np.maximum(self.yb, 1e-300), 0.0)


class KOmegaSST(TurbulenceModel):
    name = "k-omega-sst"
    second = "omega"
    field_name = "specific-diss-rate"
    sk1, sw1, b1 = 0.85, 0.5, 0.075
    sk2, sw2, b2 = 1.0, 0.856, 0.0828
    bstar, a1 = 0.09, 0.31
    g1 = b1 / bstar - sw1 * KAPPA ** 2 / math.sqrt(bstar)
    g2 = b2 / bstar - sw2 * KAPPA ** 2 / math.sqrt(bstar)

    def second_from(self, k, ratio, explicit):
        if explicit is not None:
            return np.full(np.shape(k), float(explicit))
        return self.rho * np.asarray(k) / (self.mu * ratio)

    def _blend(self):
        xp = self.xp
        k, w, y = self.k, self.w, self.y_d
        gk = self._gradient(k, self.kb_in)
        gw = self._gradient(w, self.sb_in)
        kw = xp.sum(gk * gw, axis=1)
        cd = xp.maximum(2.0 * self.rho * self.sw2 * kw / w, 1e-10)
        sk = xp.sqrt(k)
        a = xp.maximum(sk / (self.bstar * w * y), 500.0 * self.nu / (y * y * w))
        arg1 = xp.minimum(a, 4.0 * self.rho * self.sw2 * k / (cd * y * y))
        F1 = xp.tanh(arg1 ** 4)
        arg2 = xp.maximum(2.0 * sk / (self.bstar * w * y), 500.0 * self.nu / (y * y * w))
        F2 = xp.tanh(arg2 ** 2)
        return F1, F2, kw

    def _update_mut(self, S):
        xp = self.xp
        if S is None:
            self.mut = self.rho * self.k / self.w
            return
        _, F2, _ = self._blend()
        self.mut = self.rho * self.a1 * self.k / xp.maximum(self.a1 * self.w, S * F2)

    def _omega_wall(self):
        """Near-wall cell omega: sqrt(omega_vis^2 + omega_log^2)."""
        h = self.s.be.to_numpy
        yc = self.y_wall_cell[self.wall_cells]
        k = h(self.k)[self.wall_cells].astype(np.float64)
        w_vis = 6.0 * self.nu / (self.b1 * yc ** 2)
        w_log = np.sqrt(np.maximum(k, 0.0)) / (CMU ** 0.25 * KAPPA * yc)
        return np.sqrt(w_vis ** 2 + w_log ** 2)

    def wall_shear_viscosity(self, U):
        s = self.s
        h = s.be.to_numpy
        fv = s.fv
        Uh = h(U).astype(np.float64)
        Pb = h(fv.Pb)
        nb = h(fv.nb).astype(np.float64)
        ub = h(s.ub_fixed).astype(np.float64)
        rel = Uh[Pb] - ub
        ut = np.linalg.norm(rel - np.einsum("fd,fd->f", rel, nb)[:, None] * nb, axis=1)
        re_y = ut * self.yb / self.nu
        up = spalding_uplus(re_y)
        ok = (re_y > 1e-12) & (up > 0)
        mu_w = np.full(re_y.shape, self.mu)
        mu_w[ok] = self.mu * re_y[ok] / up[ok] ** 2  # tau_w = mu_w u_t / y with u_tau = u_t / u+
        yp = np.zeros(re_y.shape)
        yp[ok] = re_y[ok] / up[ok]
        self.yplus = np.where(self.wall_b, yp, 0.0)
        return np.where(self.wall_b, mu_w, self.mu)

    def solve(self, gU) -> Dict[str, float]:
        s, xp = self.s, self.xp
        S, _, _, _, _ = strain_invariants(xp, gU, s.dim)
        S2 = S * S
        F1, F2, kw = self._blend()
        rho, k, w = self.rho, self.k, self.w
        bl = lambda a, b: F1 * a + (1.0 - F1) * b  # noqa: E731
        Pk = self.mut * S2
        Pk = xp.minimum(Pk, 10.0 * self.bstar * rho * k * w)
        knew, rk = self._transport(k, self.kb_in, bl(self.sk1, self.sk2), Pk, self.bstar * rho * w,
                                   self.relax_k, floor=self.k_min)
        cdt = (1.0 - F1) * 2.0 * rho * self.sw2 * kw / w
        Sc = bl(self.g1, self.g2) * rho * S2 + xp.maximum(cdt, 0.0)
        Sp = bl(self.b1, self.b2) * rho * w + xp.maximum(-cdt, 0.0) / w
        wnew, rw = self._transport(w, self.sb_in, bl(self.sw1, self.sw2), Sc, Sp, self.relax_w,
                                   fixed_cells=self.wall_cells, fixed_vals=self._omega_wall(), floor=self.w_min)
        self.k, self.w = knew, wnew
        self._update_mut(S)
        return {"k": rk, "omega": rw}


class KEpsilonRealizable(TurbulenceModel):
    name = "k-epsilon-realizable"
    second = "epsilon"
    field_name = "turb-diss-rate"
    default_relax = 0.5
    sk, se, C2, A0 = 1.0, 1.2, 1.9, 4.04

    def second_from(self, k, ratio, explicit):
        if explicit is not None:
            return np.full(np.shape(k), float(explicit))
        return self.rho * CMU * np.asarray(k) ** 2 / (self.mu * ratio)

    def _cmu(self, gU):
        xp = self.xp
        S, W, S2, W2, S3 = strain_invariants(xp, gU, self.s.dim)
        St = xp.sqrt(S2)
        Wr = xp.where(St > 0, S3 / xp.maximum(St ** 3, 1e-300), 0.0)
        phi = xp.arccos(xp.clip(math.sqrt(6.0) * Wr, -1.0, 1.0)) / 3.0
        As = math.sqrt(6.0) * xp.cos(phi)
        Ustar = xp.sqrt(S2 + W2)
        return 1.0 / (self.A0 + As * self.k * Ustar / self.w), S

    def _eps_sources(self, S, k, eps, Gk):
        """Realizable epsilon equation: production rho C1 S eps, destruction rho C2 eps^2/(k + sqrt(nu eps))."""
        xp, rho = self.xp, self.rho
        eta = S * k / eps
        C1 = xp.maximum(0.43, eta / (eta + 5.0))
        return rho * C1 * S * eps, rho * self.C2 * eps / (k + xp.sqrt(self.nu * eps))

    def _update_mut(self, gU):
        if gU is None:
            self.mut = self.rho * CMU * self.k ** 2 / self.w
            return
        cmu, _ = self._cmu(gU)
        r = self.s.case.relax_mut
        self.mut = r * (self.rho * cmu * self.k ** 2 / self.w) + (1.0 - r) * self.mut

    def _ustar(self):
        h = self.s.be.to_numpy
        Pb = h(self.s.fv.Pb)
        k = h(self.k).astype(np.float64)
        return CMU ** 0.25 * np.sqrt(np.maximum(k[Pb], 0.0))

    def wall_shear_viscosity(self, U):
        us = self._ustar()
        ystar = np.maximum(self.rho * us * self.yb / self.mu, YSTAR_LIM)
        mu_w = self.rho * KAPPA * us * self.yb / np.log(E_WALL * ystar)
        mu_w = np.maximum(mu_w, self.mu)
        self.yplus = np.where(self.wall_b, self.rho * us * self.yb / self.mu, 0.0)
        return np.where(self.wall_b, mu_w, self.mu)

    def solve(self, gU) -> Dict[str, float]:
        s, xp = self.s, self.xp
        h = s.be.to_numpy
        rho, k, eps = self.rho, self.k, self.w
        cmu, S = self._cmu(gU)
        Gk = self.mut * S * S
        # wall cells: production from the wall shear, epsilon from equilibrium
        if self.wall_cells.size:
            tau = self.wall_shear()
            Pb = h(s.fv.Pb)
            us = self._ustar()
            area = h(s.fv.Sb_mag).astype(np.float64)
            wf, wc = self.wall_faces, self.wall_cells
            # G_k = tau_w dU/dy|_P = tau_w^2 / (kappa rho u* y_P), area-averaged over the cell's wall faces
            prod = tau * tau / (KAPPA * rho * np.maximum(us, 1e-300) * np.maximum(self.yb, 1e-300))
            num = np.bincount(Pb[wf], (prod * area)[wf], minlength=s.nc)
            den = np.bincount(Pb[wf], area[wf], minlength=s.nc)
            Gk_h = h(Gk).astype(np.float64)
            Gk_h[wc] = num[wc] / np.maximum(den[wc], 1e-300)
            Gk = xp.asarray(Gk_h, dtype=s.dtype)
            kc = h(k).astype(np.float64)[wc]
            eps_w = CMU ** 0.75 * np.maximum(kc, 0.0) ** 1.5 / (KAPPA * self.y_wall_cell[wc])
        else:
            eps_w = np.zeros(0)
        knew, rk = self._transport(k, self.kb_in, 1.0 / self.sk, Gk, rho * eps / k, self.relax_k, floor=self.k_min)
        # epsilon sources with the freshly updated k: lagging k here lets the two
        # equations drift apart (eps/k blows up and k collapses to zero)
        Sc, Sp = self._eps_sources(S, knew, eps, Gk * (knew / k))
        enew, re = self._transport(eps, self.sb_in, 1.0 / self.se, Sc, Sp, self.relax_w,
                                   fixed_cells=self.wall_cells, fixed_vals=eps_w, floor=self.w_min)
        self.k, self.w = knew, enew
        self._update_mut(gU)
        return {"k": rk, "epsilon": re}


class KEpsilonStandard(KEpsilonRealizable):
    """Launder & Spalding (1974) standard k-epsilon (C_mu = 0.09, C1 = 1.44, C2 = 1.92)."""

    name = "k-epsilon-standard"
    sk, se, C1s, C2s = 1.0, 1.3, 1.44, 1.92

    def _cmu(self, gU):
        S, _, _, _, _ = strain_invariants(self.xp, gU, self.s.dim)
        return CMU, S

    def _eps_sources(self, S, k, eps, Gk):
        rho = self.rho
        return self.C1s * eps / k * Gk, rho * self.C2s * eps / k


def make_turbulence(solver) -> Optional[TurbulenceModel]:
    name = solver.case.turbulence
    if name in (None, "", "laminar"):
        return None
    if name == "k-omega-sst":
        return KOmegaSST(solver)
    if name == "k-epsilon-realizable":
        return KEpsilonRealizable(solver)
    if name == "k-epsilon-standard":
        return KEpsilonStandard(solver)
    raise ValueError(f"unknown turbulence model {name!r}; choose from {MODELS}")
=======
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
>>>>>>> 3d528151e15014be7c185d9c110063ec4634bd8d
