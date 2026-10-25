r"""P-1 radiation model (Fluent: Models > Radiation > P1).

Incident radiation G obeys

    -div(Gamma grad G) + a G = 4 a sigma T^4,      Gamma = 1 / (3 (a + sigma_s))

and feeds the energy equation with -div(q_r) = a G - 4 a sigma T^4 (linearised
in T for stability). Walls use the Marshak condition

    -Gamma dG/dn = eps / (2 (2 - eps)) (4 sigma T_w^4 - G_w),

which, eliminated together with the near-wall gradient, is a conductance
in series with the cell: q = A B / (A + B) (G_P - 4 sigma T_w^4). Inlets,
outlets and openings radiate as black surfaces at their temperature.
Temperatures must be absolute (K).
"""

from __future__ import annotations

import numpy as np

SIGMA_SB = 5.670374419e-8


class P1Model:
    def __init__(self, solver):
        s = self.s = solver
        xp, c = s.xp, s.case
        self.xp = xp
        self.gamma = 1.0 / (3.0 * (c.absorption + c.scattering))
        eps = np.ones(s.fv.n_bf)
        rad = np.zeros(s.fv.n_bf, dtype=bool)
        for z in s.mesh.boundary_zones():
            m = s.fv.bzone == z.id
            spec = s.zone_specs.get(z.name)
            if spec is None or spec.kind in ("symmetry", "periodic"):
                continue
            rad[m] = True
            eps[m] = spec.emissivity if spec.kind == "wall" else 1.0
        eps = np.clip(eps, 1e-6, 1.0)
        self.eps = xp.asarray(eps, dtype=s.dtype)
        self.rad = xp.asarray(rad.astype(float), dtype=s.dtype)
        T = s.T
        self.G = 4.0 * SIGMA_SB * T ** 4

    def _wall_conductance(self):
        s, fv = self.s, self.s.fv
        A = self.gamma * fv.alpha_b  # cell-to-face diffusion conductance (times area)
        B = self.eps / (2.0 * (2.0 - self.eps)) * fv.Sb_mag
        return self.rad * A * B / (A + B)

    def solve(self, T, Tb):
        """Update G from the current temperature field; returns the scaled residual."""
        s, fv, xp, c = self.s, self.s.fv, self.xp, self.s.case
        a = c.absorption
        V = fv.vol
        D = self.gamma * fv.alpha
        cb = self._wall_conductance()
        diag = fv.scatter(D, D, cb) + a * V
        emit = 4.0 * SIGMA_SB * T ** 4
        Gb = self.G[fv.Pb]
        gG = fv.gradient(self.G, Gb)
        gf = fv.face_interp_vec(gG)
        rhs = (fv.Cb_d @ (cb * 4.0 * SIGMA_SB * Tb ** 4) + a * emit * V
               + fv.Cdiff_d @ (self.gamma * xp.sum(gf * fv.Tvec, axis=1)))
        A = fv.matrix(diag, -D, -D)
        r = s._residual(A, diag, self.G, rhs)
        inv = 1.0 / diag
        from ..linalg.krylov import bicgstab

        self.G, _ = bicgstab(lambda v: A @ v, rhs, xp, M=lambda v: inv * v, x0=self.G, tol=1e-6, maxiter=200,
                             relative_to_initial=True)
        return r

    def energy_terms(self, T):
        """(extra diagonal, extra rhs) for the energy equation in temperature form (divided by cp)."""
        c, V = self.s.case, self.s.fv.vol
        a = c.absorption
        T3 = T ** 3
        diag = 16.0 * a * SIGMA_SB * T3 * V / c.cp
        rhs = (a * self.G + 12.0 * a * SIGMA_SB * T3 * T) * V / c.cp
        return diag, rhs

    def wall_flux(self, Tb):
        """Radiative heat flux into the domain on every boundary face (W)."""
        cb = self._wall_conductance()
        return cb * (4.0 * SIGMA_SB * Tb ** 4 - self.G[self.s.fv.Pb])
