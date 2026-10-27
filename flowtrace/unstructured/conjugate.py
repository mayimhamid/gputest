"""Conjugate heat transfer: flow in the fluid zones, conduction in solid zones, one energy equation.

Like Fluent's coupled walls: the momentum/pressure (and turbulence) equations
are solved on the fluid cells only, where the fluid-solid interfaces act as
no-slip walls; the energy equation is solved on *all* cells, with face
conductivities from the harmonic mean of the two cells (heat flux continuous
across the interface, exact for 1-D composite walls) and the convective
mass flux of the fluid. Solid zones carry their own conductivity and an
optional volumetric heat source.

``case.solids = {"heater": {"conductivity": 16.0, "source": 1e5}}``
"""

from __future__ import annotations

import copy
import time
from typing import Dict

import numpy as np

from ..hardware.backend import get_backend
from ..mesh.unstructured import UnstructuredMesh, extract_cells
from .case import BoundarySpec, UnstructuredCase
from .fvmesh import FVMesh
from .simple import T_FIXED, T_FLUX, SimpleSolver


class _EnergyOps:
    """Minimal solver-shaped object so the SIMPLEC discretisation routines run on the full mesh."""

    _split_gamma = staticmethod(SimpleSolver._split_gamma)
    _coefficients = SimpleSolver._coefficients
    _rhs = SimpleSolver._rhs
    _residual = SimpleSolver._residual
    _solve = SimpleSolver._solve

    def __init__(self, fv, xp, case):
        self.fv, self.xp, self.case = fv, xp, case
        self.F = self.Fb = None

    def _conv(self):
        return self.F, self.Fb


class ConjugateSolver:
    def __init__(self, mesh: UnstructuredMesh, case: UnstructuredCase, backend=None):
        if not case.energy:
            raise ValueError("conjugate heat transfer needs the energy equation (case.energy = True)")
        self.case, self.mesh = case, mesh
        self.be = backend if backend is not None else get_backend(case.backend)
        self.xp = xp = self.be.xp
        t0 = time.perf_counter()
        solid_ids = {}
        for name, spec in case.solids.items():
            z = mesh.zone_by_name(name)
            if z is None or z.kind != "cell":
                raise ValueError(f"solid zone {name!r} is not a cell zone of this mesh")
            solid_ids[z.id] = spec
        for z in mesh.cell_zones():
            if z.type == "solid" and z.id not in solid_ids:
                raise ValueError(f"cell zone {z.name!r} is a solid: give its conductivity in case.solids")
        solid = np.isin(mesh.cell_zone, list(solid_ids))
        if solid.all():
            raise ValueError("no fluid cells")
        sub, fmap, cmap = extract_cells(mesh, ~solid)
        self.cell_map = cmap
        fcase = copy.deepcopy(case)
        fcase.solids = {}
        for z in sub.boundary_zones():
            if z.name.startswith("interface-"):
                fcase.boundaries[z.name] = BoundarySpec("wall")
        self.flow = SimpleSolver(sub, fcase, backend=self.be)
        self.flow.external_energy = True
        self.fv = fv = FVMesh(mesh, self.be, self.flow.dtype)
        self.dtype = self.flow.dtype
        self.nc, self.dim = mesh.n_cells, mesh.dim
        self.energy = True
        self.solid = solid
        # conductivity / cp per cell (steady energy equation written for T with the fluid cp)
        gam = np.full(mesh.n_cells, case.conductivity / case.cp)
        q = np.zeros(mesh.n_cells)
        for zid, spec in solid_ids.items():
            m = mesh.cell_zone == zid
            gam[m] = float(spec["conductivity"]) / case.cp
            q[m] = float(spec.get("source", 0.0)) / case.cp
        self.gam_cell = gam
        self.q_cell = xp.asarray(q * fv.geometry.cell_volume, dtype=self.dtype)
        # map sub-mesh faces to full-mesh interior/boundary positions
        pos_if = -np.ones(mesh.n_faces, dtype=np.int64)
        pos_if[fv.if_idx] = np.arange(fv.n_if)
        pos_bf = -np.ones(mesh.n_faces, dtype=np.int64)
        pos_bf[fv.bf_idx] = np.arange(fv.n_bf)
        sfv = self.flow.fv
        full_of_sub = np.where(fmap >= 0, fmap, -fmap - 1)
        self._if_src = sfv.if_idx  # sub-mesh interior faces -> full interior positions
        self._if_dst = pos_if[full_of_sub[sfv.if_idx]]
        bsub = sfv.bf_idx
        keep = pos_bf[full_of_sub[bsub]] >= 0  # sub boundary faces that are full boundary faces
        self._bf_src = np.nonzero(keep)[0]
        self._bf_dst = pos_bf[full_of_sub[bsub[keep]]]
        self._build_thermal_bcs()
        self.ops = _EnergyOps(fv, xp, case)
        self.T = xp.full(self.nc, case.init_T, dtype=self.dtype)
        self._gT = None
        self.setup_seconds = time.perf_counter() - t0
        self.timers = self.flow.timers
        self.last_linear = {}
        self._res_energy = 1.0

    # ------------------------------------------------------------------
    def _build_thermal_bcs(self):
        from .case import FLUENT_TO_KIND

        fv, xp = self.fv, self.xp
        n = fv.n_bf
        tk = np.zeros(n, dtype=np.int64)
        tv = np.zeros(n)
        out = np.zeros(n, dtype=bool)
        self.zone_specs = {}
        for z in self.mesh.boundary_zones():
            spec = self.case.boundaries.get(z.name) or BoundarySpec(kind=FLUENT_TO_KIND.get(z.type, "wall"))
            self.zone_specs[z.name] = spec
            m = fv.bzone == z.id
            if spec.kind == "wall":
                tk[m] = {"adiabatic": 0, "temperature": T_FIXED, "flux": T_FLUX}[spec.thermal]
            elif spec.kind in ("inlet", "pressure"):
                tk[m] = T_FIXED
            out[m] = spec.kind == "outlet"
            tv[m] = spec.value
        self.tkind = xp.asarray(tk)
        self.tval = xp.asarray(tv, dtype=self.dtype)
        self.is_outlet = xp.asarray(out)
        self.tkind_h, self.tval_h = tk, tv

    def _full_fluxes(self):
        xp, fv = self.xp, self.fv
        F = xp.zeros(fv.n_if, dtype=self.dtype)
        Fb = xp.zeros(fv.n_bf, dtype=self.dtype)
        sF, sFb = self.flow._conv()
        sub_if_pos = np.arange(sF.shape[0])
        F[xp.asarray(self._if_dst)] = sF[xp.asarray(sub_if_pos)]
        Fb[xp.asarray(self._bf_dst)] = sFb[xp.asarray(self._bf_src)]
        return F, Fb

    def _face_gamma(self):
        """Harmonic interpolation of cell diffusivities (+ turbulent part in the fluid)."""
        xp, fv = self.xp, self.fv
        g = self.gam_cell.copy()
        turb = self.flow.turb
        if turb is not None:
            from .turbulence import PR_T

            g[self.cell_map] += self.be.to_numpy(turb.mut).astype(np.float64) / PR_T
        gd = xp.asarray(g, dtype=self.dtype)
        gP, gN = gd[fv.P], gd[fv.N]
        w = fv.w  # owner interpolation weight: the owner side holds (1 - w) of d
        gf = 1.0 / ((1.0 - w) / gP + w / gN)
        return gf, gd[fv.Pb]

    def _energy(self) -> float:
        xp, fv, ops = self.xp, self.fv, self.ops
        ops.F, ops.Fb = self._full_fluxes()
        T = self.T
        TP = T[fv.Pb]
        if self._gT is not None:
            TP = TP + xp.sum(self._gT[fv.Pb] * fv.db_t, axis=1)
        fixed = self.tkind == T_FIXED
        flux = self.tkind == T_FLUX
        Tb = xp.where(fixed, self.tval, TP)
        Tb = xp.where(self.is_outlet & (ops.Fb < 0), self.tval, Tb)
        gT = fv.gradient(T, Tb)
        self._gT = gT
        gam = self._face_gamma()
        q_in = fv.Cb_d @ (flux * self.tval * fv.Sb_mag / self.case.cp) + self.q_cell
        diag, aPN, aNP, Db = ops._coefficients(gam, fixed.astype(self.dtype))
        rhs = ops._rhs(gam, T, gT, Tb, Db, q_in)
        A = fv.matrix(diag, aPN, aNP)
        r = ops._residual(A, diag, T, rhs)
        aT = self.case.relax_T
        dr = diag / aT
        Ar = fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - aT) / aT) * diag * T
        self.T, _ = ops._solve(Ar, b, T, dr, tol=1e-4, maxiter=200)
        self.flow.T = self.T[xp.asarray(self.cell_map)]
        return r

    # ------------------------------------------------------------------ solver API
    def step(self) -> float:
        if self.finished:
            return 0.0
        self.flow.step()
        t0 = time.perf_counter()
        self._res_energy = self._energy()
        self.flow.timers["energy (conjugate)"] += time.perf_counter() - t0
        self.last_linear = self.flow.last_linear
        if self.flow.iteration > 5 and self._res_energy >= self.case.tol:
            self.flow.converged = False
        return 1.0

    @property
    def iteration(self):
        return self.flow.iteration

    @property
    def step_count(self):
        return self.flow.iteration

    @property
    def converged(self):
        return self.flow.converged and self._res_energy < self.case.tol

    @property
    def finished(self):
        return self.converged or self.flow.iteration >= self.case.max_iter

    def residuals(self) -> dict:
        r = self.flow.residuals()
        for k in r:
            r[k] = dict(r[k], energy=self._res_energy)
        return r

    def diagnostics(self) -> dict:
        """Diagnostics of the flow solver's latest iteration on the fluid sub-mesh (cell
        indices refer to that sub-mesh). The conjugate energy equation is solved here, not
        by the flow solver; its residual is in :meth:`residuals`."""
        d = self.flow.diagnostics()
        if d.get("available"):
            d["scope"] = "fluid cells (flow sub-mesh); conjugate energy reported by residuals()"
        return d

    def cell_centres(self) -> np.ndarray:
        return self.fv.geometry.cell_centre

    def fields(self, names=None) -> Dict[str, np.ndarray]:
        f = self.flow.fields()
        out = {}
        for k, v in f.items():
            a = np.zeros(self.nc)
            a[self.cell_map] = v
            out[k] = a
        out["temperature"] = self.be.to_numpy(self.T).astype(np.float64)
        if names:
            out = {k: v for k, v in out.items() if k in names}
        return out

    def mass_balance(self) -> dict:
        return self.flow.mass_balance()

    def wall_forces(self) -> dict:
        """Fluid forces on the walls (fluid-solid interfaces appear as ``interface-*`` zones)."""
        return self.flow.wall_forces()

    def heat_flow(self) -> dict:
        """Heat flow into the domain per full-mesh boundary zone, plus the heat crossing
        every fluid-solid interface (positive from solid into fluid)."""
        h = self.be.to_numpy
        fv, c = self.fv, self.case
        T = h(self.T).astype(np.float64)
        Pb = h(fv.Pb)
        TP = T[Pb]
        F, Fb = self._full_fluxes()
        Fb = h(Fb).astype(np.float64)
        tk, tv = self.tkind_h, self.tval_h
        out_m = h(self.is_outlet)
        Tb = np.where(tk == T_FIXED, tv, TP)
        Tb = np.where(out_m & (Fb < 0), tv, Tb)
        alpha_b = h(fv.alpha_b).astype(np.float64)
        g = self.gam_cell[Pb] * c.cp
        cond = g * alpha_b * (Tb - TP) * (tk == T_FIXED) + np.where(tk == T_FLUX, tv * h(fv.Sb_mag), 0.0)
        q = cond - c.cp * Fb * Tb
        res = {z.name: float(q[fv.bzone == z.id].sum()) for z in self.mesh.boundary_zones()}
        # interface conduction
        gf, _ = self._face_gamma()
        gf = h(gf).astype(np.float64) * c.cp
        P, N = h(fv.P), h(fv.N)
        alpha = h(fv.alpha).astype(np.float64)
        sP, sN = self.solid[P], self.solid[N]
        iface = sP != sN
        flux = gf * alpha * (T[N] - T[P])  # from N into P across the face (orthogonal part)
        into_fluid = np.where(sN & ~sP, flux, -flux)
        res["fluid-solid interfaces"] = float(into_fluid[iface].sum())
        res["volumetric sources"] = float(h(self.q_cell).sum() * c.cp)
        return res

    def describe(self) -> dict:
        d = self.flow.describe()
        d.update({"solver": d["solver"] + " + conjugate heat transfer", "cells": self.nc,
                  "solid_cells": int(self.solid.sum()), "solid_zones": list(self.case.solids)})
        return d

    @property
    def zone_specs_flow(self):
        return self.flow.zone_specs

    def state_dict(self) -> dict:
        st = self.flow.state_dict()
        st["T_full"] = self.be.to_numpy(self.T)
        return st

    def load_state(self, st: dict):
        self.flow.load_state(st)
        if "T_full" in st:
            self.T = self.xp.asarray(st["T_full"], dtype=self.dtype)

    def initialize(self, fields: dict):
        sub = {k: np.asarray(v)[self.cell_map] for k, v in fields.items() if k != "temperature"}
        self.flow.initialize(sub)
        if "temperature" in fields:
            self.T = self.xp.asarray(np.asarray(fields["temperature"], dtype=np.float64), dtype=self.dtype)

    @property
    def U(self):
        return self.flow.U

    @property
    def turb(self):
        return self.flow.turb


def make_solver(mesh: UnstructuredMesh, case: UnstructuredCase, backend=None):
    """SimpleSolver, or ConjugateSolver when the case has solid zones."""
    if case.solids or any(z.type == "solid" for z in mesh.cell_zones()):
        return ConjugateSolver(mesh, case, backend)
    return SimpleSolver(mesh, case, backend)
