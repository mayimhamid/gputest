r"""Incompressible Navier-Stokes (+ Boussinesq energy) on a staggered MAC grid.

Discretisation
--------------
* Staggered (MAC) arrangement: ``u`` on x-faces, ``v`` on y-faces, ``p``/``T``
  at cell centres -> no pressure checkerboarding, exact discrete
  div-grad compatibility.
* Convection in conservative flux form with central, first-order upwind,
  QUICK or van Leer TVD face interpolation; second-order Adams-Bashforth
  in time (variable step).
* Diffusion Crank-Nicolson in delta form,
  ``(sigma - L0) du = (2/nu) R`` with homogeneous BCs, solved exactly by the
  separable fast solver (DCT/DST/GEMM); explicit when obstacles make the
  operator non-separable.
* Incremental pressure projection: ``-L phi = -div(u*)/dt``, solved by the
  fast separable solver (machine-precision divergence) or by
  multigrid-preconditioned CG with immersed obstacles.

All field arrays live on the selected backend (NumPy or CuPy); only scalars
for time-step control and monitors are transferred to the host.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..hardware.backend import get_backend
from ..linalg import Axis1D, MGPoisson2D, SeparableSolver
from . import kernels as _k
from .case import StructuredCase

G = 2  # ghost layers
# AB2 is stable for nu*dt*(1/dx^2 + 1/dy^2) <= 0.25 on pure diffusion; keep a margin
DIFF_LIMIT_AB2 = 0.2


def resolve_dtype(case: StructuredCase, backend) -> np.dtype:
    if case.precision == "fp64":
        return np.dtype(np.float64)
    if case.precision == "fp32":
        return np.dtype(np.float32)
    if backend.is_gpu:
        from ..hardware.probe import cached_profile

        gpu = cached_profile().best_gpu
        if gpu is not None and gpu.is_datacenter_fp64:
            return np.dtype(np.float64)
        return np.dtype(np.float32)
    return np.dtype(np.float64)


class MACSolver:
    def __init__(self, case: StructuredCase, backend=None, dtype=None, poisson: Optional[str] = None):
        self.case = case
        self.be = backend if backend is not None else get_backend(case.backend)
        self.xp = xp = self.be.xp
        self.dtype = np.dtype(dtype) if dtype is not None else resolve_dtype(case, self.be)
        self.nx, self.ny = nx, ny = case.nx, case.ny
        self.dx, self.dy = case.dx, case.dy
        self.sides = {s: getattr(case, s) for s in ("left", "right", "bottom", "top")}
        self.per_x = case.left.kind == "periodic"
        self.per_y = case.bottom.kind == "periodic"
        self.energy = case.energy
        self.use_jit = bool(self.be.jit) and _k.HAVE_NUMBA
        self.use_cuda = False
        if self.be.is_gpu:
            from . import cuda_kernels as _ck

            self.use_cuda = _ck.available()
            self._ck = _ck
        self.scheme_code = _k.SCHEME_CODES[case.scheme]
        self.timers: Dict[str, float] = defaultdict(float)
        self.history: List[dict] = []
        self.probe_series: List[list] = []
        self._res_norm: Dict[str, float] = {}
        self._res_count = 0
        self.poisson_iters = 0
        self.sources = {"u": None, "v": None, "T": None}

        dt_ = self.dtype
        self.U = xp.zeros((nx + 1 + 2 * G, ny + 2 * G), dtype=dt_)
        self.V = xp.zeros((nx + 2 * G, ny + 1 + 2 * G), dtype=dt_)
        self.P = xp.zeros((nx, ny), dtype=dt_)
        self.T = xp.zeros((nx + 2 * G, ny + 2 * G), dtype=dt_) if self.energy else None
        self.NU_old = self.NV_old = self.NT_old = None
        self.t = 0.0
        self.step_count = 0
        self.dt = None
        self.dt_prev = None
        self.converged = False

        self._build_masks()
        self._build_inlets()
        self.implicit = self._want_implicit() and not self.has_obstacles
        self._build_helmholtz()
        self._build_poisson(poisson or case.poisson)
        self._initialise()

    def set_sources(self, su=None, sv=None, sT=None) -> None:
        """Steady volumetric source terms (per unit volume, momentum/energy equation units).

        ``su``: (nx+1, ny) on the x-faces, ``sv``: (nx, ny+1) on the y-faces,
        ``sT``: (nx, ny) at cell centres. Used for manufactured-solution
        verification and for user body forces that vary in space.
        """
        xp, dt_ = self.xp, self.dtype
        for key, arr, shape in (("u", su, (self.nx + 1, self.ny)), ("v", sv, (self.nx, self.ny + 1)),
                                ("T", sT, (self.nx, self.ny))):
            if arr is None:
                self.sources[key] = None
                continue
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != shape:
                raise ValueError(f"source {key!r} must have shape {shape}, got {a.shape}")
            if key == "T" and not self.energy:
                raise ValueError("energy source given but the energy equation is off")
            self.sources[key] = xp.asarray(a, dtype=dt_)

    # ------------------------------------------------------------------
    # Geometry, masks and boundary data
    # ------------------------------------------------------------------
    @property
    def has_obstacles(self) -> bool:
        return bool(self.solid_h.any())

    def _build_masks(self):
        c, xp = self.case, self.xp
        nx, ny = self.nx, self.ny
        xc = (np.arange(nx) + 0.5) * self.dx
        yc = (np.arange(ny) + 0.5) * self.dy
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        solid = np.zeros((nx, ny), dtype=bool)
        for ob in c.obstacles:
            if ob.shape == "circle":
                solid |= (X - ob.x0) ** 2 + (Y - ob.y0) ** 2 <= ob.r ** 2
            else:
                solid |= (X >= ob.x0) & (X <= ob.x1) & (Y >= ob.y0) & (Y <= ob.y1)
        self.solid_h = solid

        # u faces: adjacent cells (i-1, i)
        sl = np.zeros((nx + 1, ny), dtype=bool)  # left cell solid
        sr = np.zeros((nx + 1, ny), dtype=bool)  # right cell solid
        sl[1:] = solid
        sr[:-1] = solid
        if self.per_x:
            sl[0] = solid[-1]
            sr[-1] = solid[0]
        else:
            sl[0] = solid[0]
            sr[-1] = solid[-1]
        u_blocked = sl | sr
        u_inner = sl & sr
        sb = np.zeros((nx, ny + 1), dtype=bool)
        st = np.zeros((nx, ny + 1), dtype=bool)
        sb[:, 1:] = solid
        st[:, :-1] = solid
        if self.per_y:
            sb[:, 0] = solid[:, -1]
            st[:, -1] = solid[:, 0]
        else:
            sb[:, 0] = solid[:, 0]
            st[:, -1] = solid[:, -1]
        v_blocked = sb | st
        v_inner = sb & st

        # update ranges (faces advanced by the momentum equation)
        self.ux0, self.ux1 = (0, nx) if self.per_x else (1, nx)
        self.vy0, self.vy1 = (0, ny) if self.per_y else (1, ny)
        self.u_fluid = xp.asarray(~u_blocked[self.ux0:self.ux1], dtype=self.dtype)
        self.v_fluid = xp.asarray(~v_blocked[:, self.vy0:self.vy1], dtype=self.dtype)
        self.u_blocked = xp.asarray(u_blocked)
        self.v_blocked = xp.asarray(v_blocked)

        # pressure-correction masks over all faces
        ucorr = ~u_blocked
        vcorr = ~v_blocked
        if not self.per_x:
            ucorr[0] &= self.sides["left"].kind == "outlet"
            ucorr[-1] &= self.sides["right"].kind == "outlet"
        if not self.per_y:
            vcorr[:, 0] &= self.sides["bottom"].kind == "outlet"
            vcorr[:, -1] &= self.sides["top"].kind == "outlet"
        self.u_corr = xp.asarray(ucorr, dtype=self.dtype)
        self.v_corr = xp.asarray(vcorr, dtype=self.dtype)

        # ghost reflections inside obstacles: interior solid faces next to fluid faces
        self._u_refl = self._reflection_pairs(u_inner, u_blocked, axis=1, periodic=self.per_y)
        self._v_refl = self._reflection_pairs(v_inner, v_blocked, axis=0, periodic=self.per_x)
        self._t_ghost = self._scalar_ghosts(solid)
        self.fluid_cells = xp.asarray(~solid, dtype=self.dtype)
        self.n_fluid = int((~solid).sum())

    def _reflection_pairs(self, inner, blocked, axis, periodic):
        fluid = ~blocked
        n = inner.shape[axis]
        groups = {"both": ([], [], []), "one": ([], [])}
        idx = np.argwhere(inner)
        for i, j in idx:
            nbs = []
            for d in (-1, 1):
                k = (j if axis == 1 else i) + d
                if periodic:
                    k %= n
                elif k < 0 or k >= n:
                    continue
                ii, jj = (i, k) if axis == 1 else (k, j)
                if fluid[ii, jj]:
                    nbs.append((ii, jj))
            if len(nbs) == 1:
                groups["one"][0].append((i, j))
                groups["one"][1].append(nbs[0])
            elif len(nbs) == 2:
                groups["both"][0].append((i, j))
                groups["both"][1].append(nbs[0])
                groups["both"][2].append(nbs[1])
        xp = self.xp

        def arr(lst):
            a = np.array(lst, dtype=np.int64).reshape(-1, 2) + G
            return xp.asarray(a[:, 0]), xp.asarray(a[:, 1])

        out = {}
        if groups["one"][0]:
            out["one"] = (arr(groups["one"][0]), arr(groups["one"][1]))
        if groups["both"][0]:
            out["both"] = (arr(groups["both"][0]), arr(groups["both"][1]), arr(groups["both"][2]))
        return out

    def _scalar_ghosts(self, solid):
        nx, ny = solid.shape
        tgt, nbs, w = [], [], []
        for i, j in np.argwhere(solid):
            row_nb, row_w = [], []
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if self.per_x:
                    ii %= nx
                if self.per_y:
                    jj %= ny
                if 0 <= ii < nx and 0 <= jj < ny and not solid[ii, jj]:
                    row_nb.append((ii, jj))
                    row_w.append(1.0)
            if row_nb:
                while len(row_nb) < 4:
                    row_nb.append(row_nb[0])
                    row_w.append(0.0)
                tgt.append((i, j))
                nbs.append(row_nb)
                w.append(row_w)
        if not tgt:
            return None
        xp = self.xp
        tgt = np.array(tgt) + G
        nbs = np.array(nbs) + G
        w = np.array(w)
        # obstacle thermal value per target cell
        tw = np.full(len(tgt), np.nan)
        xc = (tgt[:, 0] - G + 0.5) * self.dx
        yc = (tgt[:, 1] - G + 0.5) * self.dy
        for ob in self.case.obstacles:
            if ob.thermal != "temperature":
                continue
            if ob.shape == "circle":
                m = (xc - ob.x0) ** 2 + (yc - ob.y0) ** 2 <= ob.r ** 2
            else:
                m = (xc >= ob.x0) & (xc <= ob.x1) & (yc >= ob.y0) & (yc <= ob.y1)
            tw[m] = ob.value
        iso = ~np.isnan(tw)
        return {
            "ti": xp.asarray(tgt[:, 0]), "tj": xp.asarray(tgt[:, 1]),
            "ni": xp.asarray(nbs[:, :, 0]), "nj": xp.asarray(nbs[:, :, 1]),
            "w": xp.asarray(w / w.sum(axis=1, keepdims=True), dtype=self.dtype),
            "iso": xp.asarray(iso), "tw": xp.asarray(np.nan_to_num(tw), dtype=self.dtype),
        }

    def _inlet_profile(self, side: str):
        s = self.sides[side]
        n = self.ny if side in ("left", "right") else self.nx
        h = self.dy if side in ("left", "right") else self.dx
        # open (non-solid) span along that side
        if side == "left":
            open_ = ~self.solid_h[0, :]
        elif side == "right":
            open_ = ~self.solid_h[-1, :]
        elif side == "bottom":
            open_ = ~self.solid_h[:, 0]
        else:
            open_ = ~self.solid_h[:, -1]
        prof = np.zeros(n)
        if s.profile == "uniform":
            prof[open_] = s.velocity
        else:
            idx = np.nonzero(open_)[0]
            if idx.size:
                a, b = idx[0] * h, (idx[-1] + 1) * h
                pos = (np.arange(n) + 0.5) * h
                sfrac = np.clip((pos - a) / (b - a), 0.0, 1.0)
                prof = 6.0 * s.velocity * sfrac * (1.0 - sfrac) * open_
        sign = 1.0 if side in ("left", "bottom") else -1.0
        return self.xp.asarray(sign * prof, dtype=self.dtype)

    def _build_inlets(self):
        self.inlet = {s: self._inlet_profile(s) for s, sd in self.sides.items() if sd.kind == "inlet"}

    def _want_implicit(self) -> bool:
        """Crank-Nicolson costs two or three extra separable solves per step; it
        only pays off when explicit diffusion would limit the time step."""
        c = self.case
        if c.implicit_diffusion in (True, False):
            return bool(c.implicit_diffusion)
        inv_h2 = 1.0 / self.dx ** 2 + 1.0 / self.dy ** 2
        diff = max(c.nu, c.kappa if self.energy else 0.0)
        dt_diff = DIFF_LIMIT_AB2 / (diff * inv_h2)
        dt_cfl = c.cfl / (c.u_ref * (1.0 / self.dx + 1.0 / self.dy))
        return dt_diff < min(dt_cfl, c.dt_max)

    # ------------------------------------------------------------------
    # Linear solvers
    # ------------------------------------------------------------------
    @staticmethod
    def _tan_bc(kind):  # tangential velocity component
        return {"wall": "dirichlet", "inlet": "dirichlet", "symmetry": "neumann",
                "outlet": "neumann", "periodic": "periodic"}[kind]

    @staticmethod
    def _p_bc(kind):
        return {"outlet": "dirichlet", "periodic": "periodic"}.get(kind, "neumann")

    def _t_bc(self, side):
        s = self.sides[side]
        if s.kind == "periodic":
            return "periodic"
        if s.kind == "inlet" or (s.kind == "wall" and s.thermal == "temperature"):
            return "dirichlet"
        return "neumann"

    def _build_helmholtz(self):
        self.helm = {}
        if not self.implicit:
            return
        S, nx, ny, dx, dy = self.sides, self.nx, self.ny, self.dx, self.dy
        ax_u = Axis1D(nx, dx, "periodic", "periodic") if self.per_x else Axis1D(nx - 1, dx, "dirichlet", "dirichlet", "node")
        ay_u = Axis1D(ny, dy, self._tan_bc(S["bottom"].kind), self._tan_bc(S["top"].kind))
        ax_v = Axis1D(nx, dx, self._tan_bc(S["left"].kind), self._tan_bc(S["right"].kind))
        ay_v = Axis1D(ny, dy, "periodic", "periodic") if self.per_y else Axis1D(ny - 1, dy, "dirichlet", "dirichlet", "node")
        self.helm["u"] = SeparableSolver([ax_u, ay_u], self.be, self.dtype)
        self.helm["v"] = SeparableSolver([ax_v, ay_v], self.be, self.dtype)
        if self.energy:
            ax_t = Axis1D(nx, dx, self._t_bc("left"), self._t_bc("right"))
            ay_t = Axis1D(ny, dy, self._t_bc("bottom"), self._t_bc("top"))
            self.helm["T"] = SeparableSolver([ax_t, ay_t], self.be, self.dtype)

    def _pressure_axes(self):
        S = self.sides
        return [
            Axis1D(self.nx, self.dx, self._p_bc(S["left"].kind), self._p_bc(S["right"].kind)),
            Axis1D(self.ny, self.dy, self._p_bc(S["bottom"].kind), self._p_bc(S["top"].kind)),
        ]

    def _make_poisson(self, method):
        if method in ("fast", "auto"):
            return SeparableSolver(self._pressure_axes(), self.be, self.dtype, methods="auto")
        if method == "eig":
            return SeparableSolver(self._pressure_axes(), self.be, self.dtype, methods="eig")
        if method == "mg":
            if self.per_x or self.per_y:
                raise ValueError("multigrid pressure solver does not support periodic boundaries yet")
            xp, S = self.xp, self.sides
            nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
            cx = xp.full((nx + 1, ny), dy / dx, dtype=self.dtype)
            cy = xp.full((nx, ny + 1), dx / dy, dtype=self.dtype)
            cx[0] = 2 * dy / dx if S["left"].kind == "outlet" else 0.0
            cx[-1] = 2 * dy / dx if S["right"].kind == "outlet" else 0.0
            cy[:, 0] = 2 * dx / dy if S["bottom"].kind == "outlet" else 0.0
            cy[:, -1] = 2 * dx / dy if S["top"].kind == "outlet" else 0.0
            return MGPoisson2D(cx, cy, 0.0, ~self.xp.asarray(self.solid_h), self.be)
        raise ValueError(f"unknown poisson method {method!r}")

    def _build_poisson(self, method):
        self.poisson_timings = {}
        if self.has_obstacles:
            method = "mg"
        if method == "auto":
            cands = {"fast": self._make_poisson("fast")}
            # dense O(n^3) eigen-decompositions only make sense for moderate n
            if max(self.nx, self.ny) <= 1024 or "eig" in "".join(cands["fast"].methods):
                cands["eig"] = self._make_poisson("eig")
            if "eig" in cands and cands["fast"].methods == cands["eig"].methods:
                cands.pop("eig")
            rng = np.random.default_rng(1)
            rhs = self.xp.asarray(rng.standard_normal((self.nx, self.ny)), dtype=self.dtype)
            for name, solver in cands.items():
                solver.solve(rhs)
                t0 = self.be.timer()
                for _ in range(3):
                    solver.solve(rhs)
                self.poisson_timings[name] = (self.be.timer() - t0) / 3
            method = min(self.poisson_timings, key=self.poisson_timings.get)
            self.poisson = cands[method]
        else:
            self.poisson = self._make_poisson(method)
        self.poisson_method = method
        if isinstance(self.poisson, SeparableSolver):
            self.poisson_detail = "+".join(self.poisson.methods)
        else:
            self.poisson_detail = f"MG-PCG ({self.poisson.n_levels} levels)"

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    def _initialise(self):
        c, xp = self.case, self.xp
        nx, ny = self.nx, self.ny
        self.U[...] = c.init_u
        self.V[...] = c.init_v
        if self.energy:
            self.T[...] = c.init_T
        if c.init_kind == "taylor_green":
            xf = np.arange(nx + 1) * self.dx
            yc = (np.arange(ny) + 0.5) * self.dy
            xc = (np.arange(nx) + 0.5) * self.dx
            yf = np.arange(ny + 1) * self.dy
            u = np.sin(xf)[:, None] * np.cos(yc)[None, :]
            v = -np.cos(xc)[:, None] * np.sin(yf)[None, :]
            p = -0.25 * (np.cos(2 * xc)[:, None] + np.cos(2 * yc)[None, :])
            self.U[G:G + nx + 1, G:G + ny] = xp.asarray(u, dtype=self.dtype)
            self.V[G:G + nx, G:G + ny + 1] = xp.asarray(v, dtype=self.dtype)
            self.P[...] = xp.asarray(p, dtype=self.dtype)
        if c.perturb:
            rng = np.random.default_rng(c.seed)
            if self.energy:
                noise = rng.uniform(-1, 1, (nx, ny)) * c.perturb
                self.T[G:G + nx, G:G + ny] += xp.asarray(noise, dtype=self.dtype)
            else:
                nv = rng.uniform(-1, 1, (nx, ny + 1)) * c.perturb * c.u_ref
                self.V[G:G + nx, G:G + ny + 1] += xp.asarray(nv, dtype=self.dtype)
        self._fill_velocity()
        self._apply_obstacles()
        if not self.per_x:
            for side in ("left", "right"):
                if self.sides[side].kind == "outlet":
                    k = G if side == "left" else G + nx
                    kin = G + 1 if side == "left" else G + nx - 1
                    self.U[k, G:G + ny] = self.U[kin, G:G + ny]
        if not self.per_y:
            for side in ("bottom", "top"):
                if self.sides[side].kind == "outlet":
                    k = G if side == "bottom" else G + ny
                    kin = G + 1 if side == "bottom" else G + ny - 1
                    self.V[G:G + nx, k] = self.V[G:G + nx, kin]
        if c.init_kind != "taylor_green":
            self._project(dt=1.0, update_pressure=False)

    # ------------------------------------------------------------------
    # Boundary conditions (ghost cells)
    # ------------------------------------------------------------------
    def _fill_normal(self, A, axis, n, lo_side, hi_side, periodic):
        """Fill the normal-direction ghosts of a face-centred component."""
        S = self.sides

        def sl(k):
            return (k, slice(None)) if axis == 0 else (slice(None), k)

        if periodic:
            A[sl(G + n)] = A[sl(G)]
            for k in (1, 2):
                A[sl(G - k)] = A[sl(G + n - k)]
                A[sl(G + n + k)] = A[sl(G + k)]
            return
        for side, face, sgn in ((lo_side, G, -1), (hi_side, G + n, +1)):
            kind = S[side].kind
            if kind in ("wall", "symmetry"):
                A[sl(face)] = 0.0
                for k in (1, 2):
                    A[sl(face + sgn * k)] = -A[sl(face - sgn * k)]
            elif kind == "inlet":
                prof = self.inlet[side]
                if axis == 0:
                    A[face, G:G + prof.shape[0]] = prof
                else:
                    A[G:G + prof.shape[0], face] = prof
                for k in (1, 2):
                    A[sl(face + sgn * k)] = A[sl(face)]
            else:  # outlet: face value from the convective condition
                for k in (1, 2):
                    A[sl(face + sgn * k)] = A[sl(face)]

    def _fill_tangential(self, A, axis, n, lo_side, hi_side, periodic):
        """Fill ghosts across the sides parallel to a component (cell-centred along axis)."""
        S = self.sides

        def sl(k):
            return (k, slice(None)) if axis == 0 else (slice(None), k)

        if periodic:
            for k in (1, 2):
                A[sl(G - k)] = A[sl(G + n - k)]
                A[sl(G + n - 1 + k)] = A[sl(G + k - 1)]
            return
        for side, lo in ((lo_side, True), (hi_side, False)):
            s = S[side]
            for k in (1, 2):
                ghost = G - k if lo else G + n - 1 + k
                mirror = G + k - 1 if lo else G + n - k
                if s.kind == "wall":
                    A[sl(ghost)] = 2.0 * s.velocity - A[sl(mirror)]
                elif s.kind == "inlet":
                    A[sl(ghost)] = -A[sl(mirror)]
                elif s.kind == "symmetry":
                    A[sl(ghost)] = A[sl(mirror)]
                else:  # outlet: zero gradient
                    A[sl(ghost)] = A[sl(G if lo else G + n - 1)]

    def _fill_velocity(self):
        nx, ny = self.nx, self.ny
        self._fill_normal(self.U, 0, nx, "left", "right", self.per_x)
        self._fill_tangential(self.U, 1, ny, "bottom", "top", self.per_y)
        self._fill_normal(self.V, 1, ny, "bottom", "top", self.per_y)
        self._fill_tangential(self.V, 0, nx, "left", "right", self.per_x)

    def _fill_temperature(self):
        T, nx, ny, S = self.T, self.nx, self.ny, self.sides
        kap = self.case.kappa
        for axis, n, lo_side, hi_side, per, h in ((0, nx, "left", "right", self.per_x, self.dx),
                                                  (1, ny, "bottom", "top", self.per_y, self.dy)):
            def sl(k):
                return (k, slice(None)) if axis == 0 else (slice(None), k)

            if per:
                for k in (1, 2):
                    T[sl(G - k)] = T[sl(G + n - k)]
                    T[sl(G + n - 1 + k)] = T[sl(G + k - 1)]
                continue
            for side, lo in ((lo_side, True), (hi_side, False)):
                s = S[side]
                for k in (1, 2):
                    ghost = G - k if lo else G + n - 1 + k
                    mirror = G + k - 1 if lo else G + n - k
                    if s.kind == "inlet" or (s.kind == "wall" and s.thermal == "temperature"):
                        T[sl(ghost)] = 2.0 * s.value - T[sl(mirror)]
                    elif s.kind == "wall" and s.thermal == "flux":
                        T[sl(ghost)] = T[sl(mirror)] + (2 * k - 1) * s.value * h / kap
                    else:
                        T[sl(ghost)] = T[sl(mirror)]
        g = self._t_ghost
        if g is not None:
            nb = (T[g["ni"], g["nj"]] * g["w"]).sum(axis=1)
            T[g["ti"], g["tj"]] = self.xp.where(g["iso"], 2.0 * g["tw"] - nb, nb)

    def _apply_obstacles(self):
        if not self.has_obstacles:
            return
        nx, ny = self.nx, self.ny
        Ui = self.U[G:G + nx + 1, G:G + ny]
        Ui[self.u_blocked] = 0.0
        Vi = self.V[G:G + nx, G:G + ny + 1]
        Vi[self.v_blocked] = 0.0
        for A, refl in ((self.U, self._u_refl), (self.V, self._v_refl)):
            if "one" in refl:
                (ti, tj), (si, sj) = refl["one"]
                A[ti, tj] = -A[si, sj]
            if "both" in refl:
                (ti, tj), (ai, aj), (bi, bj) = refl["both"]
                A[ti, tj] = -0.5 * (A[ai, aj] + A[bi, bj])

    # ------------------------------------------------------------------
    # Discrete operators
    # ------------------------------------------------------------------
    def _face(self, a, b, aa, bb, w):
        xp, s = self.xp, self.case.scheme
        if s == "central":
            return 0.5 * (a + b)
        if s == "upwind":
            return xp.where(w >= 0, a, b)
        if s == "quick":
            return xp.where(w >= 0, 0.75 * a + 0.375 * b - 0.125 * aa, 0.75 * b + 0.375 * a - 0.125 * bb)

        def tvd(c, u, d):
            d1 = c - u
            d2 = d - c
            return c + 0.5 * (d1 * xp.abs(d2) + xp.abs(d1) * d2) / (xp.abs(d1) + xp.abs(d2) + 1e-30)

        return xp.where(w >= 0, tvd(a, aa, b), tvd(b, bb, a))

    def _tangential_value(self, side, A_bnd_interior):
        s = self.sides[side]
        if s.kind == "wall":
            return s.velocity
        if s.kind == "inlet":
            return 0.0
        return A_bnd_interior

    def _vel_patch(self, side):
        s = self.sides[side]
        if s.kind == "periodic":
            return 0, 0.0
        if s.kind == "wall":
            return 1, float(s.velocity)
        if s.kind == "inlet":
            return 1, 0.0
        return 2, 0.0

    def _scalar_patch(self, side):
        s = self.sides[side]
        if s.kind == "inlet":
            return 1, float(s.value)
        if s.kind == "outlet":
            return 2, 0.0
        return 0, 0.0

    def _conv_u(self):
        U, V, nx, ny = self.U, self.V, self.nx, self.ny
        if self.use_cuda:
            bm, bv = self._vel_patch("bottom")
            tm, tv = self._vel_patch("top")
            return self._ck.conv_u(U, V, nx, ny, self.dx, self.dy, self.scheme_code, bm, bv, tm, tv)
        if self.use_jit:
            out = np.empty((nx + 1, ny), dtype=self.dtype)
            bm, bv = self._vel_patch("bottom")
            tm, tv = self._vel_patch("top")
            _k.conv_u_nb(U, V, out, nx, ny, self.dx, self.dy, self.scheme_code, bm, bv, tm, tv)
            return out
        a = U[G - 1:G + nx + 1, G:G + ny]
        b = U[G:G + nx + 2, G:G + ny]
        w = 0.5 * (a + b)
        F = w * self._face(a, b, U[G - 2:G + nx, G:G + ny], U[G + 1:G + nx + 3, G:G + ny], w)
        dF = (F[1:] - F[:-1]) / self.dx
        a = U[G:G + nx + 1, G - 1:G + ny]
        b = U[G:G + nx + 1, G:G + ny + 1]
        w = 0.5 * (V[G - 1:G + nx, G:G + ny + 1] + V[G:G + nx + 1, G:G + ny + 1])
        Gf = w * self._face(a, b, U[G:G + nx + 1, G - 2:G + ny - 1], U[G:G + nx + 1, G + 1:G + ny + 2], w)
        if not self.per_y:
            Gf[:, 0] = w[:, 0] * self._tangential_value("bottom", U[G:G + nx + 1, G])
            Gf[:, -1] = w[:, -1] * self._tangential_value("top", U[G:G + nx + 1, G + ny - 1])
        dGy = (Gf[:, 1:] - Gf[:, :-1]) / self.dy
        return dF + dGy

    def _conv_v(self):
        U, V, nx, ny = self.U, self.V, self.nx, self.ny
        if self.use_cuda:
            lm, lv = self._vel_patch("left")
            rm, rv = self._vel_patch("right")
            return self._ck.conv_v(U, V, nx, ny, self.dx, self.dy, self.scheme_code, lm, lv, rm, rv)
        if self.use_jit:
            out = np.empty((nx, ny + 1), dtype=self.dtype)
            lm, lv = self._vel_patch("left")
            rm, rv = self._vel_patch("right")
            _k.conv_v_nb(U, V, out, nx, ny, self.dx, self.dy, self.scheme_code, lm, lv, rm, rv)
            return out
        a = V[G:G + nx, G - 1:G + ny + 1]
        b = V[G:G + nx, G:G + ny + 2]
        w = 0.5 * (a + b)
        Gf = w * self._face(a, b, V[G:G + nx, G - 2:G + ny], V[G:G + nx, G + 1:G + ny + 3], w)
        dGy = (Gf[:, 1:] - Gf[:, :-1]) / self.dy
        a = V[G - 1:G + nx, G:G + ny + 1]
        b = V[G:G + nx + 1, G:G + ny + 1]
        w = 0.5 * (U[G:G + nx + 1, G - 1:G + ny] + U[G:G + nx + 1, G:G + ny + 1])
        F = w * self._face(a, b, V[G - 2:G + nx - 1, G:G + ny + 1], V[G + 1:G + nx + 2, G:G + ny + 1], w)
        if not self.per_x:
            F[0] = w[0] * self._tangential_value("left", V[G, G:G + ny + 1])
            F[-1] = w[-1] * self._tangential_value("right", V[G + nx - 1, G:G + ny + 1])
        dF = (F[1:] - F[:-1]) / self.dx
        return dF + dGy

    def _scalar_face_value(self, side, interior):
        s = self.sides[side]
        if s.kind == "inlet":
            return s.value
        if s.kind == "outlet":
            return interior
        return None

    def _conv_T(self):
        U, V, T, nx, ny = self.U, self.V, self.T, self.nx, self.ny
        if self.use_jit or self.use_cuda:
            pm = [self._scalar_patch(sd) for sd in ("left", "right", "bottom", "top")]
            if self.per_x:
                pm[0] = pm[1] = (0, 0.0)
            if self.per_y:
                pm[2] = pm[3] = (0, 0.0)
            if self.use_cuda:
                return self._ck.conv_T(U, V, T, nx, ny, self.dx, self.dy, self.scheme_code,
                                       [m for m, _ in pm], [v for _, v in pm])
            out = np.empty((nx, ny), dtype=self.dtype)
            modes = np.array([m for m, _ in pm], dtype=np.int64)
            vals = np.array([v for _, v in pm], dtype=np.float64)
            _k.conv_T_nb(U, V, T, out, nx, ny, self.dx, self.dy, self.scheme_code, modes, vals)
            return out
        w = U[G:G + nx + 1, G:G + ny]
        a = T[G - 1:G + nx, G:G + ny]
        b = T[G:G + nx + 1, G:G + ny]
        F = w * self._face(a, b, T[G - 2:G + nx - 1, G:G + ny], T[G + 1:G + nx + 2, G:G + ny], w)
        if not self.per_x:
            for side, k, kin in (("left", 0, G), ("right", -1, G + nx - 1)):
                val = self._scalar_face_value(side, T[kin, G:G + ny])
                if val is not None:
                    F[k] = w[k] * val
        dF = (F[1:] - F[:-1]) / self.dx
        w = V[G:G + nx, G:G + ny + 1]
        a = T[G:G + nx, G - 1:G + ny]
        b = T[G:G + nx, G:G + ny + 1]
        Gf = w * self._face(a, b, T[G:G + nx, G - 2:G + ny - 1], T[G:G + nx, G + 1:G + ny + 2], w)
        if not self.per_y:
            for side, k, kin in (("bottom", 0, G), ("top", -1, G + ny - 1)):
                val = self._scalar_face_value(side, T[G:G + nx, kin])
                if val is not None:
                    Gf[:, k] = w[:, k] * val
        dGy = (Gf[:, 1:] - Gf[:, :-1]) / self.dy
        return dF + dGy

    def _lap(self, A, nxa, nya):
        """5-point Laplacian on the interior block [G:G+nxa, G:G+nya] of a ghosted array."""
        if self.use_cuda:
            return self._ck.lap(A, nxa, nya, self.dx, self.dy)
        if self.use_jit:
            out = np.empty((nxa, nya), dtype=self.dtype)
            _k.lap_nb(A, out, nxa, nya, self.dx, self.dy)
            return out
        c = A[G:G + nxa, G:G + nya]
        return ((A[G + 1:G + nxa + 1, G:G + nya] - 2.0 * c + A[G - 1:G + nxa - 1, G:G + nya]) / self.dx ** 2
                + (A[G:G + nxa, G + 1:G + nya + 1] - 2.0 * c + A[G:G + nxa, G - 1:G + nya - 1]) / self.dy ** 2)

    def _grad_x(self, P):
        xp, nx, S = self.xp, self.nx, self.sides
        g = xp.zeros((nx + 1, self.ny), dtype=self.dtype)
        g[1:nx] = (P[1:] - P[:-1]) / self.dx
        if self.per_x:
            g[0] = (P[0] - P[-1]) / self.dx
            g[nx] = g[0]
        else:
            if S["left"].kind == "outlet":
                g[0] = P[0] / (0.5 * self.dx)
            if S["right"].kind == "outlet":
                g[nx] = -P[-1] / (0.5 * self.dx)
        return g

    def _grad_y(self, P):
        xp, ny, S = self.xp, self.ny, self.sides
        g = xp.zeros((self.nx, ny + 1), dtype=self.dtype)
        g[:, 1:ny] = (P[:, 1:] - P[:, :-1]) / self.dy
        if self.per_y:
            g[:, 0] = (P[:, 0] - P[:, -1]) / self.dy
            g[:, ny] = g[:, 0]
        else:
            if S["bottom"].kind == "outlet":
                g[:, 0] = P[:, 0] / (0.5 * self.dy)
            if S["top"].kind == "outlet":
                g[:, ny] = -P[:, -1] / (0.5 * self.dy)
        return g

    def divergence(self):
        U, V, nx, ny = self.U, self.V, self.nx, self.ny
        return ((U[G + 1:G + nx + 1, G:G + ny] - U[G:G + nx, G:G + ny]) / self.dx
                + (V[G:G + nx, G + 1:G + ny + 1] - V[G:G + nx, G:G + ny]) / self.dy)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def _sync_periodic(self):
        nx, ny = self.nx, self.ny
        if self.per_x:
            self.U[G + nx] = self.U[G]
        if self.per_y:
            self.V[:, G + ny] = self.V[:, G]

    def _solve_pressure(self, rhs):
        """Solve -L phi = rhs (rhs = -div/dt)."""
        if isinstance(self.poisson, SeparableSolver):
            return self.poisson.solve(rhs)
        # warm start from the previous increment: successive phi are strongly correlated
        x0 = getattr(self, "_phi_prev", None)
        phi, info = self.poisson.solve(rhs * (self.dx * self.dy), x0=x0,
                                       tol=1e-8 if self.dtype == np.float64 else 1e-5, maxiter=200)
        self.poisson_iters = info.iterations
        self._phi_prev = phi
        return phi

    def _project(self, dt, update_pressure=True):
        nx, ny = self.nx, self.ny
        self._sync_periodic()
        div = self.divergence()
        if self.has_obstacles:
            div = div * self.fluid_cells
        phi = self._solve_pressure(-div / dt)
        self.U[G:G + nx + 1, G:G + ny] -= dt * self._grad_x(phi) * self.u_corr
        self.V[G:G + nx, G:G + ny + 1] -= dt * self._grad_y(phi) * self.v_corr
        self._sync_periodic()
        if update_pressure:
            self.P += phi
        return div

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _choose_dt(self):
        c = self.case
        xp = self.xp
        nx, ny = self.nx, self.ny
        umax = self.be.scalar(xp.abs(self.U[G:G + nx + 1, G:G + ny]).max())
        vmax = self.be.scalar(xp.abs(self.V[G:G + nx, G:G + ny + 1]).max())
        rate = umax / self.dx + vmax / self.dy
        ref_rate = c.u_ref * (1.0 / self.dx + 1.0 / self.dy)
        dt = c.cfl / max(rate, 0.5 * ref_rate)
        inv_h2 = 1.0 / self.dx ** 2 + 1.0 / self.dy ** 2
        diff = max(c.nu, c.kappa if self.energy else 0.0)
        if not self.implicit:
            dt = min(dt, DIFF_LIMIT_AB2 / (diff * inv_h2))
        if c.scheme != "central":
            # AB2 is stable for real eigenvalues down to dt*lambda = -1. Upwind-biased
            # convection (|Re lambda| <= 2u/h) and explicit diffusion (4 nu / h^2) add
            # up, so the separate limits above are not enough (measured: a channel at
            # CFL 0.5 + explicit diffusion oscillates forever near the outlet).
            eig = 2.0 * max(rate, 0.5 * ref_rate) + (4.0 * diff * inv_h2 if not self.implicit else 0.0)
            dt = min(dt, 0.9 / eig)
        dt = min(dt, c.dt_max)
        if self.dt_prev is not None:
            dt = min(dt, 1.2 * self.dt_prev)
        if c.t_end is not None:
            rem = c.t_end - self.t
            if rem <= dt:
                dt = rem
            elif rem < 1.5 * dt:
                dt = 0.5 * rem  # avoid a sliver of a final step
        if not math.isfinite(dt) or dt <= 0:
            raise FloatingPointError("time step collapsed: the solution diverged")
        return dt

    def _outlet_bc(self, dt):
        nx, ny = self.nx, self.ny
        c = self.case
        for side in ("left", "right", "bottom", "top"):
            if self.sides[side].kind != "outlet":
                continue
            if side in ("left", "right"):
                k, kin = (G, G + 1) if side == "left" else (G + nx, G + nx - 1)
                ub = self.U[k, G:G + ny]
                ui = self.U[kin, G:G + ny]
                h = self.dx
            else:
                k, kin = (G, G + 1) if side == "bottom" else (G + ny, G + ny - 1)
                ub = self.V[G:G + nx, k]
                ui = self.V[G:G + nx, kin]
                h = self.dy
            uc = max(abs(self.be.scalar(ub.mean())), 1e-3 * c.u_ref)
            new = ub - dt * uc * (ub - ui) / h
            if side in ("left", "right"):
                self.U[k, G:G + ny] = new
            else:
                self.V[G:G + nx, k] = new

    @property
    def finished(self) -> bool:
        c = self.case
        if c.t_end is not None and self.t >= c.t_end * (1 - 1e-12):
            return True
        return self.step_count >= c.max_steps or self.converged

    def step(self) -> float:
        """Advance one time step; returns dt (0 when the run is already finished)."""
        if self.finished:
            return 0.0
        c = self.case
        nx, ny = self.nx, self.ny
        tm = self.timers
        t0 = time.perf_counter()
        dt = self._choose_dt()
        omega = dt / self.dt_prev if self.dt_prev else 0.0

        self._fill_velocity()
        self._apply_obstacles()
        if self.energy:
            self._fill_temperature()
        t1 = time.perf_counter()
        tm["boundary"] += t1 - t0

        Nu = self._conv_u()
        Nv = self._conv_v()
        NT = self._conv_T() if self.energy else None

        def ab2(N, Nold):
            if Nold is None:
                return N
            return (1.0 + 0.5 * omega) * N - 0.5 * omega * Nold

        if self.implicit:
            # Crank-Nicolson (delta form): explicit nu*L(u^n) here, implicit half below
            Nu_ab, Nv_ab = ab2(Nu, self.NU_old), ab2(Nv, self.NV_old)
            self.NU_old, self.NV_old = Nu, Nv
            Ru = -Nu_ab + c.nu * self._lap(self.U, nx + 1, ny) - self._grad_x(self.P)
            Rv = -Nv_ab + c.nu * self._lap(self.V, nx, ny + 1) - self._grad_y(self.P)
        else:
            # fully explicit: Adams-Bashforth 2 on convection AND diffusion (2nd order in time)
            Nu = Nu - c.nu * self._lap(self.U, nx + 1, ny)
            Nv = Nv - c.nu * self._lap(self.V, nx, ny + 1)
            Nu_ab, Nv_ab = ab2(Nu, self.NU_old), ab2(Nv, self.NV_old)
            self.NU_old, self.NV_old = Nu, Nv
            Ru = -Nu_ab - self._grad_x(self.P)
            Rv = -Nv_ab - self._grad_y(self.P)
        if self.energy:
            bx, by = c.buoyancy
            Tc = self.T
            if bx:
                Ru = Ru + bx * (0.5 * (Tc[G - 1:G + nx, G:G + ny] + Tc[G:G + nx + 1, G:G + ny]) - c.t_ref)
            if by:
                Rv = Rv + by * (0.5 * (Tc[G:G + nx, G - 1:G + ny] + Tc[G:G + nx, G:G + ny + 1]) - c.t_ref)
            if self.implicit:
                NT_ab = ab2(NT, self.NT_old)
                self.NT_old = NT
                RT = -NT_ab + c.kappa * self._lap(self.T, nx, ny)
            else:
                NT = NT - c.kappa * self._lap(self.T, nx, ny)
                RT = -ab2(NT, self.NT_old)
                self.NT_old = NT
            if self.sources["T"] is not None:
                RT = RT + self.sources["T"]
        if self.sources["u"] is not None:
            Ru = Ru + self.sources["u"]
        if self.sources["v"] is not None:
            Rv = Rv + self.sources["v"]
        t2 = time.perf_counter()
        tm["explicit"] += t2 - t1

        Ru = Ru[self.ux0:self.ux1]
        Rv = Rv[:, self.vy0:self.vy1]
        if self.implicit:
            du = self.helm["u"].solve(Ru * (2.0 / c.nu), sigma=2.0 / (c.nu * dt))
            dv = self.helm["v"].solve(Rv * (2.0 / c.nu), sigma=2.0 / (c.nu * dt))
        else:
            du = dt * Ru * self.u_fluid
            dv = dt * Rv * self.v_fluid
        self._outlet_bc(dt)
        self.U[G + self.ux0:G + self.ux1, G:G + ny] += du
        self.V[G:G + nx, G + self.vy0:G + self.vy1] += dv
        if self.energy:
            if self.implicit:
                dT = self.helm["T"].solve(RT * (2.0 / c.kappa), sigma=2.0 / (c.kappa * dt))
            else:
                dT = dt * RT
            if self.has_obstacles:
                dT = dT * self.fluid_cells
            self.T[G:G + nx, G:G + ny] += dT
        self._apply_obstacles()
        t3 = time.perf_counter()
        tm["momentum_solve"] += t3 - t2

        self._last_div = self._project(dt)
        self._apply_obstacles()
        t4 = time.perf_counter()
        tm["pressure"] += t4 - t3

        self.dt_prev = dt
        self.dt = dt
        self.t += dt
        self.step_count += 1
        self._last_du = (du, dv, dT if self.energy else None)
        return dt

    # ------------------------------------------------------------------
    # Monitors
    # ------------------------------------------------------------------
    def residuals(self) -> dict:
        """Fluent-style residuals of the last step (raw and scaled).

        continuity: RMS of div(u*) before projection; momentum/energy: RMS of
        the time derivative (the steady-state residual of the discrete
        equations). Scaled values are divided by the largest raw value of the
        first five monitor calls, as Fluent does; normalized values are divided
        by the physical scales (u_ref/L, u_ref^2/L, dT u_ref/L) and drive the
        steady-state convergence test.
        """
        xp, be = self.xp, self.be
        du, dv, dT = self._last_du
        dt = self.dt
        raw = {
            "continuity": be.scalar(xp.sqrt(xp.mean(self._last_div ** 2))),
            "x-velocity": be.scalar(xp.sqrt(xp.mean(du ** 2))) / dt,
            "y-velocity": be.scalar(xp.sqrt(xp.mean(dv ** 2))) / dt,
        }
        if dT is not None:
            raw["energy"] = be.scalar(xp.sqrt(xp.mean(dT ** 2))) / dt
        self._res_count += 1
        if self._res_count <= 5:
            for k, v in raw.items():
                self._res_norm[k] = max(self._res_norm.get(k, 0.0), v, 1e-300)
        scaled = {k: v / self._res_norm.get(k, 1.0) for k, v in raw.items()}
        c = self.case
        L = min(c.lx, c.ly)
        ref = {"continuity": c.u_ref / L, "x-velocity": c.u_ref ** 2 / L, "y-velocity": c.u_ref ** 2 / L,
               "energy": self._delta_T() * c.u_ref / L}
        norm = {k: v / ref[k] for k, v in raw.items()}
        if c.steady_tol is not None and self._res_count > 5 and max(norm.values()) < c.steady_tol:
            self.converged = True
        return {"raw": raw, "scaled": scaled, "normalized": norm}

    def max_divergence(self) -> float:
        div = self.divergence()
        if self.has_obstacles:
            div = div * self.fluid_cells
        return self.be.scalar(self.xp.abs(div).max())

    def kinetic_energy(self) -> float:
        xp, nx, ny = self.xp, self.nx, self.ny
        u = self.U[G:G + nx + 1, G:G + ny]
        v = self.V[G:G + nx, G:G + ny + 1]
        return 0.5 * self.be.scalar((xp.sum(u[:-1] ** 2) + xp.sum(v[:, :-1] ** 2))) * self.dx * self.dy

    def nusselt(self, side: str = "left") -> Optional[float]:
        """Mean Nusselt number on a fixed-temperature side (L/dT scaling = 1)."""
        if not self.energy:
            return None
        s = self.sides[side]
        if not (s.kind == "wall" and s.thermal == "temperature"):
            return None
        T, nx, ny = self.T, self.nx, self.ny
        if side in ("left", "right"):
            if side == "left":
                t0, t1 = T[G, G:G + ny], T[G + 1, G:G + ny]
            else:
                t0, t1 = T[G + nx - 1, G:G + ny], T[G + nx - 2, G:G + ny]
            h = self.dx
        else:
            if side == "bottom":
                t0, t1 = T[G:G + nx, G], T[G:G + nx, G + 1]
            else:
                t0, t1 = T[G:G + nx, G + ny - 1], T[G:G + nx, G + ny - 2]
            h = self.dy
        # second-order one-sided wall gradient, magnitude of the wall heat flux
        grad = (-8.0 * s.value + 9.0 * t0 - t1) / (3.0 * h)
        dT = self._delta_T()
        return abs(self.be.scalar(grad.mean())) / dT if dT else None

    def _delta_T(self):
        vals = [s.value for s in self.sides.values() if s.kind == "wall" and s.thermal == "temperature"]
        vals += [ob.value for ob in self.case.obstacles if ob.thermal == "temperature"]
        return (max(vals) - min(vals)) if len(vals) >= 2 and max(vals) > min(vals) else 1.0

    def sample_probes(self) -> list:
        out = []
        f = None
        for (px, py) in self.case.probes:
            if f is None:
                f = self.cell_velocity()
            i = min(max(int(px / self.dx), 0), self.nx - 1)
            j = min(max(int(py / self.dy), 0), self.ny - 1)
            out.append((self.be.scalar(f[0][i, j]), self.be.scalar(f[1][i, j])))
        return out

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def cell_velocity(self):
        nx, ny = self.nx, self.ny
        u = 0.5 * (self.U[G:G + nx, G:G + ny] + self.U[G + 1:G + nx + 1, G:G + ny])
        v = 0.5 * (self.V[G:G + nx, G:G + ny] + self.V[G:G + nx, G + 1:G + ny + 1])
        return u, v

    def vorticity(self):
        """Vorticity at cell centres (average of the four corner values)."""
        nx, ny = self.nx, self.ny
        self._fill_velocity()
        self._apply_obstacles()
        # corners (i, j) for i in 0..nx, j in 0..ny
        dvdx = (self.V[G:G + nx + 1, G:G + ny + 1] - self.V[G - 1:G + nx, G:G + ny + 1]) / self.dx
        dudy = (self.U[G:G + nx + 1, G:G + ny + 1] - self.U[G:G + nx + 1, G - 1:G + ny]) / self.dy
        w = dvdx - dudy
        return 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])

    def streamfunction(self):
        """Discrete stream function at cell centres (psi = 0 on the bottom-left corner)."""
        xp, nx, ny = self.xp, self.nx, self.ny
        u = self.U[G:G + nx + 1, G:G + ny]
        psi_f = xp.zeros((nx + 1, ny + 1), dtype=self.dtype)
        psi_f[:, 1:] = xp.cumsum(u, axis=1) * self.dy
        # shift each x-line by the flux through the bottom boundary
        vb = self.V[G:G + nx, G]
        psi_f -= xp.concatenate([xp.zeros(1, dtype=self.dtype), xp.cumsum(vb) * self.dx])[:, None]
        return 0.25 * (psi_f[:-1, :-1] + psi_f[1:, :-1] + psi_f[:-1, 1:] + psi_f[1:, 1:])

    def fields(self, names=None) -> Dict[str, np.ndarray]:
        """Cell-centred host arrays indexed [i, j] (x, y); solids are NaN."""
        names = names or ["velocity-magnitude", "x-velocity", "y-velocity", "pressure",
                          "vorticity", "stream-function"] + (["temperature"] if self.energy else [])
        nx, ny = self.nx, self.ny
        out = {}
        u = v = None
        for n in names:
            if n in ("x-velocity", "y-velocity", "velocity-magnitude"):
                if u is None:
                    u, v = self.cell_velocity()
                a = {"x-velocity": u, "y-velocity": v}.get(n)
                if a is None:
                    a = self.xp.sqrt(u * u + v * v)
            elif n == "pressure":
                a = self.P - (self.P * self.fluid_cells).sum() / self.n_fluid if not any(
                    s.kind == "outlet" for s in self.sides.values()) else self.P
            elif n == "temperature" and self.energy:
                a = self.T[G:G + nx, G:G + ny]
            elif n == "vorticity":
                a = self.vorticity()
            elif n == "stream-function":
                a = self.streamfunction()
            else:
                continue
            h = self.be.to_numpy(a).astype(np.float64)
            if self.has_obstacles:
                h = np.where(self.solid_h, np.nan, h)
            out[n] = h
        return out

    def centerline(self):
        """(y, u(x=L/2, y)) and (x, v(x, y=H/2)) for cavity validation."""
        nx, ny = self.nx, self.ny
        if nx % 2 == 0:
            u = self.U[G + nx // 2, G:G + ny]
        else:
            u = 0.5 * (self.U[G + nx // 2, G:G + ny] + self.U[G + nx // 2 + 1, G:G + ny])
        if ny % 2 == 0:
            v = self.V[G:G + nx, G + ny // 2]
        else:
            v = 0.5 * (self.V[G:G + nx, G + ny // 2] + self.V[G:G + nx, G + ny // 2 + 1])
        y = (np.arange(ny) + 0.5) * self.dy
        x = (np.arange(nx) + 0.5) * self.dx
        return (y, self.be.to_numpy(u).astype(float)), (x, self.be.to_numpy(v).astype(float))

    def cell_centres(self):
        xc = (np.arange(self.nx) + 0.5) * self.dx
        yc = (np.arange(self.ny) + 0.5) * self.dy
        return xc, yc

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        h = self.be.to_numpy
        st = {"U": h(self.U), "V": h(self.V), "P": h(self.P), "t": self.t, "step": self.step_count,
              "dt_prev": self.dt_prev if self.dt_prev else 0.0}
        if self.energy:
            st["T"] = h(self.T)
        for k in ("NU_old", "NV_old", "NT_old"):
            a = getattr(self, k)
            if a is not None:
                st[k] = h(a)
        return st

    def load_state(self, st: dict):
        xp = self.xp
        self.U[...] = xp.asarray(st["U"], dtype=self.dtype)
        self.V[...] = xp.asarray(st["V"], dtype=self.dtype)
        self.P[...] = xp.asarray(st["P"], dtype=self.dtype)
        if self.energy and "T" in st:
            self.T[...] = xp.asarray(st["T"], dtype=self.dtype)
        for k in ("NU_old", "NV_old", "NT_old"):
            if k in st:
                setattr(self, k, xp.asarray(st[k], dtype=self.dtype))
        self.t = float(st["t"])
        self.step_count = int(st["step"])
        self.dt_prev = float(st.get("dt_prev", 0.0)) or None

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "backend": self.be.name,
            "device": self.be.device,
            "precision": str(self.dtype),
            "grid": [self.nx, self.ny],
            "cells": self.nx * self.ny,
            "pressure_solver": self.poisson_method,
            "pressure_detail": self.poisson_detail,
            "pressure_timings_ms": {k: v * 1e3 for k, v in self.poisson_timings.items()},
            "diffusion": "Crank-Nicolson (exact separable solve)" if self.implicit else "explicit",
            "convection": self.case.scheme,
            "obstacles": self.has_obstacles,
        }
