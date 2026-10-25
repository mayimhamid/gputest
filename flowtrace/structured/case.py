"""Case description for the structured (Cartesian MAC) solver, plus presets.

Everything is plain data so that cases round-trip through JSON (UI, CLI,
checkpoints). Units are whatever the user chooses; presets are
non-dimensional.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

SIDES = ("left", "right", "bottom", "top")
KINDS = ("wall", "inlet", "outlet", "symmetry", "periodic")
THERMAL = ("adiabatic", "temperature", "flux")


@dataclass
class Side:
    """Boundary condition on one side of the rectangle.

    wall:     no penetration; ``velocity`` is the tangential (sliding) speed.
    inlet:    ``velocity`` is the inflow speed (mean speed for a parabolic
              profile), measured into the domain.
    outlet:   convective outflow, static pressure 0.
    symmetry: slip wall.
    periodic: must be set on the opposite side too.
    Thermal: 'adiabatic', 'temperature' (=value) or 'flux' (=value, into the domain).
    """

    kind: str = "wall"
    velocity: float = 0.0
    profile: str = "uniform"
    thermal: str = "adiabatic"
    value: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown boundary kind {self.kind!r}")
        if self.thermal not in THERMAL:
            raise ValueError(f"unknown thermal condition {self.thermal!r}")
        if self.profile not in ("uniform", "parabolic"):
            raise ValueError("profile must be 'uniform' or 'parabolic'")


@dataclass
class Obstacle:
    """Solid body blocked out of the grid (staircase immersed boundary)."""

    shape: str = "circle"  # circle | rect
    x0: float = 0.0  # circle: centre x; rect: lower-left x
    y0: float = 0.0
    x1: float = 0.0  # rect: upper-right x
    y1: float = 0.0
    r: float = 0.5  # circle radius
    thermal: str = "adiabatic"  # adiabatic | temperature
    value: float = 0.0


@dataclass
class StructuredCase:
    name: str = "case"
    nx: int = 128
    ny: int = 128
    lx: float = 1.0
    ly: float = 1.0
    left: Side = field(default_factory=Side)
    right: Side = field(default_factory=Side)
    bottom: Side = field(default_factory=Side)
    top: Side = field(default_factory=Side)
    nu: float = 0.01
    energy: bool = False
    kappa: float = 0.01
    buoyancy: List[float] = field(default_factory=lambda: [0.0, 0.0])
    t_ref: float = 0.0
    obstacles: List[Obstacle] = field(default_factory=list)
    # numerics
    scheme: str = "quick"  # central | upwind | quick | vanleer
    cfl: float = 0.5
    dt_max: float = 1.0
    implicit_diffusion: object = "auto"  # True | False | "auto" (implicit only if diffusion limits dt)
    poisson: str = "auto"  # auto | fast | eig | mg
    # run control
    t_end: Optional[float] = None
    max_steps: int = 20000
    steady_tol: Optional[float] = 1e-5
    monitor_interval: int = 10
    # initial state
    init_u: float = 0.0
    init_v: float = 0.0
    init_T: float = 0.0
    init_kind: str = "uniform"  # uniform | taylor_green
    perturb: float = 0.0
    seed: int = 0
    # execution
    backend: str = "auto"
    precision: str = "auto"  # auto | fp64 | fp32
    probes: List[List[float]] = field(default_factory=list)
    u_ref: float = 1.0
    description: str = ""

    def __post_init__(self):
        for s in SIDES:
            v = getattr(self, s)
            if isinstance(v, dict):
                setattr(self, s, Side(**v))
        self.obstacles = [Obstacle(**o) if isinstance(o, dict) else o for o in self.obstacles]
        if (self.left.kind == "periodic") != (self.right.kind == "periodic"):
            raise ValueError("left/right: periodic must be on both sides")
        if (self.bottom.kind == "periodic") != (self.top.kind == "periodic"):
            raise ValueError("bottom/top: periodic must be on both sides")
        if self.scheme not in ("central", "upwind", "quick", "vanleer"):
            raise ValueError(f"unknown convection scheme {self.scheme!r}")
        if self.nu <= 0:
            raise ValueError("nu must be positive")
        if self.nx < 4 or self.ny < 4:
            raise ValueError("grid must be at least 4x4")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def cells(self) -> int:
        return self.nx * self.ny

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredCase":
        d = dict(d)
        d.pop("solver", None)
        return cls(**d)


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

def lid_driven_cavity(re: float = 100.0, n: int = 128, **kw) -> StructuredCase:
    c = StructuredCase(
        name=f"Lid-driven cavity Re={re:g}",
        nx=n, ny=n, lx=1.0, ly=1.0,
        top=Side("wall", velocity=1.0),
        nu=1.0 / re,
        t_end=None, max_steps=200000, steady_tol=1e-6,
        u_ref=1.0,
        description="Square cavity, moving lid U=1. Validated against Ghia, Ghia & Shin (1982).",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def natural_convection(ra: float = 1e5, pr: float = 0.71, n: int = 128, **kw) -> StructuredCase:
    """Differentially heated square cavity (de Vahl Davis 1983), scaled by alpha/L."""
    c = StructuredCase(
        name=f"Natural convection Ra={ra:g}",
        nx=n, ny=n, lx=1.0, ly=1.0,
        left=Side("wall", thermal="temperature", value=1.0),
        right=Side("wall", thermal="temperature", value=0.0),
        nu=pr, energy=True, kappa=1.0, buoyancy=[0.0, ra * pr], t_ref=0.5,
        init_T=0.5, steady_tol=1e-6, max_steps=200000, dt_max=1e-3,
        u_ref=max(1.0, (ra * pr) ** 0.5 * 0.2),
        description="Hot left wall, cold right wall, adiabatic top/bottom. Reference Nu: "
                    "Ra=1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800.",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def channel_flow(re: float = 100.0, nx: int = 256, ny: int = 32, length: float = 8.0, **kw) -> StructuredCase:
    c = StructuredCase(
        name=f"Channel flow Re={re:g}",
        nx=nx, ny=ny, lx=length, ly=1.0,
        left=Side("inlet", velocity=1.0, profile="uniform"),
        right=Side("outlet"),
        nu=1.0 / re, steady_tol=1e-6, max_steps=100000,
        description="Developing laminar channel flow; fully developed profile u=1.5(1-(2y-1)^2).",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def cylinder_wake(re: float = 100.0, n_per_d: int = 16, **kw) -> StructuredCase:
    lx, ly = 16.0, 8.0
    c = StructuredCase(
        name=f"Cylinder wake Re={re:g}",
        nx=int(lx * n_per_d), ny=int(ly * n_per_d), lx=lx, ly=ly,
        left=Side("inlet", velocity=1.0),
        right=Side("outlet"),
        bottom=Side("symmetry"), top=Side("symmetry"),
        nu=1.0 / re,
        obstacles=[Obstacle("circle", x0=4.0, y0=4.0 + 0.5 / n_per_d, r=0.5)],
        steady_tol=None, t_end=200.0, max_steps=10 ** 7,
        probes=[[8.0, 4.0]], perturb=0.01,
        description="Unsteady von Karman street; Strouhal ~0.16-0.17 at Re=100.",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def backward_facing_step(re: float = 400.0, n_per_h: int = 32, **kw) -> StructuredCase:
    h = 1.0
    lx, ly = 20.0, 2.0 * h
    c = StructuredCase(
        name=f"Backward-facing step Re={re:g}",
        nx=int(lx * n_per_h), ny=int(ly * n_per_h), lx=lx, ly=ly,
        left=Side("inlet", velocity=1.0, profile="parabolic"),
        right=Side("outlet"),
        nu=1.0 / re,
        obstacles=[Obstacle("rect", x0=0.0, y0=0.0, x1=1.0, y1=h)],
        steady_tol=1e-6, max_steps=200000,
        description="Expansion ratio 2, Re based on step height and mean inlet velocity.",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def rayleigh_benard(ra: float = 1e5, pr: float = 1.0, nx: int = 256, ny: int = 64, aspect: float = 4.0, **kw) -> StructuredCase:
    c = StructuredCase(
        name=f"Rayleigh-Benard Ra={ra:g}",
        nx=nx, ny=ny, lx=aspect, ly=1.0,
        left=Side("periodic"), right=Side("periodic"),
        bottom=Side("wall", thermal="temperature", value=1.0),
        top=Side("wall", thermal="temperature", value=0.0),
        nu=pr, energy=True, kappa=1.0, buoyancy=[0.0, ra * pr], t_ref=0.5,
        init_T=0.5, perturb=0.05, steady_tol=None, t_end=1.0, max_steps=10 ** 7,
        dt_max=1e-3, u_ref=max(1.0, (ra * pr) ** 0.5 * 0.2),
        description="Convection rolls between a hot floor and a cold ceiling; periodic in x.",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def taylor_green(n: int = 64, re: float = 100.0, **kw) -> StructuredCase:
    import math

    L = 2.0 * math.pi
    c = StructuredCase(
        name=f"Taylor-Green vortex Re={re:g}",
        nx=n, ny=n, lx=L, ly=L,
        left=Side("periodic"), right=Side("periodic"),
        bottom=Side("periodic"), top=Side("periodic"),
        nu=1.0 / re, steady_tol=None, t_end=1.0, scheme="central", init_kind="taylor_green",
        description="Decaying periodic vortex array with an exact solution: "
                    "u=sin x cos y e^{-2 nu t}.",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


PRESETS = {
    "cavity": lid_driven_cavity,
    "natural_convection": natural_convection,
    "channel": channel_flow,
    "cylinder": cylinder_wake,
    "step": backward_facing_step,
    "rayleigh_benard": rayleigh_benard,
    "taylor_green": taylor_green,
}


def make_preset(name: str, **params) -> StructuredCase:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name](**params)
