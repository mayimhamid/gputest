"""Case description for the density-based compressible solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

BC_KINDS = ("wall", "slip-wall", "symmetry", "pressure-inlet", "pressure-outlet", "far-field", "velocity-inlet",
            "supersonic-inlet")

# Fluent zone type -> compressible boundary kind
FLUENT_TO_COMP = {
    "wall": "wall",
    "symmetry": "symmetry",
    "axis": "symmetry",
    "pressure-inlet": "pressure-inlet",
    "inlet-vent": "pressure-inlet",
    "intake-fan": "pressure-inlet",
    "pressure-outlet": "pressure-outlet",
    "outlet-vent": "pressure-outlet",
    "exhaust-fan": "pressure-outlet",
    "outflow": "pressure-outlet",
    "pressure-far-field": "far-field",
    "velocity-inlet": "velocity-inlet",
    "mass-flow-inlet": "pressure-inlet",
}


@dataclass
class GasModel:
    """Calorically perfect gas. ``mu`` > 0 with ``sutherland`` uses Sutherland's law around T_ref."""

    gamma: float = 1.4
    R: float = 287.0  # J/(kg K)
    mu: float = 1.7894e-5  # Pa s (Fluent's default for air)
    prandtl: float = 0.72
    sutherland: bool = False
    t_ref: float = 273.11
    s_const: float = 110.56

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @classmethod
    def from_cp(cls, cp: float, R: float = 287.0, **kw) -> "GasModel":
        return cls(gamma=cp / (cp - R), R=R, **kw)


@dataclass
class CompBC:
    """Boundary condition of one face zone (absolute pressures, K).

    wall:             no-slip in viscous runs (slip in inviscid ones); ``wall_T`` None = adiabatic.
    slip-wall / symmetry: zero normal velocity (also the axis of an axisymmetric case).
    pressure-inlet:   total pressure ``p0`` and temperature ``T0``, flow along ``direction``
                      (default: normal into the domain); ``p`` is the static pressure used
                      while the inflow is supersonic (Fluent: supersonic/initial gauge pressure).
    pressure-outlet:  static ``p`` when the outflow is subsonic; ``T`` for backflow.
    far-field:        free stream ``mach``, ``p``, ``T``, ``direction`` (characteristic BC).
    velocity-inlet:   ``velocity`` vector and ``T``; pressure from the interior.
    supersonic-inlet: ``velocity``, ``p`` and ``T`` all imposed.
    """

    kind: str = "wall"
    p0: float = 101325.0
    T0: float = 300.0
    p: float = 101325.0
    T: float = 300.0
    mach: float = 0.0
    direction: Optional[List[float]] = None
    velocity: Optional[List[float]] = None
    wall_T: Optional[float] = None

    def __post_init__(self):
        if self.kind not in BC_KINDS:
            raise ValueError(f"unknown compressible boundary kind {self.kind!r}; choose from {BC_KINDS}")


@dataclass
class CompressibleCase:
    name: str = "compressible case"
    boundaries: Dict[str, CompBC] = field(default_factory=dict)
    gas: GasModel = field(default_factory=GasModel)
    viscous: bool = False  # laminar Navier-Stokes when True, Euler otherwise
    axisymmetric: bool = False  # 2-D meshes only: x = axis, y = radius (Fluent's convention)
    order: int = 2  # 1: first order; 2: MUSCL with limited least-squares gradients
    limiter: str = "venkatakrishnan"  # venkatakrishnan | barth | none
    venkat_k: float = 5.0
    # steady-implicit: matrix-free implicit Euler (Rusanov Jacobian, Jacobi sweeps), local time step;
    # steady-explicit: SSP-RK3 with local time step; transient: SSP-RK3 with a global time step
    time: str = "steady-implicit"
    cfl: float = 1.0
    cfl_max: float = 50.0
    cfl_growth: float = 1.1  # per iteration while the residual decreases
    jacobi_sweeps: int = 8
    dt: Optional[float] = None  # transient: fixed time step (None: CFL-based)
    t_end: float = 1.0
    max_iter: int = 20000
    tol: float = 1e-6  # scaled residuals (each normalised by its largest value in the first 5 iterations)
    init_p: float = 101325.0
    init_T: float = 300.0
    init_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    backend: str = "auto"
    precision: str = "fp64"
    monitor_interval: int = 10

    def __post_init__(self):
        self.boundaries = {k: (CompBC(**v) if isinstance(v, dict) else v) for k, v in self.boundaries.items()}
        if isinstance(self.gas, dict):
            self.gas = GasModel(**self.gas)
        if self.time not in ("steady-implicit", "steady-explicit", "transient"):
            raise ValueError("time must be steady-implicit, steady-explicit or transient")
        if self.limiter not in ("venkatakrishnan", "barth", "none"):
            raise ValueError("limiter must be venkatakrishnan, barth or none")
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")

    def to_dict(self) -> dict:
        return asdict(self)


def default_comp_boundaries(mesh) -> Dict[str, CompBC]:
    return {z.name: CompBC(kind=FLUENT_TO_COMP.get(z.type, "wall")) for z in mesh.boundary_zones()}
