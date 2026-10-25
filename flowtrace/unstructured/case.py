"""Case description for the unstructured (Fluent-mesh) pressure-based solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Fluent zone type -> FlowTrace boundary kind
FLUENT_TO_KIND = {
    "wall": "wall",
    "velocity-inlet": "inlet",
    "mass-flow-inlet": "inlet",
    "pressure-inlet": "pressure",
    "pressure-outlet": "outlet",
    "outflow": "outlet",
    "pressure-far-field": "outlet",
    "outlet-vent": "outlet",
    "exhaust-fan": "outlet",
    "inlet-vent": "pressure",
    "intake-fan": "pressure",
    "symmetry": "symmetry",
    "axis": "symmetry",
}
KINDS = ("wall", "inlet", "outlet", "pressure", "symmetry")


@dataclass
class BoundarySpec:
    """Boundary condition for one face zone.

    wall:      velocity = wall velocity vector (default no-slip at rest);
               thermal: 'adiabatic' | 'temperature' | 'flux' with ``value``.
    inlet:     velocity vector, or ``normal_speed`` (> 0 into the domain); T = value.
    outlet:    static gauge ``pressure``; zero-gradient velocity and temperature.
    pressure:  pressure inlet/opening: given static ``pressure``, inflow allowed, T = value.
    symmetry:  slip, zero normal flux.
    """

    kind: str = "wall"
    velocity: Optional[List[float]] = None
    normal_speed: Optional[float] = None
    pressure: float = 0.0
    thermal: str = "adiabatic"
    value: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown boundary kind {self.kind!r}")


@dataclass
class UnstructuredCase:
    name: str = "unstructured case"
    mesh_file: str = ""
    boundaries: Dict[str, BoundarySpec] = field(default_factory=dict)
    density: float = 1.0
    viscosity: float = 0.01  # dynamic viscosity
    energy: bool = False
    conductivity: float = 0.01
    cp: float = 1.0
    buoyancy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # rho*beta*g vector
    t_ref: float = 0.0
    algorithm: str = "SIMPLEC"  # SIMPLE | SIMPLEC
    relax_u: float = 0.7
    relax_p: float = 0.3
    relax_T: float = 1.0
    scheme: str = "second-order"  # upwind | second-order
    max_iter: int = 2000
    tol: float = 1e-5
    init_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    init_T: float = 0.0
    backend: str = "auto"
    precision: str = "fp64"
    monitor_interval: int = 1

    def __post_init__(self):
        self.boundaries = {k: (BoundarySpec(**v) if isinstance(v, dict) else v) for k, v in self.boundaries.items()}
        if self.algorithm.upper() not in ("SIMPLE", "SIMPLEC"):
            raise ValueError("algorithm must be SIMPLE or SIMPLEC")
        self.algorithm = self.algorithm.upper()
        if self.algorithm == "SIMPLEC" and self.relax_p == 0.3:
            self.relax_p = 1.0  # SIMPLEC needs no pressure under-relaxation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UnstructuredCase":
        return cls(**d)


def default_boundaries(mesh) -> Dict[str, BoundarySpec]:
    out = {}
    for z in mesh.boundary_zones():
        kind = FLUENT_TO_KIND.get(z.type, "wall")
        spec = BoundarySpec(kind=kind)
        if kind == "inlet":
            spec.normal_speed = 1.0
        out[z.name] = spec
    return out
