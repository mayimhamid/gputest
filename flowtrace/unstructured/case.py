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
    "periodic": "periodic",
    "periodic-shadow": "periodic",
}
KINDS = ("wall", "inlet", "outlet", "pressure", "symmetry", "periodic")
TURBULENCE_MODELS = ("laminar", "k-omega-sst")
RADIATION_MODELS = ("off", "p1")


@dataclass
class BoundarySpec:
    """Boundary condition for one face zone.

    wall:      velocity = wall velocity vector (default no-slip at rest);
               thermal: 'adiabatic' | 'temperature' | 'flux' with ``value``;
               ``emissivity`` for the P-1 radiation model.
    inlet:     velocity vector, or ``normal_speed`` (> 0 into the domain); T = value;
               turbulence from ``turb_intensity`` (fraction) and ``turb_viscosity_ratio``.
    outlet:    static gauge ``pressure``; zero-gradient velocity, T, k, omega
               (backflow uses the inlet-style turbulence values).
    pressure:  pressure inlet/opening: given static ``pressure``, inflow allowed, T = value.
    ``species``: {name: value} at inlets/openings (and on walls to fix a wall value;
               walls are zero-flux otherwise).
    symmetry:  slip, zero normal flux.
    periodic:  translational periodic zone; pair it with ``UnstructuredCase.periodic``.
    """

    kind: str = "wall"
    velocity: Optional[List[float]] = None
    normal_speed: Optional[float] = None
    pressure: float = 0.0
    thermal: str = "adiabatic"
    value: float = 0.0
    turb_intensity: Optional[float] = None
    turb_viscosity_ratio: Optional[float] = None
    emissivity: float = 1.0
    species: Optional[Dict[str, float]] = None  # inlet/backflow (or fixed wall) values per species

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
    relax_turb: float = 0.8
    scheme: str = "second-order"  # upwind | second-order
    max_iter: int = 2000
    tol: float = 1e-5
    init_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    init_T: float = 0.0
    backend: str = "auto"
    precision: str = "fp64"
    monitor_interval: int = 1
    # --- turbulence (Fluent: Models > Viscous) ---
    turbulence: str = "laminar"  # laminar | k-omega-sst
    turb_intensity: float = 0.05  # default for inlets / initial field
    turb_viscosity_ratio: float = 10.0
    prandtl_turbulent: float = 0.85
    turb_scheme: str = "upwind"  # k and omega convection: upwind (robust) | second-order
    # --- periodic flow (Fluent: periodic conditions) ---
    periodic: List[List[str]] = field(default_factory=list)  # [[zone_a, zone_b], ...]
    periodic_pressure_gradient: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # mean dp/dx
    # --- transient (Fluent: General > Time > Transient) ---
    unsteady: bool = False
    dt: float = 0.01
    t_end: float = 1.0
    time_scheme: str = "bdf2"  # euler | bdf2
    inner_iterations: int = 20
    inner_tol: float = 1e-4
    # --- radiation (Fluent: Models > Radiation > P1) ---
    radiation: str = "off"  # off | p1
    absorption: float = 1.0  # 1/m
    scattering: float = 0.0  # 1/m
    # --- species transport (Fluent: species / user-defined scalars, non-reacting) ---
    species: List[str] = field(default_factory=list)
    species_diffusivity: float = 1e-5  # mass diffusivity D (m^2/s); flux = rho D grad Y
    schmidt_turbulent: float = 0.7
    # --- reports ---
    ref_velocity: float = 1.0
    ref_area: float = 1.0

    def __post_init__(self):
        self.boundaries = {k: (BoundarySpec(**v) if isinstance(v, dict) else v) for k, v in self.boundaries.items()}
        if self.algorithm.upper() not in ("SIMPLE", "SIMPLEC"):
            raise ValueError("algorithm must be SIMPLE or SIMPLEC")
        self.algorithm = self.algorithm.upper()
        if self.algorithm == "SIMPLEC" and self.relax_p == 0.3:
            self.relax_p = 1.0  # SIMPLEC needs no pressure under-relaxation
        aliases = {"sst": "k-omega-sst", "kw-sst": "k-omega-sst", "k-omega": "k-omega-sst", "none": "laminar"}
        self.turbulence = aliases.get(str(self.turbulence).lower(), str(self.turbulence).lower())
        if self.turbulence not in TURBULENCE_MODELS:
            raise ValueError(f"turbulence must be one of {TURBULENCE_MODELS}")
        self.radiation = str(self.radiation).lower()
        if self.radiation not in RADIATION_MODELS:
            raise ValueError(f"radiation must be one of {RADIATION_MODELS}")
        if self.radiation != "off" and not self.energy:
            raise ValueError("the radiation model needs the energy equation (energy=True)")
        if self.time_scheme not in ("euler", "bdf2"):
            raise ValueError("time_scheme must be 'euler' or 'bdf2'")
        self.periodic = [list(p) for p in self.periodic]

    @property
    def turbulent(self) -> bool:
        return self.turbulence != "laminar"

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


def default_periodic_pairs(mesh, boundaries: Optional[Dict[str, BoundarySpec]] = None) -> List[List[str]]:
    """Pair periodic zones: Fluent 'periodic' + 'periodic-shadow' types first, then any zones
    whose boundary spec says ``periodic`` (consecutive pairs in zone order)."""
    zones = sorted(mesh.boundary_zones(), key=lambda z: z.id)
    prim = [z for z in zones if z.type == "periodic"]
    shad = [z for z in zones if z.type == "periodic-shadow"]
    pairs = [[a.name, b.name] for a, b in zip(prim, shad)]
    used = {n for p in pairs for n in p}
    rest = [z.name for z in zones if z.name not in used and boundaries is not None
            and getattr(boundaries.get(z.name), "kind", None) == "periodic"]
    pairs += [[rest[i], rest[i + 1]] for i in range(0, len(rest) - 1, 2)]
    return pairs
