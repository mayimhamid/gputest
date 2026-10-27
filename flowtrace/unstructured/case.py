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
RADIATION_MODELS = ("off", "p1")
TURBULENCE_ALIASES = {"sst": "k-omega-sst", "kw-sst": "k-omega-sst", "k-omega": "k-omega-sst", "none": "laminar"}


@dataclass
class BoundarySpec:
    """Boundary condition for one face zone.

    wall:      velocity = wall velocity vector (default no-slip at rest);
               thermal: 'adiabatic' | 'temperature' | 'flux' with ``value``.
    inlet:     velocity vector, or ``normal_speed`` (> 0 into the domain); T = value.
    outlet:    static gauge ``pressure``; zero-gradient velocity and temperature.
    pressure:  pressure inlet/opening: given static ``pressure``, inflow allowed, T = value.
    symmetry:  slip, zero normal flux.
    periodic:  translational periodic zone; pair it with ``UnstructuredCase.periodic``.
    ``emissivity``: wall emissivity for the P-1 radiation model.
    ``species``: {name: value} at inlets/openings (and on walls to fix a wall value;
               walls are zero-flux otherwise).
    """

    kind: str = "wall"
    velocity: Optional[List[float]] = None
    normal_speed: Optional[float] = None
    pressure: float = 0.0
    thermal: str = "adiabatic"
    value: float = 0.0
    # turbulence at inlets / backflow (None: the case defaults)
    turb_intensity: Optional[float] = None
    turb_viscosity_ratio: Optional[float] = None
    k: Optional[float] = None
    omega: Optional[float] = None
    epsilon: Optional[float] = None
    # wall rotating about an axis (absolute frame): {"omega": rad/s, "origin": [x, y, z], "axis": [0, 0, 1]}
    rotation: Optional[dict] = None
    emissivity: float = 1.0
    species: Optional[Dict[str, float]] = None

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
    body_force: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # uniform force per volume
    algorithm: str = "SIMPLEC"  # SIMPLE | SIMPLEC
    relax_u: float = 0.7
    # pressure under-relaxation; None = automatic: SIMPLE 0.3, SIMPLEC 1.0 on orthogonal meshes and
    # 0.8 when the mesh is non-orthogonal (> 5 deg), following Fluent's guidance for skewed meshes
    relax_p: Optional[float] = None
    relax_T: float = 1.0
    scheme: str = "second-order"  # upwind | second-order
    # convection scheme of energy and species (None: as ``scheme``); first-order stays within the boundary
    # values where the second-order scheme overshoots (skewed meshes)
    scalar_scheme: Optional[str] = None  # None | upwind | second-order
    # pressure force in momentum: "green-gauss" = sum_f p_f S_f (face pressures, as
    # Fluent does; consistent on triangles) | "least-squares" (cell LS gradient)
    pressure_gradient: str = "green-gauss"
    # relative residual reduction of the AMG-PCG pressure-correction solve per outer
    # iteration (Fluent's AMG default is 0.1; measured: same outer iterations as 1e-3,
    # 15-50 % less wall time on the cavity and the SST flat plate)
    pressure_tol: float = 0.1
    # turbulence: laminar | k-omega-sst | k-epsilon-realizable
    turbulence: str = "laminar"
    turb_intensity: float = 0.05  # inlet default (Fluent: 5 %)
    turb_viscosity_ratio: float = 10.0  # inlet default (Fluent: 10)
    turb_reference_velocity: Optional[float] = None  # for domains without inlets (periodic)
    turb_init: Optional[Dict[str, float]] = None  # {"k": .., "omega"|"epsilon": ..}
    # under-relaxation of k and omega/epsilon (None: model default, SST 0.8, k-epsilon 0.5;
    # the segregated k-epsilon iteration measured unstable at 0.8 on wall-function meshes)
    relax_k: Optional[float] = None
    relax_w: Optional[float] = None
    relax_mut: float = 1.0  # turbulent viscosity under-relaxation
    turb_scheme: str = "upwind"  # upwind | second-order
    # moving reference frames: {cell zone name: {"omega": rad/s, "origin": [..], "axis": [0, 0, 1]}}
    mrf: Dict[str, dict] = field(default_factory=dict)
    # solid cell zones for conjugate heat transfer: {zone name: {"conductivity": k, "source": q'''}}
    solids: Dict[str, dict] = field(default_factory=dict)
    max_iter: int = 2000
    tol: float = 1e-5
    init_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    init_T: float = 0.0
    backend: str = "auto"
    precision: str = "fp64"
    monitor_interval: int = 1
    # --- periodic flow given as zone pairs (applied to the mesh with make_periodic) ---
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
    # --- species transport (non-reacting scalars) ---
    species: List[str] = field(default_factory=list)
    species_diffusivity: float = 1e-5  # mass diffusivity D (m^2/s); flux = rho D grad Y
    schmidt_turbulent: float = 0.7
    prandtl_turbulent: float = 0.85
    # --- reports ---
    ref_velocity: float = 1.0
    ref_area: float = 1.0

    def __post_init__(self):
        self.boundaries = {k: (BoundarySpec(**v) if isinstance(v, dict) else v) for k, v in self.boundaries.items()}
        if self.algorithm.upper() not in ("SIMPLE", "SIMPLEC"):
            raise ValueError("algorithm must be SIMPLE or SIMPLEC")
        self.algorithm = self.algorithm.upper()
        if self.relax_p is None and self.algorithm == "SIMPLE":
            self.relax_p = 0.3
        from .turbulence import MODELS

        self.turbulence = TURBULENCE_ALIASES.get(str(self.turbulence).lower(), str(self.turbulence).lower())
        if self.turbulence not in MODELS:
            raise ValueError(f"turbulence must be one of {MODELS}")
        if self.pressure_gradient not in ("green-gauss", "least-squares"):
            raise ValueError("pressure_gradient must be 'green-gauss' or 'least-squares'")
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
    if getattr(mesh, "periodic", None):
        return []  # already paired on the mesh (make_periodic)
    prim = [z for z in zones if z.type == "periodic"]
    shad = [z for z in zones if z.type == "periodic-shadow"]
    pairs = [[a.name, b.name] for a, b in zip(prim, shad)]
    used = {n for p in pairs for n in p}
    rest = [z.name for z in zones if z.name not in used and boundaries is not None
            and getattr(boundaries.get(z.name), "kind", None) == "periodic"]
    pairs += [[rest[i], rest[i + 1]] for i in range(0, len(rest) - 1, 2)]
    return pairs
