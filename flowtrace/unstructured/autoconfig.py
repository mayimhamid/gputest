"""Automatic solver options from the mesh and the boundary conditions.

``auto_configure`` looks at what the case is (inlet speed and size, fluid properties, wall-cell
height, mesh skewness, temperatures and species named on the boundaries) and sets the options a
careful user would set by hand, returning every decision with its reason so the run log shows it:

* mesh length unit (Fluent: General > Scale): 400-unit extent -> millimetres;
* flow regime from Re = rho U L / mu: laminar below 2300, otherwise a turbulence model, chosen by
  the estimated wall y+ (viscous sublayer resolved -> k-omega SST, wall functions -> realizable k-epsilon);
* energy equation when the boundaries carry different or fixed temperatures;
* species transport for the species named on inlets;
* bounded (first-order) scalar transport on skewed meshes, where the second-order scheme was
  measured to overshoot the boundary temperatures by 10 % of their range (elbow, 45 deg);
* first-order momentum on extremely skewed meshes, and an iteration budget that grows with the mesh.

Only the options listed in ``AUTO_FIELDS`` are touched; the fluid properties and the boundary values
always stay the user's.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..mesh.unstructured import UnstructuredMesh, compute_geometry, scale_mesh

AUTO_FIELDS = ("turbulence", "energy", "scheme", "scalar_scheme", "species", "max_iter", "init_T", "t_ref")
UNIT_FACTORS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "in": 0.0254}


def guess_units(mesh: UnstructuredMesh) -> Tuple[str, str]:
    """Length unit of a mesh from its extent: pipes, elbows and test sections are drawn in mm far more
    often than they are 100 m long, so an extent of 20 or more is read as millimetres."""
    span = float(np.ptp(mesh.nodes, axis=0).max())
    if span >= 20.0:
        return "mm", f"mesh extent {span:g}: taken as millimetres (set the unit explicitly to override)"
    return "m", f"mesh extent {span:g}: taken as metres"


def _inlets(case):
    return {n: b for n, b in case.boundaries.items() if b.kind == "inlet"}


def _zone_faces(mesh, name):
    z = mesh.zone_by_name(name)
    return np.nonzero((mesh.neighbour < 0) & (mesh.face_zone == z.id))[0] if z is not None else np.zeros(0, int)


def _non_orthogonality(mesh, geo, limit: int = 2_000_000) -> float:
    inner = np.nonzero(mesh.neighbour >= 0)[0]
    if inner.size == 0:
        return 0.0
    if inner.size > limit:
        inner = inner[:: inner.size // limit + 1]
    d = geo.cell_centre[mesh.neighbour[inner]] - geo.cell_centre[mesh.owner[inner]]
    S = geo.face_area_vec[inner]
    cos = np.einsum("fd,fd->f", S, d) / np.maximum(np.linalg.norm(S, axis=1) * np.linalg.norm(d, axis=1), 1e-300)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max())


def auto_configure(mesh: UnstructuredMesh, case, units: str = "auto") -> Tuple[UnstructuredMesh, List[str], Dict]:
    """Return ``(mesh, notes, applied)``: the mesh scaled to metres, one note per decision and the
    option values set on ``case`` (which is modified in place)."""
    notes: List[str] = []
    applied: Dict[str, object] = {}

    def set_(name, value, why):
        setattr(case, name, value)
        applied[name] = value
        notes.append(f"auto: {name} = {value!r}: {why}")

    # ---- length unit
    if units == "auto":
        units, why = guess_units(mesh)
        notes.append("auto: " + why)
    if units not in UNIT_FACTORS:
        raise ValueError(f"unknown mesh unit {units!r}; use one of {', '.join(UNIT_FACTORS)}")
    if UNIT_FACTORS[units] != 1.0:
        mesh = scale_mesh(mesh, UNIT_FACTORS[units])
        notes.append(f"mesh scaled by {UNIT_FACTORS[units]:g} ({units} -> m)")
    geo = compute_geometry(mesh)
    skew = _non_orthogonality(mesh, geo)

    # ---- flow regime
    inlets = _inlets(case)
    speed, length = 0.0, 0.0
    for name, b in inlets.items():
        u = b.normal_speed if b.normal_speed is not None else (np.linalg.norm(b.velocity) if b.velocity else 0.0)
        f = _zone_faces(mesh, name)
        area = float(geo.face_area[f].sum())
        d = area if mesh.dim == 2 else math.sqrt(4.0 * area / math.pi)  # width, or equivalent diameter
        speed = max(speed, abs(float(u)))
        if area * abs(float(u)) >= length * speed:  # the inlet that carries the most
            length = d
    if speed > 0 and length > 0:
        re = case.density * speed * length / case.viscosity
        notes.append(f"flow estimate: U = {speed:g} m/s, L = {length:g} m, Re = {re:.3g}")
        walls = [n for n, b in case.boundaries.items() if b.kind == "wall"]
        wf = np.concatenate([_zone_faces(mesh, n) for n in walls]) if walls else np.zeros(0, int)
        if re < 2300.0:
            set_("turbulence", "laminar", f"Re {re:.3g} < 2300")
        else:
            h = 1e30
            if wf.size:
                # wall-adjacent cell height ~ cell volume / wall face area
                own = mesh.owner[wf]
                h = float(np.median(geo.cell_volume[own] / np.maximum(geo.face_area[wf], 1e-300)) / 2.0)
            yplus = h * (0.05 * speed) * case.density / case.viscosity
            model = "k-omega-sst" if yplus <= 5.0 else "k-epsilon-realizable"
            set_("turbulence", model, f"Re {re:.3g} >= 2300; estimated first-cell y+ ~ {yplus:.3g} "
                 + ("(sublayer resolved)" if yplus <= 5.0 else "(wall functions)"))
    else:
        notes.append("no inlet speed given (pressure- or force-driven case): turbulence model left as set")

    # ---- thermal
    temps = {round(float(b.value), 9) for b in case.boundaries.values()
             if (b.kind in ("inlet", "pressure")) or (b.kind == "wall" and b.thermal != "adiabatic")}
    thermal_walls = any(b.kind == "wall" and b.thermal != "adiabatic" for b in case.boundaries.values())
    inlet_T = {round(float(b.value), 9) for b in inlets.values()}
    if (len(inlet_T - {0.0}) >= 1 and len(inlet_T) >= 2) or thermal_walls:
        set_("energy", True, "boundaries carry different or fixed temperatures")
        ref = [t for t in temps if t > 0.0]
        if ref and case.init_T == 0.0:
            set_("init_T", float(np.mean(ref)), "mean of the boundary temperatures")
        if ref and case.t_ref == 0.0:
            set_("t_ref", float(np.mean(ref)), "mean of the boundary temperatures (buoyancy reference)")
        if skew > 30.0:
            set_("scalar_scheme", "upwind", f"{skew:.0f} deg non-orthogonality: second-order energy overshoots "
                 "the boundary temperatures here, first-order stays bounded")

    # ---- species named on the inlets
    names = sorted({k for b in case.boundaries.values() for k in (b.species or {})})
    if names and not case.species:
        set_("species", names, "species given on the boundaries")
        if skew > 30.0 and "scalar_scheme" not in applied:
            set_("scalar_scheme", "upwind", f"{skew:.0f} deg non-orthogonality: bounded species transport")

    # ---- discretisation and iteration budget
    if skew > 60.0:
        set_("scheme", "upwind", f"{skew:.0f} deg non-orthogonality: second-order momentum is not robust")
    budget = 3000 if mesh.n_cells < 100_000 else 5000
    if case.max_iter < budget:
        set_("max_iter", budget, f"{mesh.n_cells:,} cells")
    notes.append(f"mesh: max non-orthogonality {skew:.1f} deg")
    return mesh, notes, applied
