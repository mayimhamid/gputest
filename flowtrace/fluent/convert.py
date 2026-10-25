"""Bridges between FlowTrace's structured cases and Fluent's unstructured world."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..mesh.unstructured import UnstructuredMesh, Zone, sort_faces_by_zone

# FlowTrace structured boundary kind -> Fluent zone type
STRUCTURED_TO_FLUENT = {
    "wall": "wall",
    "inlet": "velocity-inlet",
    "outlet": "pressure-outlet",
    "symmetry": "symmetry",
    "periodic": "wall",  # periodic pairs are not exported; flagged in the report
}


def structured_mesh(case, solid: Optional[np.ndarray] = None) -> UnstructuredMesh:
    """Quad mesh of the fluid cells of a structured case.

    Domain sides become zones named left/right/bottom/top with Fluent types
    from their FlowTrace kinds; faces between fluid and obstacle cells form a
    wall zone named 'obstacle'. Fluid cells keep the (i, j) order (i-major).
    """
    nx, ny = case.nx, case.ny
    dx, dy = case.dx, case.dy
    if solid is None:
        solid = np.zeros((nx, ny), dtype=bool)
    fluid = ~solid
    cid = -np.ones((nx, ny), dtype=np.int64)
    cid[fluid] = np.arange(int(fluid.sum()))
    xs = np.arange(nx + 1) * dx
    ys = np.arange(ny + 1) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes_all = np.stack([X.ravel(), Y.ravel()], axis=1)

    def nid(i, j):
        return i * (ny + 1) + j

    faces, own, nei, zone = [], [], [], []
    ZINT, ZL, ZR, ZB, ZT, ZOB = 3, 4, 5, 6, 7, 8
    # vertical faces x = i*dx, i = 0..nx, between cells (i-1, j) and (i, j)
    I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing="ij")
    I, J = I.ravel(), J.ravel()
    left_c = np.where(I > 0, cid[np.clip(I - 1, 0, nx - 1), J], -1)
    right_c = np.where(I < nx, cid[np.clip(I, 0, nx - 1), J], -1)
    n0, n1 = nid(I, J), nid(I, J + 1)  # S = (dy, -dx) -> +x
    _emit(faces, own, nei, zone, n0, n1, left_c, right_c, I == 0, I == nx, ZL, ZR, ZINT, ZOB)
    # horizontal faces y = j*dy between (i, j-1) and (i, j)
    I, J = np.meshgrid(np.arange(nx), np.arange(ny + 1), indexing="ij")
    I, J = I.ravel(), J.ravel()
    low_c = np.where(J > 0, cid[I, np.clip(J - 1, 0, ny - 1)], -1)
    up_c = np.where(J < ny, cid[I, np.clip(J, 0, ny - 1)], -1)
    n0, n1 = nid(I + 1, J), nid(I, J)  # S -> +y
    _emit(faces, own, nei, zone, n0, n1, low_c, up_c, J == 0, J == ny, ZB, ZT, ZINT, ZOB)
    fn = np.concatenate([f for f in faces]).astype(np.int64)
    owner = np.concatenate(own).astype(np.int64)
    neigh = np.concatenate(nei).astype(np.int64)
    fz = np.concatenate(zone).astype(np.int64)
    # drop unused nodes
    used = np.unique(fn)
    remap = -np.ones(nodes_all.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    fn = remap[fn]
    nodes = nodes_all[used]
    nc = int(fluid.sum())
    zones = {2: Zone(2, "fluid", "fluid", "cell", 0, nc - 1),
             ZINT: Zone(ZINT, "interior-fluid", "interior", "face")}
    for zid, side in ((ZL, "left"), (ZR, "right"), (ZB, "bottom"), (ZT, "top")):
        zones[zid] = Zone(zid, side, STRUCTURED_TO_FLUENT[getattr(case, side).kind], "face")
    if np.any(fz == ZOB):
        zones[ZOB] = Zone(ZOB, "obstacle", "wall", "face")
    present = set(np.unique(fz).tolist())
    zones = {k: z for k, z in zones.items() if z.kind == "cell" or k in present}
    m = UnstructuredMesh(dim=2, nodes=nodes, face_ptr=np.arange(0, 2 * fn.shape[0] + 1, 2, dtype=np.int64),
                         face_nodes=fn.ravel(), owner=owner, neighbour=neigh, face_zone=fz, n_cells=nc,
                         cell_zone=np.full(nc, 2, dtype=np.int64), zones=zones, name=case.name)
    return sort_faces_by_zone(m)


def _emit(faces, own, nei, zone, n0, n1, lo_c, hi_c, at_lo, at_hi, z_lo, z_hi, z_int, z_ob):
    # interior fluid-fluid
    m = (lo_c >= 0) & (hi_c >= 0)
    faces.append(np.stack([n0[m], n1[m]], axis=1)); own.append(lo_c[m]); nei.append(hi_c[m])
    zone.append(np.full(m.sum(), z_int))
    # boundary faces with the fluid cell on the high side (normal must point to -axis)
    m = (hi_c >= 0) & (lo_c < 0)
    faces.append(np.stack([n1[m], n0[m]], axis=1)); own.append(hi_c[m]); nei.append(np.full(m.sum(), -1))
    zone.append(np.where(at_lo[m], z_lo, z_ob))
    # boundary faces with the fluid cell on the low side
    m = (lo_c >= 0) & (hi_c < 0)
    faces.append(np.stack([n0[m], n1[m]], axis=1)); own.append(lo_c[m]); nei.append(np.full(m.sum(), -1))
    zone.append(np.where(at_hi[m], z_hi, z_ob))


def structured_solution_points(solver) -> Dict[str, np.ndarray]:
    """Cell-centre coordinates and Fluent-named fields of the fluid cells."""
    xc, yc = solver.cell_centres()
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    fluid = ~solver.solid_h
    names = ["x-velocity", "y-velocity", "pressure"] + (["temperature"] if solver.energy else [])
    f = solver.fields(names)
    pts = np.stack([X[fluid], Y[fluid]], axis=1)
    return {"points": pts, "fields": {k: v[fluid] for k, v in f.items()}}


def structured_boundary_specs(case) -> dict:
    """FlowTrace structured sides -> unstructured BoundarySpec dicts (by zone name)."""
    out = {}
    for side in ("left", "right", "bottom", "top"):
        s = getattr(case, side)
        if s.kind == "wall":
            tang = s.velocity
            vel = [tang, 0.0] if side in ("bottom", "top") else [0.0, tang]
            out[side] = {"kind": "wall", "velocity": vel, "thermal": s.thermal, "value": s.value}
        elif s.kind == "inlet":
            out[side] = {"kind": "inlet", "normal_speed": s.velocity, "value": s.value}
        elif s.kind == "outlet":
            out[side] = {"kind": "outlet", "pressure": 0.0}
        elif s.kind == "symmetry":
            out[side] = {"kind": "symmetry"}
    if case.obstacles:
        ob = case.obstacles[0]
        out["obstacle"] = {"kind": "wall", "thermal": "temperature" if ob.thermal == "temperature" else "adiabatic",
                           "value": ob.value}
    return out
