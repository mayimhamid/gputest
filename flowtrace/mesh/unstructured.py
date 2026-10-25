"""Face-based unstructured mesh (Fluent's data model) and its FV geometry.

Connectivity is stored exactly the way Fluent stores it: a list of faces, each
with its nodes and the two adjacent cells (owner = c0, neighbour = c1, -1 on
boundaries). Any cell shape (triangles, quads, polygons, tets, hexes, prisms,
pyramids, polyhedra) is therefore supported by the same code.

Orientation: face area vectors S_f point from the owner to the neighbour (and
out of the domain on boundary faces), as in Fluent. The geometry routine
checks this against the cell centroids and repairs left-handed faces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Fluent boundary-condition type codes (decimal) <-> names
BC_CODES = {
    2: "interior", 3: "wall", 4: "pressure-inlet", 5: "pressure-outlet", 7: "symmetry",
    8: "periodic-shadow", 9: "pressure-far-field", 10: "velocity-inlet", 12: "periodic",
    14: "fan", 20: "mass-flow-inlet", 24: "interface", 31: "parent", 36: "outflow", 37: "axis",
}
BC_NAMES = {v: k for k, v in BC_CODES.items()}


@dataclass
class Zone:
    id: int
    name: str
    type: str  # Fluent zone type: interior, wall, velocity-inlet, ..., fluid, solid
    kind: str  # 'face' or 'cell'
    first: int = 0  # 0-based first index in the face/cell array
    last: int = -1  # 0-based, inclusive

    @property
    def count(self) -> int:
        return self.last - self.first + 1


@dataclass
class UnstructuredMesh:
    dim: int
    nodes: np.ndarray  # (n_nodes, dim)
    face_ptr: np.ndarray  # (n_faces+1,)
    face_nodes: np.ndarray  # concatenated 0-based node indices
    owner: np.ndarray  # (n_faces,) 0-based
    neighbour: np.ndarray  # (n_faces,) 0-based, -1 on boundary faces
    face_zone: np.ndarray  # (n_faces,) zone id
    n_cells: int
    cell_zone: np.ndarray  # (n_cells,) zone id
    zones: Dict[int, Zone] = field(default_factory=dict)
    name: str = "mesh"
    # translational periodic pairs: [{"zone_a", "zone_b", "faces_a", "faces_b", "translation"}],
    # faces_b[i] is the image of faces_a[i] shifted by translation (x_b = x_a + T)
    periodic: List[dict] = field(default_factory=list)

    @property
    def n_faces(self) -> int:
        return self.owner.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    def face_zones(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.kind == "face"]

    def periodic_zone_ids(self) -> set:
        return {p["zone_a"] for p in self.periodic} | {p["zone_b"] for p in self.periodic}

    def boundary_zones(self) -> List[Zone]:
        """Face zones that need a boundary condition (not interior, not paired periodic)."""
        per = self.periodic_zone_ids()
        return [z for z in self.face_zones() if z.type != "interior" and z.id not in per]

    def cell_zones(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.kind == "cell"]

    def zone_by_name(self, name: str) -> Optional[Zone]:
        for z in self.zones.values():
            if z.name == name:
                return z
        return None

    def faces_of_zone(self, zid: int) -> np.ndarray:
        return np.nonzero(self.face_zone == zid)[0]

    def summary(self) -> dict:
        nb = int((self.neighbour < 0).sum())
        return {
            "name": self.name, "dim": self.dim, "nodes": self.n_nodes, "faces": self.n_faces,
            "interior_faces": self.n_faces - nb, "boundary_faces": nb, "cells": self.n_cells,
            "zones": [{"id": z.id, "name": z.name, "type": z.type, "kind": z.kind, "count": z.count}
                      for z in sorted(self.zones.values(), key=lambda z: z.id)],
        }

    def bounding_box(self):
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def cell_polygons_2d(self):
        """Ordered node loops of each 2-D cell (for plotting). Returns (ptr, nodes)."""
        if self.dim != 2:
            raise ValueError("cell polygons are only defined for 2-D meshes")
        n0 = self.face_nodes[self.face_ptr[:-1]]
        n1 = self.face_nodes[self.face_ptr[:-1] + 1]
        # S = (dy, -dx) points out of the owner, i.e. to the right of n0 -> n1, so
        # the owner lies on the left: n0 -> n1 runs counter-clockwise around it.
        cells = np.concatenate([self.owner, self.neighbour[self.neighbour >= 0]])
        a = np.concatenate([n0, n1[self.neighbour >= 0]])
        b = np.concatenate([n1, n0[self.neighbour >= 0]])
        order = np.argsort(cells, kind="stable")
        cells, a, b = cells[order], a[order], b[order]
        counts = np.bincount(cells, minlength=self.n_cells)
        ptr = np.concatenate([[0], np.cumsum(counts)])
        loops = np.empty(a.shape[0], dtype=np.int64)
        for c in range(self.n_cells):
            s, e = ptr[c], ptr[c + 1]
            nxt = dict(zip(a[s:e].tolist(), b[s:e].tolist()))
            start = a[s]
            cur = start
            k = s
            for _ in range(e - s):
                loops[k] = cur
                k += 1
                cur = nxt.get(cur, start)
        return ptr, loops


@dataclass
class Geometry:
    face_centre: np.ndarray  # (nf, dim)
    face_area_vec: np.ndarray  # (nf, dim), owner -> neighbour / outward
    face_area: np.ndarray  # (nf,)
    cell_centre: np.ndarray  # (nc, dim)
    cell_volume: np.ndarray  # (nc,)
    flipped_faces: int = 0

    @property
    def total_volume(self) -> float:
        return float(self.cell_volume.sum())


def _face_geometry(mesh: UnstructuredMesh):
    X = mesh.nodes
    nf = mesh.n_faces
    ptr, fn = mesh.face_ptr, mesh.face_nodes
    counts = np.diff(ptr)
    dim = mesh.dim
    fc = np.zeros((nf, dim))
    S = np.zeros((nf, dim))
    if dim == 2:
        if np.any(counts != 2):
            raise ValueError("2-D meshes must have 2-node faces")
        a = X[fn[ptr[:-1]]]
        b = X[fn[ptr[:-1] + 1]]
        fc = 0.5 * (a + b)
        d = b - a
        S = np.stack([d[:, 1], -d[:, 0]], axis=1)  # (n1 - n0) x z
        return fc, S
    for k in np.unique(counts):
        idx = np.nonzero(counts == k)[0]
        nodes = fn[ptr[idx][:, None] + np.arange(k)[None, :]]  # (m, k)
        P = X[nodes]  # (m, k, 3)
        xc = P.mean(axis=1)
        Pn = np.roll(P, -1, axis=1)
        tri = 0.5 * np.cross(P - xc[:, None, :], Pn - xc[:, None, :])  # (m, k, 3)
        Sk = tri.sum(axis=1)
        nhat = Sk / np.maximum(np.linalg.norm(Sk, axis=1, keepdims=True), 1e-300)
        w = np.einsum("mkd,md->mk", tri, nhat)
        cent = (xc[:, None, :] + P + Pn) / 3.0
        wsum = w.sum(axis=1, keepdims=True)
        fc[idx] = (w[:, :, None] * cent).sum(axis=1) / np.where(np.abs(wsum) > 0, wsum, 1.0)
        S[idx] = Sk
    return fc, S


def compute_geometry(mesh: UnstructuredMesh, repair: bool = True) -> Geometry:
    dim = mesh.dim
    nc = mesh.n_cells
    own, nei = mesh.owner, mesh.neighbour
    interior = nei >= 0
    fc, S = _face_geometry(mesh)
    # approximate centres: average of face centres
    cnt = np.bincount(own, minlength=nc) + np.bincount(nei[interior], minlength=nc)
    ca = np.zeros((nc, dim))
    for d in range(dim):
        ca[:, d] = (np.bincount(own, fc[:, d], minlength=nc)
                    + np.bincount(nei[interior], fc[interior, d], minlength=nc))
    ca /= np.maximum(cnt, 1)[:, None]
    flipped = 0
    if repair:
        ref = np.where(interior[:, None], ca[np.where(interior, nei, 0)] - ca[own], fc - ca[own])
        bad = np.einsum("fd,fd->f", S, ref) < 0
        flipped = int(bad.sum())
        S[bad] *= -1.0
    # pyramid decomposition about the approximate centre
    vol = np.zeros(nc)
    mom = np.zeros((nc, dim))
    frac = 2.0 / 3.0 if dim == 2 else 0.75
    for cells, sign, mask in ((own, 1.0, slice(None)), (nei[interior], -1.0, interior)):
        f_c = fc[mask]
        s = S[mask] * sign
        h = f_c - ca[cells]
        v = np.einsum("fd,fd->f", s, h) / dim
        cen = ca[cells] + frac * h
        vol += np.bincount(cells, v, minlength=nc)
        for d in range(dim):
            mom[:, d] += np.bincount(cells, v * cen[:, d], minlength=nc)
    if np.any(vol <= 0):
        raise ValueError(f"{int((vol <= 0).sum())} cells have non-positive volume")
    cc = mom / vol[:, None]
    return Geometry(face_centre=fc, face_area_vec=S, face_area=np.linalg.norm(S, axis=1),
                    cell_centre=cc, cell_volume=vol, flipped_faces=flipped)


def split_zone(mesh: UnstructuredMesh, zone: str, where, new_name: str, new_type: Optional[str] = None
               ) -> UnstructuredMesh:
    """Move the faces of ``zone`` whose centres satisfy ``where(xc) -> bool[]`` into a new zone."""
    z = mesh.zone_by_name(zone)
    if z is None:
        raise ValueError(f"unknown zone {zone!r}")
    faces = mesh.faces_of_zone(z.id)
    fc, _ = _face_geometry(mesh)
    sel = np.asarray(where(fc[faces]), dtype=bool)
    if not sel.any():
        raise ValueError(f"no face of {zone!r} matches the split condition")
    if sel.all():
        z.name = new_name
        z.type = new_type or z.type
        return mesh
    nid = max(mesh.zones) + 1
    fz = mesh.face_zone.copy()
    fz[faces[sel]] = nid
    zones = dict(mesh.zones)
    zones[nid] = Zone(nid, new_name, new_type or z.type, "face")
    m = UnstructuredMesh(dim=mesh.dim, nodes=mesh.nodes, face_ptr=mesh.face_ptr, face_nodes=mesh.face_nodes,
                         owner=mesh.owner, neighbour=mesh.neighbour, face_zone=fz, n_cells=mesh.n_cells,
                         cell_zone=mesh.cell_zone, zones=zones, name=mesh.name, periodic=list(mesh.periodic))
    return sort_faces_by_zone(m)


def make_periodic(mesh: UnstructuredMesh, zone_a: str, zone_b: str, translation=None,
                  tol: float = 1e-6) -> UnstructuredMesh:
    """Pair two boundary zones as translational periodic (``zone_b = zone_a + T``).

    Faces are matched geometrically (face centres within ``tol`` x the local
    face size after the shift). ``translation`` defaults to the difference of
    the two zones' face-centre means. Zone types become ``periodic`` /
    ``periodic-shadow`` as in Fluent. Returns the mesh (modified in place).
    """
    from scipy.spatial import cKDTree

    za, zb = mesh.zone_by_name(zone_a), mesh.zone_by_name(zone_b)
    if za is None or zb is None:
        raise ValueError(f"unknown zone(s): {zone_a!r}, {zone_b!r}")
    fa, fb = mesh.faces_of_zone(za.id), mesh.faces_of_zone(zb.id)
    if fa.size != fb.size:
        raise ValueError(f"periodic zones must have the same number of faces ({fa.size} vs {fb.size})")
    fc, S = _face_geometry(mesh)
    T = np.asarray(translation, dtype=float) if translation is not None else fc[fb].mean(0) - fc[fa].mean(0)
    dist, idx = cKDTree(fc[fb]).query(fc[fa] + T)
    size = np.linalg.norm(S[fa], axis=1) ** (1.0 / max(mesh.dim - 1, 1))
    if np.any(dist > tol * np.maximum(size, 1e-300) + 1e-12) or np.unique(idx).size != idx.size:
        raise ValueError(f"faces of {zone_a!r} and {zone_b!r} do not match under translation {T}")
    za.type, zb.type = "periodic", "periodic-shadow"
    mesh.periodic = [p for p in mesh.periodic if p["zone_a"] not in (za.id, zb.id)] + [
        {"zone_a": za.id, "zone_b": zb.id, "faces_a": fa, "faces_b": fb[idx], "translation": T}]
    return mesh


# --------------------------------------------------------------------------
# Generators (used for tests, demos and for exporting FlowTrace grids to Fluent)
# --------------------------------------------------------------------------

def cartesian_mesh_2d(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
                      names=("left", "right", "bottom", "top"),
                      types=("wall", "wall", "wall", "wall"), triangulate: bool = False,
                      jitter: float = 0.0, seed: int = 0, x0: float = 0.0, y0: float = 0.0,
                      warp: float = 0.0, xs=None, ys=None) -> UnstructuredMesh:
    """Structured quad (or triangle) mesh of a rectangle with four named boundary zones.

    ``jitter``: random interior-node perturbation (fraction of the spacing; a
    new random pattern on every grid, i.e. *not* a systematic refinement).
    ``warp``: smooth sinusoidal distortion of the interior (fraction of the
    domain size); refining a warped mesh is a systematic refinement, which is
    what order-of-accuracy studies need. ``xs``/``ys``: explicit (graded)
    node coordinates along x and y, overriding nx/ny/lx/ly.
    """
    if xs is not None:
        xs = np.asarray(xs, dtype=float)
        nx, lx, x0 = xs.size - 1, float(xs[-1] - xs[0]), 0.0
    else:
        xs = x0 + np.linspace(0, lx, nx + 1)
    if ys is not None:
        ys = np.asarray(ys, dtype=float)
        ny, ly, y0 = ys.size - 1, float(ys[-1] - ys[0]), 0.0
    else:
        ys = y0 + np.linspace(0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if jitter:
        rng = np.random.default_rng(seed)
        hx, hy = lx / nx, ly / ny
        X[1:-1, 1:-1] += rng.uniform(-jitter, jitter, (nx - 1, ny - 1)) * hx
        Y[1:-1, 1:-1] += rng.uniform(-jitter, jitter, (nx - 1, ny - 1)) * hy
    if warp:
        sx = (X - xs[0]) / (xs[-1] - xs[0])
        sy = (Y - ys[0]) / (ys[-1] - ys[0])
        d = warp * np.sin(2 * np.pi * sx) * np.sin(2 * np.pi * sy)
        X = X + d * (xs[-1] - xs[0])
        Y = Y + d * (ys[-1] - ys[0])
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)

    def nid(i, j):
        return i * (ny + 1) + j

    if not triangulate:
        def cid(i, j):
            return i * ny + j

        n_cells = nx * ny
        faces, owners, neighs, zones = [], [], [], []
        # interior vertical faces (x = const), between cells (i-1, j) and (i, j)
        I, J = np.meshgrid(np.arange(1, nx), np.arange(ny), indexing="ij")
        I, J = I.ravel(), J.ravel()
        # S = (n1 - n0) x z = (dy, -dx) must point from owner (i-1) to neighbour (i): +x
        faces.append(np.stack([nid(I, J), nid(I, J + 1)], axis=1))
        owners.append(cid(I - 1, J))
        neighs.append(cid(I, J))
        zones.append(np.full(I.size, 3))
        I, J = np.meshgrid(np.arange(nx), np.arange(1, ny), indexing="ij")
        I, J = I.ravel(), J.ravel()
        faces.append(np.stack([nid(I + 1, J), nid(I, J)], axis=1))  # S -> +y
        owners.append(cid(I, J - 1))
        neighs.append(cid(I, J))
        zones.append(np.full(I.size, 3))
        j = np.arange(ny)
        i = np.arange(nx)
        bdef = {
            "left": (np.stack([nid(0, j + 1), nid(0, j)], axis=1), cid(0, j)),  # S -> -x
            "right": (np.stack([nid(nx, j), nid(nx, j + 1)], axis=1), cid(nx - 1, j)),  # S -> +x
            "bottom": (np.stack([nid(i, 0), nid(i + 1, 0)], axis=1), cid(i, 0)),  # S -> -y
            "top": (np.stack([nid(i + 1, ny), nid(i, ny)], axis=1), cid(i, ny - 1)),  # S -> +y
        }
    else:
        # each quad split along its diagonal into lower-right / upper-left triangles
        def cid(i, j, t):
            return 2 * (i * ny + j) + t

        n_cells = 2 * nx * ny
        faces, owners, neighs, zones = [], [], [], []
        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        I, J = I.ravel(), J.ravel()
        # diagonal from (i,j) to (i+1,j+1): tri 0 below-right, tri 1 above-left
        faces.append(np.stack([nid(I, J), nid(I + 1, J + 1)], axis=1))  # S=(dy,-dx) -> lower-right
        owners.append(cid(I, J, 1))
        neighs.append(cid(I, J, 0))
        zones.append(np.full(I.size, 3))
        I, J = np.meshgrid(np.arange(1, nx), np.arange(ny), indexing="ij")
        I, J = I.ravel(), J.ravel()
        faces.append(np.stack([nid(I, J), nid(I, J + 1)], axis=1))  # vertical, -> +x
        owners.append(cid(I - 1, J, 0))
        neighs.append(cid(I, J, 1))
        zones.append(np.full(I.size, 3))
        I, J = np.meshgrid(np.arange(nx), np.arange(1, ny), indexing="ij")
        I, J = I.ravel(), J.ravel()
        faces.append(np.stack([nid(I + 1, J), nid(I, J)], axis=1))  # horizontal, -> +y
        owners.append(cid(I, J - 1, 1))
        neighs.append(cid(I, J, 0))
        zones.append(np.full(I.size, 3))
        j = np.arange(ny)
        i = np.arange(nx)
        bdef = {
            "left": (np.stack([nid(0, j + 1), nid(0, j)], axis=1), cid(0, j, 1)),
            "right": (np.stack([nid(nx, j), nid(nx, j + 1)], axis=1), cid(nx - 1, j, 0)),
            "bottom": (np.stack([nid(i, 0), nid(i + 1, 0)], axis=1), cid(i, 0, 0)),
            "top": (np.stack([nid(i + 1, ny), nid(i, ny)], axis=1), cid(i, ny - 1, 1)),
        }
    zone_objs = {2: Zone(2, "fluid", "fluid", "cell", 0, n_cells - 1)}
    n_int = sum(len(o) for o in owners)
    zone_objs[3] = Zone(3, "interior-fluid", "interior", "face", 0, n_int - 1)
    start = n_int
    for k, side in enumerate(("left", "right", "bottom", "top")):
        f, o = bdef[side]
        zid = 4 + k
        faces.append(f)
        owners.append(o)
        neighs.append(np.full(len(o), -1))
        zones.append(np.full(len(o), zid))
        zone_objs[zid] = Zone(zid, names[k], types[k], "face", start, start + len(o) - 1)
        start += len(o)
    fnodes = np.concatenate(faces).astype(np.int64)
    return UnstructuredMesh(
        dim=2, nodes=nodes,
        face_ptr=np.arange(0, 2 * fnodes.shape[0] + 1, 2, dtype=np.int64),
        face_nodes=fnodes.ravel(),
        owner=np.concatenate(owners).astype(np.int64),
        neighbour=np.concatenate(neighs).astype(np.int64),
        face_zone=np.concatenate(zones).astype(np.int64),
        n_cells=n_cells, cell_zone=np.full(n_cells, 2, dtype=np.int64), zones=zone_objs,
        name=f"rect{nx}x{ny}{'-tri' if triangulate else ''}",
    )


def annulus_mesh(r_in: float, r_out: float, nr: int, nt: int, names=("inner", "outer"),
                 types=("wall", "wall"), rs=None) -> UnstructuredMesh:
    """Polar quad mesh of the annulus r_in < r < r_out (full circle, nt cells around).

    ``rs``: explicit radial node positions (graded), overriding nr/r_in/r_out.
    """
    rs = np.asarray(rs, dtype=float) if rs is not None else np.linspace(r_in, r_out, nr + 1)
    nr = rs.size - 1
    th = np.arange(nt) * 2 * np.pi / nt
    R, T = np.meshgrid(rs, th, indexing="ij")
    nodes = np.stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()], axis=1)

    def nid(i, j):
        return i * nt + (j % nt)

    def cid(i, j):
        return i * nt + (j % nt)

    faces, own, nei, zone = [], [], [], []
    I, J = np.meshgrid(np.arange(nr), np.arange(nt), indexing="ij")
    I, J = I.ravel(), J.ravel()
    faces.append(np.stack([nid(I, J), nid(I + 1, J)], axis=1))  # S = -e_theta: owner at larger theta
    own.append(cid(I, J))
    nei.append(cid(I, J - 1))
    zone.append(np.full(I.size, 3))
    I, J = np.meshgrid(np.arange(1, nr), np.arange(nt), indexing="ij")
    I, J = I.ravel(), J.ravel()
    faces.append(np.stack([nid(I, J), nid(I, J + 1)], axis=1))  # S = +e_r: owner at smaller r
    own.append(cid(I - 1, J))
    nei.append(cid(I, J))
    zone.append(np.full(I.size, 3))
    j = np.arange(nt)
    faces.append(np.stack([nid(0, j + 1), nid(0, j)], axis=1))  # inner boundary, S -> -e_r
    own.append(cid(0, j))
    nei.append(np.full(nt, -1))
    zone.append(np.full(nt, 4))
    faces.append(np.stack([nid(nr, j), nid(nr, j + 1)], axis=1))  # outer boundary, S -> +e_r
    own.append(cid(nr - 1, j))
    nei.append(np.full(nt, -1))
    zone.append(np.full(nt, 5))
    fn = np.concatenate(faces).astype(np.int64)
    n_cells = nr * nt
    zones = {2: Zone(2, "fluid", "fluid", "cell", 0, n_cells - 1), 3: Zone(3, "interior-fluid", "interior", "face"),
             4: Zone(4, names[0], types[0], "face"), 5: Zone(5, names[1], types[1], "face")}
    m = UnstructuredMesh(dim=2, nodes=nodes, face_ptr=np.arange(0, 2 * fn.shape[0] + 1, 2, dtype=np.int64),
                         face_nodes=fn.ravel(), owner=np.concatenate(own).astype(np.int64),
                         neighbour=np.concatenate(nei).astype(np.int64), face_zone=np.concatenate(zone).astype(np.int64),
                         n_cells=n_cells, cell_zone=np.full(n_cells, 2, dtype=np.int64), zones=zones,
                         name=f"annulus{nr}x{nt}")
    return sort_faces_by_zone(m)


def set_cell_zone(mesh: UnstructuredMesh, where, name: str, type: str = "fluid") -> UnstructuredMesh:
    """Move the cells whose centres satisfy ``where(xc)`` into a new cell zone.

    Cells are renumbered so every cell zone is contiguous (needed by the Fluent
    writer); returns a new mesh.
    """
    geo = compute_geometry(mesh)
    sel = np.asarray(where(geo.cell_centre), dtype=bool)
    if not sel.any():
        raise ValueError("no cell matches the zone condition")
    nid = max(mesh.zones) + 1
    cz = mesh.cell_zone.copy()
    cz[sel] = nid
    zones = dict(mesh.zones)
    zones[nid] = Zone(nid, name, type, "cell")
    return _renumber_cells(mesh, cz, zones)


def _renumber_cells(mesh: UnstructuredMesh, cell_zone: np.ndarray, zones: Dict[int, Zone]) -> UnstructuredMesh:
    order = np.argsort(cell_zone, kind="stable")  # new -> old
    new_of_old = np.empty_like(order)
    new_of_old[order] = np.arange(order.size)
    cz = cell_zone[order]
    zones = dict(zones)
    for zid in np.unique(cz):
        w = np.nonzero(cz == zid)[0]
        z = zones[int(zid)]
        zones[int(zid)] = Zone(z.id, z.name, z.type, "cell", int(w[0]), int(w[-1]))
    zones = {k: z for k, z in zones.items() if z.kind != "cell" or k in set(np.unique(cz).tolist())}
    nei = np.where(mesh.neighbour >= 0, new_of_old[np.maximum(mesh.neighbour, 0)], -1)
    return UnstructuredMesh(dim=mesh.dim, nodes=mesh.nodes, face_ptr=mesh.face_ptr, face_nodes=mesh.face_nodes,
                            owner=new_of_old[mesh.owner], neighbour=nei, face_zone=mesh.face_zone,
                            n_cells=mesh.n_cells, cell_zone=cz, zones=zones, name=mesh.name,
                            periodic=list(mesh.periodic))


def extrude_to_3d(mesh2d: UnstructuredMesh, nz: int, lz: float,
                  front=("front", "symmetry"), back=("back", "symmetry")) -> UnstructuredMesh:
    """Extrude a 2-D mesh into prisms/hexes (for 3-D tests and Fluent export)."""
    if mesh2d.dim != 2:
        raise ValueError("need a 2-D mesh")
    nn2, nc2 = mesh2d.n_nodes, mesh2d.n_cells
    zs = np.linspace(0.0, lz, nz + 1)
    nodes = np.concatenate([np.column_stack([mesh2d.nodes, np.full(nn2, z)]) for z in zs])

    def node(n, k):
        return n + k * nn2

    def cell(c, k):
        return c + k * nc2

    faces, own, nei, fz = [], [], [], []
    a = mesh2d.face_nodes[mesh2d.face_ptr[:-1]]
    b = mesh2d.face_nodes[mesh2d.face_ptr[:-1] + 1]
    for k in range(nz):
        # side faces: (a,k) -> (b,k) -> (b,k+1) -> (a,k+1); right-hand normal = (b-a) x z
        faces.append(np.stack([node(a, k), node(b, k), node(b, k + 1), node(a, k + 1)], axis=1))
        own.append(cell(mesh2d.owner, k))
        nei.append(np.where(mesh2d.neighbour >= 0, cell(np.maximum(mesh2d.neighbour, 0), k), -1))
        fz.append(mesh2d.face_zone)
    quad_faces = np.concatenate(faces)
    ptr_list = [np.full(quad_faces.shape[0], 4)]
    node_list = [quad_faces.ravel()]
    own_l, nei_l, fz_l = [np.concatenate(own)], [np.concatenate(nei)], [np.concatenate(fz)]
    # cap faces between layers (interior) and at both ends (boundaries)
    ptr2, loops2 = mesh2d.cell_polygons_2d()
    zmax = max(mesh2d.zones) + 1
    zones = dict(mesh2d.zones)
    int_zone = next(z.id for z in mesh2d.face_zones() if z.type == "interior")
    for k in range(nz + 1):
        cnt = np.diff(ptr2)
        nl = node(loops2, k)
        # loops are counter-clockwise seen from +z, so their normal is +z: owner below
        if k == 0:
            lst = [nl[ptr2[c]:ptr2[c + 1]][::-1] for c in range(nc2)]
            o, n_, z = cell(np.arange(nc2), 0), np.full(nc2, -1), zmax
        elif k == nz:
            lst = [nl[ptr2[c]:ptr2[c + 1]] for c in range(nc2)]
            o, n_, z = cell(np.arange(nc2), nz - 1), np.full(nc2, -1), zmax + 1
        else:
            lst = [nl[ptr2[c]:ptr2[c + 1]] for c in range(nc2)]
            o, n_, z = cell(np.arange(nc2), k - 1), cell(np.arange(nc2), k), int_zone
        ptr_list.append(cnt)
        node_list.append(np.concatenate(lst))
        own_l.append(o)
        nei_l.append(n_)
        fz_l.append(np.full(nc2, z))
    for z in mesh2d.cell_zones():
        idx = np.nonzero(np.tile(mesh2d.cell_zone, nz) == z.id)[0]
        zones[z.id] = Zone(z.id, z.name, z.type, "cell", int(idx[0]), int(idx[-1]))
    zones[zmax] = Zone(zmax, back[0], back[1], "face")
    zones[zmax + 1] = Zone(zmax + 1, front[0], front[1], "face")
    counts = np.concatenate(ptr_list)
    m = UnstructuredMesh(
        dim=3, nodes=nodes, face_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        face_nodes=np.concatenate(node_list).astype(np.int64),
        owner=np.concatenate(own_l).astype(np.int64), neighbour=np.concatenate(nei_l).astype(np.int64),
        face_zone=np.concatenate(fz_l).astype(np.int64), n_cells=nc2 * nz,
        cell_zone=np.tile(mesh2d.cell_zone, nz), zones=zones, name=mesh2d.name + f"-x{nz}",
    )
    return sort_faces_by_zone(m)


def extract_cells(mesh: UnstructuredMesh, keep: np.ndarray, interface_type: str = "wall"):
    """Sub-mesh of the cells where ``keep`` is True.

    Faces between kept and dropped cells become boundary faces of new zones
    named ``interface-<dropped cell zone>`` (oriented out of the kept cell).
    Returns ``(submesh, face_map, cell_map)``: for every sub-mesh face its
    index in ``mesh`` (+1 when the face was flipped, as ``-(index+1)``) and for
    every sub-mesh cell its index in ``mesh``.
    """
    keep = np.asarray(keep, dtype=bool)
    if mesh.periodic:
        raise NotImplementedError("extracting cells from a mesh with periodic zones is not supported")
    cmap = np.nonzero(keep)[0]
    new_id = -np.ones(mesh.n_cells, dtype=np.int64)
    new_id[cmap] = np.arange(cmap.size)
    own, nei = mesh.owner, mesh.neighbour
    ko = keep[own]
    kn = np.where(nei >= 0, keep[np.maximum(nei, 0)], False)
    zones = {zid: z for zid, z in mesh.zones.items() if z.kind == "face"}
    for z in mesh.cell_zones():
        if np.any(keep[mesh.cell_zone == z.id]):
            zones[z.id] = Zone(z.id, z.name, z.type, "cell")
    next_id = max(mesh.zones) + 1
    iface_zone = {}
    faces, f_own, f_nei, f_zone, f_map = [], [], [], [], []
    ptr = mesh.face_ptr
    for f in range(mesh.n_faces):
        a, b = ko[f], kn[f]
        if not (a or b):
            continue
        nodes = mesh.face_nodes[ptr[f]:ptr[f + 1]]
        if a and (b or nei[f] < 0):
            faces.append(nodes)
            f_own.append(new_id[own[f]])
            f_nei.append(new_id[nei[f]] if nei[f] >= 0 else -1)
            f_zone.append(mesh.face_zone[f])
            f_map.append(f)
            continue
        other = nei[f] if a else own[f]
        oz = mesh.zones[int(mesh.cell_zone[other])]
        if oz.id not in iface_zone:
            iface_zone[oz.id] = next_id
            zones[next_id] = Zone(next_id, f"interface-{oz.name}", interface_type, "face")
            next_id += 1
        if a:
            faces.append(nodes)
            f_own.append(new_id[own[f]])
            f_map.append(f)
        else:  # keep the cell on the neighbour side: flip so the normal points out of it
            faces.append(nodes[::-1])
            f_own.append(new_id[nei[f]])
            f_map.append(-(f + 1))
        f_nei.append(-1)
        f_zone.append(iface_zone[oz.id])
    counts = np.array([len(x) for x in faces], dtype=np.int64)
    present = set(f_zone)
    zones = {k: z for k, z in zones.items() if z.kind == "cell" or k in present}
    sub = UnstructuredMesh(dim=mesh.dim, nodes=mesh.nodes, face_ptr=np.concatenate([[0], np.cumsum(counts)]),
                           face_nodes=np.concatenate(faces).astype(np.int64), owner=np.array(f_own, dtype=np.int64),
                           neighbour=np.array(f_nei, dtype=np.int64), face_zone=np.array(f_zone, dtype=np.int64),
                           n_cells=cmap.size, cell_zone=mesh.cell_zone[cmap], zones=zones, name=mesh.name + "-fluid")
    perm = _zone_sort_perm(sub)
    sub = sort_faces_by_zone(sub)
    for z in sub.cell_zones():
        w = np.nonzero(sub.cell_zone == z.id)[0]
        z.first, z.last = int(w[0]), int(w[-1])
    return sub, np.array(f_map, dtype=np.int64)[perm], cmap


def _zone_sort_perm(mesh: UnstructuredMesh) -> np.ndarray:
    zorder = sorted(mesh.face_zones(), key=lambda z: (z.type != "interior", z.id))
    rank = {z.id: r for r, z in enumerate(zorder)}
    key = np.array([rank[z] for z in mesh.face_zone])
    return np.argsort(key, kind="stable")


def sort_faces_by_zone(mesh: UnstructuredMesh) -> UnstructuredMesh:
    """Reorder faces so each face zone is contiguous (interior zones first), as Fluent requires."""
    zorder = sorted(mesh.face_zones(), key=lambda z: (z.type != "interior", z.id))
    perm = _zone_sort_perm(mesh)
    counts = np.diff(mesh.face_ptr)[perm]
    starts = mesh.face_ptr[:-1][perm]
    idx = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)]) if len(perm) else np.zeros(0, int)
    fz = mesh.face_zone[perm]
    zones = dict(mesh.zones)
    for z in zorder:
        where = np.nonzero(fz == z.id)[0]
        zones[z.id] = Zone(z.id, z.name, z.type, "face", int(where[0]), int(where[-1]))
    new_of_old = np.empty(perm.size, dtype=np.int64)
    new_of_old[perm] = np.arange(perm.size)
    periodic = [dict(p, faces_a=new_of_old[p["faces_a"]], faces_b=new_of_old[p["faces_b"]]) for p in mesh.periodic]
    return UnstructuredMesh(
        dim=mesh.dim, nodes=mesh.nodes, face_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        face_nodes=mesh.face_nodes[idx], owner=mesh.owner[perm], neighbour=mesh.neighbour[perm],
        face_zone=fz, n_cells=mesh.n_cells, cell_zone=mesh.cell_zone, zones=zones, name=mesh.name,
        periodic=periodic,
    )
