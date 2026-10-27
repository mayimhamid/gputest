"""Mesh quality metrics with Fluent-style definitions and a Fluent-like summary.

* **Orthogonal quality** (Fluent's definition, 0 worst ... 1 best): for every
  cell the minimum over its faces of
  ``min( A_f . f_i / (|A_f||f_i|),  A_f . c_i / (|A_f||c_i|) )`` with ``A_f``
  the outward face area vector, ``f_i`` the vector from the cell centroid to
  the face centroid and ``c_i`` the vector to the neighbouring cell centroid.
* **Equiangle skewness** (0 best ... 1 degenerate):
  ``max((th_max - th_e)/(180 - th_e), (th_e - th_min)/th_e)`` over the
  interior angles of the cell (2-D) or of its faces (3-D), with ``th_e`` the
  angle of the equiangular polygon (60 deg triangle, 90 deg quad). It is
  defined for the standard element types only: triangles and quads in 2-D;
  tetrahedra, pyramids, prisms and hexahedra (4-6 triangular or quad faces) in
  3-D. Polygonal and polyhedral cells, as Fluent Meshing's poly meshes, are
  judged by orthogonal quality, as Fluent does; their faces may carry collinear
  hanging nodes (180 deg "corners") that the equiangle formula would score as
  degenerate. Their skewness is reported as not applicable.
* **Aspect ratio**: largest over smallest normal distance from the cell
  centroid to its faces (1 for squares, equilateral triangles, cubes).
* **Non-orthogonality**: angle between ``A_f`` and the centroid-to-centroid
  vector (OpenFOAM's definition), per interior face.
* **Volume ratio**: largest neighbour-to-cell volume ratio.

Thresholds follow the Ansys Fluent User's Guide recommendations: minimum
orthogonal quality above 0.01 (0.1 advisable), maximum skewness below 0.95,
aspect ratios above ~100 only inside wall-resolved boundary layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

from .unstructured import UnstructuredMesh, compute_geometry


@dataclass
class QualityReport:
    cells: int
    faces: int
    dim: int
    min_orthogonal_quality: float
    min_orthogonal_quality_cell: int
    max_skewness: float
    max_skewness_cell: int
    max_aspect_ratio: float
    max_aspect_ratio_cell: int
    max_non_orthogonality_deg: float
    mean_non_orthogonality_deg: float
    max_volume_ratio: float
    min_volume: float
    total_volume: float
    left_handed_faces_repaired: int
    histograms: Dict[str, dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    locations: Dict[str, list] = field(default_factory=dict)
    skewness_not_applicable_cells: int = 0  # polygonal / polyhedral cells (see the module docstring)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return not any(w.startswith("CRITICAL") for w in self.warnings)

    def summary(self) -> str:
        """Text in the style of Fluent's *Mesh > Check / Quality* output."""
        loc = self.locations
        L = ["Mesh Quality:", "",
             f"  Minimum Orthogonal Quality = {self.min_orthogonal_quality:.5e}  cell {self.min_orthogonal_quality_cell}"
             f" at {_fmt_loc(loc.get('orthogonal_quality'))}",
             "  (Fluent Meshing reports the inverse: 1 - orthogonal quality)", "",
             (f"  Maximum Equiangle Skewness = {self.max_skewness:.5e}  cell {self.max_skewness_cell}"
              f" at {_fmt_loc(loc.get('skewness'))}" if self.max_skewness == self.max_skewness
              else "  Maximum Equiangle Skewness = n/a (no standard-element cells)")
             + (f"  [not applicable to {self.skewness_not_applicable_cells} polygonal/polyhedral cells]"
                if self.skewness_not_applicable_cells else ""),
             f"  Maximum Aspect Ratio = {self.max_aspect_ratio:.5e}  cell {self.max_aspect_ratio_cell}"
             f" at {_fmt_loc(loc.get('aspect_ratio'))}",
             f"  Non-orthogonality: max {self.max_non_orthogonality_deg:.2f} deg, mean {self.mean_non_orthogonality_deg:.2f} deg",
             f"  Maximum neighbour volume ratio = {self.max_volume_ratio:.4g}",
             f"  Volume: min {self.min_volume:.5e}, total {self.total_volume:.6g}",
             f"  Left-handed faces repaired: {self.left_handed_faces_repaired}", ""]
        if self.warnings:
            L += [f"  {w}" for w in self.warnings]
        else:
            L.append("  No quality problems found (Fluent recommendations).")
        return "\n".join(L)


def _fmt_loc(p):
    if p is None:
        return "-"
    return "(" + ", ".join(f"{v:.5g}" for v in p) + ")"


def _equiangle_batch(P: np.ndarray) -> np.ndarray:
    """Equiangle skewness of m convex polygons with k vertices each, P: (m, k, dim)."""
    a = np.roll(P, 1, axis=1) - P
    b = np.roll(P, -1, axis=1) - P
    cos = np.einsum("mkd,mkd->mk", a, b) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-300)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    k = P.shape[1]
    te = 180.0 * (k - 2) / k
    return np.maximum((ang.max(axis=1) - te) / (180.0 - te), (te - ang.min(axis=1)) / te)


def _skew_polygons(nodes, ptr, idx_list) -> np.ndarray:
    """Equiangle skewness of polygons given as (ptr, node index list), grouped by vertex count."""
    counts = np.diff(ptr)
    out = np.zeros(counts.size)
    for k in np.unique(counts):
        sel = np.nonzero(counts == k)[0]
        verts = idx_list[ptr[sel][:, None] + np.arange(k)[None, :]]
        out[sel] = _equiangle_batch(nodes[verts])
    return out


def mesh_quality(mesh: UnstructuredMesh, bins: int = 10) -> QualityReport:
    geo = compute_geometry(mesh)
    nc, dim = mesh.n_cells, mesh.dim
    xc, fc, S = geo.cell_centre, geo.face_centre, geo.face_area_vec
    own, nei = mesh.owner, mesh.neighbour
    inter = nei >= 0
    Smag = np.linalg.norm(S, axis=1)
    n = S / np.maximum(Smag, 1e-300)[:, None]

    # orthogonal quality: owner side (outward = +n), neighbour side (outward = -n)
    def cosang(u, v):
        return np.einsum("fd,fd->f", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)

    oq = np.ones(nc)
    f_own = cosang(n, fc - xc[own])
    np.minimum.at(oq, own, f_own)
    ni = nei[inter]
    d = xc[ni] - xc[own[inter]]
    c_own = cosang(n[inter], d)
    np.minimum.at(oq, own[inter], c_own)
    f_nei = cosang(-n[inter], fc[inter] - xc[ni])
    np.minimum.at(oq, ni, f_nei)
    np.minimum.at(oq, ni, c_own)  # same angle seen from the neighbour
    oq = np.clip(oq, 0.0, 1.0)

    # aspect ratio from normal distances to the face planes
    hd = np.abs(np.einsum("fd,fd->f", fc - xc[own], n))
    dmin = np.full(nc, np.inf)
    dmax = np.zeros(nc)
    np.minimum.at(dmin, own, hd)
    np.maximum.at(dmax, own, hd)
    hn = np.abs(np.einsum("fd,fd->f", fc[inter] - xc[ni], n[inter]))
    np.minimum.at(dmin, ni, hn)
    np.maximum.at(dmax, ni, hn)
    ar = dmax / np.maximum(dmin, 1e-300)

    # skewness, for the standard element types only (see the module docstring)
    skew = np.zeros(nc)
    faces_per_cell = np.bincount(own, minlength=nc) + np.bincount(ni, minlength=nc)
    if dim == 2:
        ptr, loops = mesh.cell_polygons_2d()
        skew = _skew_polygons(mesh.nodes, ptr, loops)
        standard = faces_per_cell <= 4  # triangles and quads (edges = vertices)
    else:
        nodes_per_face = np.diff(mesh.face_ptr)
        fskew = _skew_polygons(mesh.nodes, mesh.face_ptr, mesh.face_nodes)
        np.maximum.at(skew, own, fskew)
        np.maximum.at(skew, ni, fskew[inter])
        most_nodes = np.zeros(nc, dtype=np.int64)
        np.maximum.at(most_nodes, own, nodes_per_face)
        np.maximum.at(most_nodes, ni, nodes_per_face[inter])
        standard = (most_nodes <= 4) & (faces_per_cell >= 4) & (faces_per_cell <= 6)
    skew_all = skew
    skew = np.where(standard, skew, np.nan)

    # non-orthogonality and volume ratio
    nonorth = np.degrees(np.arccos(np.clip(c_own, -1, 1))) if ni.size else np.zeros(0)
    vol = geo.cell_volume
    vr = np.ones(nc)
    if ni.size:
        r = np.maximum(vol[ni] / vol[own[inter]], vol[own[inter]] / vol[ni])
        np.maximum.at(vr, own[inter], r)
        np.maximum.at(vr, ni, r)

    i_oq, i_ar = int(np.argmin(oq)), int(np.argmax(ar))
    i_sk = int(np.argmax(np.where(standard, skew_all, -1.0))) if standard.any() else 0
    rep = QualityReport(
        cells=nc, faces=mesh.n_faces, dim=dim,
        min_orthogonal_quality=float(oq[i_oq]), min_orthogonal_quality_cell=i_oq,
        max_skewness=float(skew_all[i_sk]) if standard.any() else float("nan"), max_skewness_cell=i_sk,
        max_aspect_ratio=float(ar[i_ar]), max_aspect_ratio_cell=i_ar,
        max_non_orthogonality_deg=float(np.max(nonorth)) if nonorth.size else 0.0,
        mean_non_orthogonality_deg=float(nonorth.mean()) if nonorth.size else 0.0,
        max_volume_ratio=float(np.max(vr)), min_volume=float(np.min(vol)), total_volume=float(vol.sum()),
        left_handed_faces_repaired=geo.flipped_faces,
        locations={"orthogonal_quality": xc[i_oq].tolist(), "skewness": xc[i_sk].tolist(),
                   "aspect_ratio": xc[i_ar].tolist()},
        skewness_not_applicable_cells=int((~standard).sum()),
    )
    for name, arr, rng in (("orthogonal_quality", oq, (0, 1)), ("skewness", skew_all[standard], (0, 1)),
                           ("aspect_ratio", np.log10(np.maximum(ar, 1.0)), (0, max(1.0, float(np.log10(ar.max())) + 1e-9)))):
        h, edges = np.histogram(arr, bins=bins, range=rng)
        rep.histograms[name] = {"counts": h.tolist(), "edges": edges.tolist(),
                                "log10": name == "aspect_ratio"}
    w = rep.warnings
    if rep.min_orthogonal_quality < 0.01:
        w.append(f"CRITICAL: minimum orthogonal quality {rep.min_orthogonal_quality:.3g} < 0.01 (Fluent: unacceptable)")
    elif rep.min_orthogonal_quality < 0.1:
        w.append(f"WARNING: minimum orthogonal quality {rep.min_orthogonal_quality:.3g} < 0.1")
    if rep.max_skewness > 0.95:
        w.append(f"CRITICAL: maximum skewness {rep.max_skewness:.3g} > 0.95 (Fluent: unacceptable)")
    elif rep.max_skewness > 0.85:
        w.append(f"WARNING: maximum skewness {rep.max_skewness:.3g} > 0.85")
    if rep.max_aspect_ratio > 100:
        w.append(f"NOTE: maximum aspect ratio {rep.max_aspect_ratio:.3g} > 100 "
                 "(acceptable only in wall-resolved boundary layers)")
    if rep.max_volume_ratio > 5:
        w.append(f"WARNING: neighbouring cell volumes differ by up to x{rep.max_volume_ratio:.3g} (> 5)")
    if rep.left_handed_faces_repaired:
        w.append(f"NOTE: {rep.left_handed_faces_repaired} left-handed faces were re-oriented on import")
    if rep.skewness_not_applicable_cells:
        w.append(f"NOTE: equiangle skewness not evaluated for {rep.skewness_not_applicable_cells} polygonal/polyhedral "
                 "cells; orthogonal quality applies to them")
    return rep
