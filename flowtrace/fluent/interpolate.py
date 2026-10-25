"""Map an interpolation (.ip) point cloud onto the cells of another mesh.

This is what Fluent's *File > Interpolate > Read and Interpolate* does with a
FlowTrace ``.ip`` file; FlowTrace uses the same path to warm-start its own
SIMPLEC solver from a solution computed on a different grid (e.g. the fast
structured MAC solver), which is how the warm-start benefit is measured
without a Fluent licence.

Linear (Delaunay) interpolation inside the convex hull of the source points,
nearest-neighbour outside it (Fluent likewise falls back to the closest
source point).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .export import read_interpolation_file


def interpolate_points(points: np.ndarray, fields: Dict[str, np.ndarray], targets: np.ndarray,
                       method: str = "linear") -> Dict[str, np.ndarray]:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    pts = np.asarray(points, dtype=float)
    tgt = np.asarray(targets, dtype=float)[:, :pts.shape[1]]
    names = list(fields)
    vals = np.column_stack([np.asarray(fields[k], dtype=float) for k in names])
    near = NearestNDInterpolator(pts, vals)
    if method == "nearest":
        out = near(tgt)
    else:
        lin = LinearNDInterpolator(pts, vals)
        out = lin(tgt)
        bad = ~np.all(np.isfinite(out), axis=1)
        if bad.any():
            out[bad] = near(tgt[bad])
    return {k: out[:, i] for i, k in enumerate(names)}


def interpolate_ip_file(path: str, targets: np.ndarray, method: str = "linear") -> Dict[str, np.ndarray]:
    pts, fields = read_interpolation_file(path)
    return interpolate_points(pts, fields, targets, method)
