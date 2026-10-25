import gzip

import numpy as np
import pytest

from flowtrace.fluent.convert import structured_mesh
from flowtrace.mesh.fluent_msh import read_fluent_mesh, write_fluent_mesh
from flowtrace.mesh.unstructured import cartesian_mesh_2d, compute_geometry, extrude_to_3d
from flowtrace.structured.case import backward_facing_step
from flowtrace.structured.solver import MACSolver
from flowtrace.hardware.backend import get_backend


def _same(a, b):
    assert (a.dim, a.n_cells, a.n_faces) == (b.dim, b.n_cells, b.n_faces)
    ga, gb = compute_geometry(a), compute_geometry(b)
    assert gb.flipped_faces == 0
    assert np.isclose(ga.total_volume, gb.total_volume)
    assert np.allclose(np.sort(ga.cell_volume), np.sort(gb.cell_volume))
    za = sorted((z.name, z.type, z.count) for z in a.zones.values())
    zb = sorted((z.name, z.type, z.count) for z in b.zones.values())
    assert za == zb


@pytest.mark.parametrize("tri", [False, True])
def test_geometry_orientation_and_volume(tri):
    m = cartesian_mesh_2d(6, 5, 2.0, 1.5, triangulate=tri, jitter=0.25)
    g = compute_geometry(m)
    assert g.flipped_faces == 0
    assert np.isclose(g.total_volume, 3.0)
    ptr, loops = m.cell_polygons_2d()
    X = m.nodes
    for c in range(m.n_cells):
        P = X[loops[ptr[c]:ptr[c + 1]]]
        area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        assert np.isclose(area, g.cell_volume[c])
    m3 = extrude_to_3d(m, 3, 0.5)
    g3 = compute_geometry(m3)
    assert g3.flipped_faces == 0 and np.isclose(g3.total_volume, 1.5)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("ext", [".msh", ".msh.gz"])
@pytest.mark.parametrize("dim3", [False, True])
def test_fluent_roundtrip(tmp_path, binary, ext, dim3):
    m = cartesian_mesh_2d(5, 4, 2.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "symmetry"), triangulate=True, jitter=0.2)
    if dim3:
        m = extrude_to_3d(m, 2, 0.4)
    p = tmp_path / f"m{ext}"
    write_fluent_mesh(m, str(p), binary=binary)
    _same(m, read_fluent_mesh(str(p)))


def test_reader_accepts_boundary_faces_with_c0_zero(tmp_path):
    # hand-written mesh: 2 quads, one boundary face stored as "n0 n1 0 c" (c0 = 0)
    txt = """(0 "hand made")
(2 2)
(10 (0 1 6 0 2))
(10 (1 1 6 1 2)(
0 0
1 0
2 0
0 1
1 1
2 1
))
(12 (0 1 2 0))
(12 (2 1 2 1 3))
(13 (0 1 7 0))
(13 (3 1 1 2 2)(
2 5 1 2
))
(13 (4 2 7 3 2)(
1 2 1 0
3 2 0 2
6 3 2 0
5 6 2 0
4 5 1 0
1 4 0 1
))
(45 (2 fluid fluid)())
(45 (3 interior interior-1)())
(45 (4 wall walls)())
"""
    p = tmp_path / "hand.msh"
    p.write_text(txt)
    m = read_fluent_mesh(str(p))
    g = compute_geometry(m)
    assert m.n_cells == 2 and np.isclose(g.total_volume, 2.0)
    assert np.all(m.owner >= 0)
    assert m.zone_by_name("walls").count == 6


def test_structured_case_to_fluent_mesh(tmp_path):
    c = backward_facing_step(re=100, n_per_h=4)
    s = MACSolver(c, backend=get_backend("numpy"))
    m = structured_mesh(c, s.solid_h)
    g = compute_geometry(m)
    assert g.flipped_faces == 0
    assert np.isclose(g.total_volume, c.lx * c.ly - 1.0)
    names = {z.name: z.type for z in m.face_zones()}
    assert names["left"] == "velocity-inlet" and names["right"] == "pressure-outlet" and names["obstacle"] == "wall"
    p = tmp_path / "step.msh.gz"
    write_fluent_mesh(m, str(p))
    with gzip.open(p, "rb") as fh:
        assert fh.read(2) == b"(0"
    _same(m, read_fluent_mesh(str(p)))
