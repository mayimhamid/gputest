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


def _stored_msh_faces(path):
    """(node tuple, c0) of every face as written in an ASCII .msh file (no FlowTrace reader logic)."""
    import re

    txt = open(path).read()
    out = []
    for m in re.finditer(r"\(13 \((\w+) (\w+) (\w+) (\w+) (\w+)\)\(\n(.*?)\n\)\)", txt, re.S):
        mixed = int(m.group(5), 16) == 0
        for line in m.group(6).split("\n"):
            v = [int(t, 16) for t in line.split()]
            if mixed:
                v = v[1:]
            out.append((tuple(v[:-2]), v[-2]))
    return out


@pytest.mark.parametrize("fmt", ["msh", "cas.h5"])
def test_3d_faces_are_written_in_fluents_orientation(tmp_path, fmt):
    """Fluent stores a 3-D face so that the right-hand normal of its node order points toward c0
    (measured on Fluent-written files: flowtrace/validation/fluent_files.py). Checked on the file
    contents themselves, not through FlowTrace's reader."""
    m = extrude_to_3d(cartesian_mesh_2d(4, 3, jitter=0.2), 2, 0.5)
    p = tmp_path / f"m.{fmt}"
    if fmt == "msh":
        write_fluent_mesh(m, str(p))
        faces = _stored_msh_faces(str(p))
    else:
        pytest.importorskip("h5py")
        from flowtrace.mesh.fluent_h5 import write_case_h5
        from flowtrace.validation.fluent_files import _stored_faces

        write_case_h5(m, str(p))
        nodes, c0, _ = _stored_faces(str(p))
        faces = list(zip(nodes, c0))
    xc = compute_geometry(m).cell_centre
    X = m.nodes
    assert len(faces) == m.n_faces
    for nodes, c0 in faces:
        P = X[np.asarray(nodes) - 1]
        ctr = P.mean(axis=0)
        normal = 0.5 * np.cross(P - ctr, np.roll(P, -1, axis=0) - ctr).sum(axis=0)
        assert np.dot(normal, xc[c0 - 1] - ctr) > 0


def test_reads_a_hand_written_3d_cell_in_fluents_orientation(tmp_path):
    """Unit cube, every face listed with its right-hand normal pointing into the cell (c0), as
    Fluent writes them: it must read without any face needing re-orientation."""
    txt = """(0 "unit hex, Fluent orientation")
(2 3)
(10 (0 1 8 0 3))
(10 (1 1 8 1 3)(
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
))
(12 (0 1 1 0))
(12 (2 1 1 1 4))
(13 (0 1 6 0))
(13 (3 1 6 3 4)(
1 2 3 4 1 0
5 8 7 6 1 0
1 5 6 2 1 0
2 6 7 3 1 0
3 7 8 4 1 0
4 8 5 1 1 0
))
(45 (2 fluid fluid)())
(45 (3 wall walls)())
"""
    p = tmp_path / "hex.msh"
    p.write_text(txt)
    m = read_fluent_mesh(str(p))
    g = compute_geometry(m, repair=False)  # raises if a face points the wrong way
    assert g.flipped_faces == 0 and np.isclose(g.total_volume, 1.0)
    assert np.allclose(g.cell_centre[0], [0.5, 0.5, 0.5])


@pytest.mark.parametrize("damage", ["empty", "text", "half", "no-faces", "binary-half", "node-count", "gzip", "nul"])
def test_malformed_mesh_files_raise_a_clear_error(tmp_path, damage):
    from flowtrace.mesh.fluent_msh import FluentMeshError

    good = tmp_path / "good.msh"
    write_fluent_mesh(cartesian_mesh_2d(4, 3), str(good))
    txt = good.read_text()
    goodb = tmp_path / "goodb.msh"
    write_fluent_mesh(cartesian_mesh_2d(4, 3), str(goodb), binary=True)
    raw = goodb.read_bytes()
    data = {"empty": b"", "text": b"not a mesh\n", "half": txt[:len(txt) // 2].encode(),
            "no-faces": txt[:txt.index("(13 (")].encode(), "binary-half": raw[:len(raw) // 2],
            "node-count": txt.replace("(10 (1 1 14 1 2)(", "(10 (1 1 13 1 2)(").encode(),
            "gzip": b"\x1f\x8b\x08\x00garbage", "nul": b"\x00" * 64}[damage]
    bad = tmp_path / "bad.msh"
    bad.write_bytes(data)
    with pytest.raises(FluentMeshError):
        read_fluent_mesh(str(bad))
    with pytest.raises(FileNotFoundError):
        read_fluent_mesh(str(tmp_path / "missing.msh"))


def test_malformed_cff_files_raise_a_clear_error(tmp_path):
    """HDF5 files that are not complete CFF meshes or data: FluentMeshError naming the file, not a
    KeyError from deep inside h5py; a file that is not HDF5 at all stays an OSError."""
    h5py = pytest.importorskip("h5py")
    from flowtrace.mesh.fluent_h5 import read_case_h5, read_data_h5
    from flowtrace.mesh.fluent_msh import FluentMeshError

    other, partial = tmp_path / "other.cas.h5", tmp_path / "partial.cas.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("x", data=np.arange(3))
    with h5py.File(partial, "w") as f:
        f.create_group("meshes/1").attrs["dimension"] = np.int32(3)
    for p in (other, partial):
        with pytest.raises(FluentMeshError, match=p.name):
            read_case_h5(str(p))
    with pytest.raises(FluentMeshError, match="data file"):
        read_data_h5(str(other))
    junk = tmp_path / "junk.cas.h5"
    junk.write_bytes(b"\x89HDF\r\n\x1a\nnot really HDF5")
    with pytest.raises(OSError):
        read_case_h5(str(junk))
