"""CFF/HDF5 case & data files, mesh quality, fluent-warmstart CLI."""

import json
import os

import numpy as np
import pytest

from flowtrace.mesh.fluent_msh import read_fluent_mesh
from flowtrace.mesh.quality import mesh_quality
from flowtrace.mesh.unstructured import (cartesian_mesh_2d, compute_geometry, extrude_to_3d, make_periodic,
                                         set_cell_zone)

h5py = pytest.importorskip("h5py")
from flowtrace.mesh.fluent_h5 import (describe_h5, is_h5, read_any_mesh, read_case_h5, read_data_h5,  # noqa: E402
                                      write_case_h5, write_data_h5)

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _same(a, b):
    assert (a.dim, a.n_cells, a.n_faces) == (b.dim, b.n_cells, b.n_faces)
    ga, gb = compute_geometry(a), compute_geometry(b)
    assert gb.flipped_faces == 0
    assert np.allclose(np.sort(ga.cell_volume), np.sort(gb.cell_volume))
    za = sorted((z.name, z.type, z.kind, z.count) for z in a.zones.values())
    zb = sorted((z.name, z.type, z.kind, z.count) for z in b.zones.values())
    assert za == zb


@pytest.mark.parametrize("kind", ["quad", "tri", "3d", "zones", "periodic", "example"])
def test_cas_h5_roundtrip(tmp_path, kind):
    if kind == "example":
        m = read_fluent_mesh(os.path.join(EXAMPLES, "heated_duct_tri.msh"))
    else:
        m = cartesian_mesh_2d(6, 5, 2.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                              types=("velocity-inlet", "pressure-outlet", "wall", "symmetry"),
                              triangulate=kind == "tri", warp=0.03)
    if kind == "3d":
        m = extrude_to_3d(m, 3, 0.5)
    if kind == "zones":
        m = set_cell_zone(m, lambda x: x[:, 0] < 0.7, "heater", "solid")
    if kind == "periodic":
        m = make_periodic(m, "inlet", "outlet")
    p = tmp_path / "case.cas.h5"
    write_case_h5(m, str(p))
    assert is_h5(str(p))
    m2 = read_case_h5(str(p))
    _same(m, m2)
    if kind == "periodic":
        assert len(m2.periodic) == 1 and np.allclose(m2.periodic[0]["translation"], [2.0, 0.0])
    m3, fmt = read_any_mesh(str(p))
    assert fmt == "cff-hdf5" and m3.n_cells == m.n_cells
    # the layout the public VTK reader expects
    with h5py.File(p, "r") as f:
        assert "settings" in f and f["meshes/1"].attrs["dimension"] == m.dim
        assert f["meshes/1/faces/nodes/1/nodes"].dtype == np.uint32
        assert f["meshes/1/faces/nodes/1/nnodes"].dtype == np.int16
        assert int(f["meshes/1/faces/zoneTopology"].attrs["nZones"]) == len(m.face_zones())
    txt = describe_h5(str(p))
    assert "meshes/1/faces/c0" in txt


def test_dat_h5_roundtrip(tmp_path):
    m = set_cell_zone(cartesian_mesh_2d(4, 4), lambda x: x[:, 1] < 0.5, "lower")
    rng = np.random.default_rng(0)
    fields = {"pressure": rng.standard_normal(m.n_cells), "x-velocity": rng.standard_normal(m.n_cells),
              "turb-kinetic-energy": rng.uniform(size=m.n_cells)}
    p = tmp_path / "r.dat.h5"
    write_data_h5(str(p), m, fields, iteration=12)
    with h5py.File(p, "r") as f:
        names = f["results/1/phase-1/cells/fields"][()]
        names = names.decode() if isinstance(names, bytes) else names
        assert names.split(";") == ["SV_P", "SV_U", "SV_K"]
        assert int(f["results/1/phase-1/cells/SV_P"].attrs["nSections"]) == 2
    back = read_data_h5(str(p))
    assert set(back) == set(fields)
    for k in fields:
        assert np.allclose(back[k], fields[k])
    with pytest.raises(ValueError):
        write_data_h5(str(tmp_path / "bad.dat.h5"), m, {"pressure": np.zeros(3)})


def test_reads_fluent_2025_layout(tmp_path):
    """Fluent Meshing 2025 R1 writes scalar attributes as one-element arrays and numbers the
    node-coordinate sections 1..n (each with its own minId/maxId) instead of keying them by
    node-zone id. Both broke the reader on a real 3.2M-cell rocket mesh."""
    m = extrude_to_3d(cartesian_mesh_2d(4, 3), 2, 0.5)
    p = tmp_path / "m.msh.h5"
    write_case_h5(m, str(p))
    with h5py.File(p, "r+") as f:
        def arrayify(obj):
            for k, v in list(obj.attrs.items()):
                if np.ndim(v) == 0 and np.issubdtype(np.asarray(v).dtype, np.integer):
                    obj.attrs[k] = np.asarray([v], dtype=np.uint64)
        f.visititems(lambda name, obj: arrayify(obj))
        arrayify(f["meshes/1"])
        nz = f["meshes/1/nodes/zoneTopology"]
        coords = f["meshes/1/nodes/coords"]
        for i, (zid, lo, hi) in enumerate(zip(nz["id"][()], nz["minId"][()], nz["maxId"][()]), start=1):
            data = coords[str(int(zid))][()]
            del coords[str(int(zid))]
            ds = coords.create_dataset(str(i) + "_tmp", data=data)
            ds.attrs["minId"], ds.attrs["maxId"] = np.asarray([lo], np.uint64), np.asarray([hi], np.uint64)
        for i in range(1, len(nz["id"][()]) + 1):
            coords.move(f"{i}_tmp", str(i))
        nz["id"][...] = np.asarray(nz["id"][()]) + 50  # zone ids no longer match dataset names
    _same(m, read_case_h5(str(p)))


def test_read_case_h5_rejects_non_case(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.create_group("something")
    with pytest.raises(ValueError):
        read_case_h5(str(p))


def test_quality_metrics_known_values():
    q = mesh_quality(cartesian_mesh_2d(8, 8))
    assert q.min_orthogonal_quality == pytest.approx(1.0) and q.max_skewness == pytest.approx(0.0, abs=1e-12)
    assert q.max_aspect_ratio == pytest.approx(1.0) and q.ok and not q.warnings
    t = mesh_quality(cartesian_mesh_2d(8, 8, triangulate=True))
    assert t.max_skewness == pytest.approx(0.25)  # right isosceles triangles: angles 45/45/90
    s = mesh_quality(cartesian_mesh_2d(4, 4, xs=[0, 1, 2, 3, 4], ys=[0, 0.005, 1, 2, 3]))
    assert s.max_aspect_ratio == pytest.approx(200.0) and any("aspect ratio" in w for w in s.warnings)
    q3 = mesh_quality(extrude_to_3d(cartesian_mesh_2d(4, 4), 2, 0.5))
    assert q3.min_orthogonal_quality == pytest.approx(1.0) and q3.max_aspect_ratio == pytest.approx(1.0)
    assert "Minimum Orthogonal Quality" in q.summary()


def test_quality_flags_bad_cells():
    m = cartesian_mesh_2d(3, 3)
    m.nodes[5] = [0.66, 0.66]  # pull an interior node almost onto a corner
    q = mesh_quality(m)
    assert q.max_skewness > 0.5 and q.min_orthogonal_quality < 0.9


def test_fluent_warmstart_cli(tmp_path):
    from flowtrace.cli import main

    out = tmp_path / "ws"
    main(["fluent-warmstart", os.path.join(EXAMPLES, "cavity_quad.msh"), "--bc", "lid=wall:1,0", "--mu", "0.01",
          "--max-iter", "200", "--out", str(out), "--backend", "numpy"])
    for f in ("flowtrace.ip", "warmstart.jou", "warmstart_pyfluent.py", "report.json", "README_WARMSTART.txt"):
        assert (out / f).exists(), f
    rep = json.loads((out / "report.json").read_text())
    assert rep["iterations"] > 0 and rep["quality"]["min_orthogonal_quality"] > 0.9
    jou = (out / "warmstart.jou").read_text()
    assert "cavity_quad.msh" in jou and "/file/interpolate/read-data" in jou and "trademarks" in jou
    compile((out / "warmstart_pyfluent.py").read_text(), "ws.py", "exec")


def test_fluent_warmstart_h5_input_writes_dat_h5(tmp_path):
    from flowtrace.cli import main

    m = cartesian_mesh_2d(12, 12, names=("left", "right", "bottom", "lid"))
    case = tmp_path / "cav.cas.h5"
    write_case_h5(m, str(case))
    out = tmp_path / "ws"
    main(["fluent-warmstart", str(case), "--bc", "lid=wall:1,0", "--turbulence", "k-omega-sst", "--mu", "1e-3",
          "--max-iter", "30", "--out", str(out), "--backend", "numpy", "--setup-physics"])
    d = read_data_h5(str(out / "flowtrace.dat.h5"))
    assert {"x-velocity", "pressure", "turb-kinetic-energy", "specific-diss-rate"} <= set(d)
    assert d["x-velocity"].shape == (144,)
    assert "kw-sst" in (out / "warmstart.jou").read_text()


def test_fluent_warmstart_rejects_unknown_zone(tmp_path, capsys, monkeypatch):
    from flowtrace.cli import main

    args = ["fluent-warmstart", os.path.join(EXAMPLES, "cavity_quad.msh"), "--bc", "nonexistent=wall:1,0",
            "--max-iter", "2", "--out", str(tmp_path / "x"), "--backend", "numpy"]
    with pytest.raises(SystemExit) as exc:  # the CLI reports user errors in one line, exit status 2
        main(args)
    assert exc.value.code == 2 and "'nonexistent', which is not a boundary zone" in capsys.readouterr().err
    monkeypatch.setenv("FLOWTRACE_DEBUG", "1")  # ... and re-raises them for debugging
    with pytest.raises(ValueError):
        main(args)


def test_solve_rejects_unknown_zone_before_running(tmp_path, capsys):
    from flowtrace.cli import main

    args = ["solve", os.path.join(EXAMPLES, "cavity_quad.msh"), "--bc", "nonexistent=wall:1,0",
            "--max-iter", "2", "--workdir", str(tmp_path)]
    with pytest.raises(SystemExit) as exc:
        main(args)
    err = capsys.readouterr()
    assert exc.value.code == 2 and "'nonexistent', not a boundary zone" in err.err and "lid" in err.err
    assert "Traceback" not in err.out + err.err and not (tmp_path / "checkpoints").exists()


def test_skewness_is_not_applied_to_polyhedral_cells():
    """Equiangle skewness is an element-type metric: polyhedral cells (Fluent Meshing poly meshes)
    are judged by orthogonal quality, and standard cells keep their skewness."""
    import numpy as np

    from flowtrace.mesh.quality import mesh_quality
    from flowtrace.mesh.unstructured import agglomerate_cells, cartesian_mesh_2d, extrude_to_3d

    m3 = extrude_to_3d(cartesian_mesh_2d(4, 4, warp=0.05), 2, 0.5)
    c = np.arange(m3.n_cells)
    c2, k = c % 16, c // 16
    i, j = c2 // 4, c2 % 4
    group = np.where(i < 2, ((i // 2) * 2 + j // 2) * 2 + k // 2, 1000 + c)  # left half: 2x2x2 polyhedra
    poly = agglomerate_cells(m3, group)
    q = mesh_quality(poly)
    n_poly = int((np.bincount(poly.owner, minlength=poly.n_cells)
                  + np.bincount(poly.neighbour[poly.neighbour >= 0], minlength=poly.n_cells) > 6).sum())
    assert n_poly == 2 and q.skewness_not_applicable_cells == n_poly
    assert 0.0 < q.max_skewness <= mesh_quality(m3).max_skewness  # the remaining (warped) hexes keep theirs
    assert any("not evaluated" in w for w in q.warnings) and q.ok


def test_text_file_with_fluent_mesh_content_is_read(tmp_path):
    """The readers sniff the content, not the extension: a Fluent ASCII mesh saved as .txt loads."""
    import shutil

    from flowtrace.ui.server import AppState, register_mesh

    txt = tmp_path / "cavity.txt"
    shutil.copy(os.path.join(EXAMPLES, "cavity_quad.msh"), txt)
    mesh, fmt = read_any_mesh(str(txt))
    assert fmt == "fluent-msh" and mesh.n_cells == 1600
    assert register_mesh(AppState(str(tmp_path)), txt.read_bytes(), "cavity.txt")["cells"] == 1600
