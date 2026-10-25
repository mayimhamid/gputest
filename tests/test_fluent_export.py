import numpy as np

from flowtrace.fluent.bridge import pyfluent_status
from flowtrace.fluent.export import (read_interpolation_file, write_interpolation_file, write_journal, write_profile,
                                     write_pyfluent_script)


def test_interpolation_file_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(37, 2))
    fields = {"x-velocity": rng.standard_normal(37), "pressure": rng.standard_normal(37)}
    p = tmp_path / "a.ip"
    write_interpolation_file(str(p), pts, fields)
    head = p.read_text().splitlines()[:6]
    assert head == ["3", "2", "37", "2", "x-velocity", "pressure"]
    pts2, f2 = read_interpolation_file(str(p))
    assert np.allclose(pts, pts2)
    assert all(np.allclose(fields[k], f2[k]) for k in fields)


def test_interpolation_file_drops_nan_points(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = tmp_path / "b.ip"
    write_interpolation_file(str(p), pts, {"temperature": np.array([1.0, np.nan, 3.0])})
    pts2, f2 = read_interpolation_file(str(p))
    assert pts2.shape == (2, 3) and np.allclose(f2["temperature"], [1.0, 3.0])


def test_profile_journal_and_script(tmp_path):
    p = tmp_path / "in.prof"
    write_profile(str(p), "inlet", np.array([[0.0, 0.0], [0.0, 1.0]]), {"x-velocity": [0.0, 1.0]})
    txt = p.read_text()
    assert txt.startswith("((inlet point 2)") and "(x-velocity" in txt and txt.rstrip().endswith(")")
    j = tmp_path / "run.jou"
    write_journal(str(j), "m.msh", "s.ip", 123, energy=True)
    jt = j.read_text()
    assert '/file/read-case "m.msh"' in jt and '/file/interpolate/read-data "s.ip"' in jt and "/solve/iterate 123" in jt
    s = tmp_path / "run.py"
    write_pyfluent_script(str(s), "m.msh", "s.ip", boundaries={"lid": {"kind": "wall", "velocity": [1.0, 0.0]},
                                                               "in": {"kind": "inlet", "normal_speed": 2.0},
                                                               "out": {"kind": "outlet", "pressure": 0.0}})
    compile(s.read_text(), "run.py", "exec")


def test_pyfluent_status_never_raises():
    st = pyfluent_status()
    assert isinstance(st.available, bool) and st.message
