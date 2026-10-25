import numpy as np
import pytest

from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.unstructured import cartesian_mesh_2d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.simple import SimpleSolver
from flowtrace.unstructured.turbulence import KAPPA, spalding_uplus
from flowtrace.validation.turbulent import run_channel

be = get_backend("numpy")


def test_spalding_law_limits():
    re_y = np.array([1.0, 4.0, 25.0, 1e4, 1e6])
    up = spalding_uplus(re_y)
    yp = re_y / up
    assert np.allclose(up[:2], np.sqrt(re_y[:2]), rtol=2e-3)  # viscous sublayer: u+ = y+
    loglaw = np.log(yp[-2:]) / KAPPA + 5.2
    assert np.allclose(up[-2:], loglaw, rtol=2e-3)
    assert spalding_uplus(np.array([0.0]))[0] == 0.0


def test_wall_distance_is_exact_on_a_channel():
    m = cartesian_mesh_2d(6, 10, 3.0, 1.0, names=("in", "out", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"), warp=0.0)
    c = UnstructuredCase(boundaries={"in": BoundarySpec("inlet", normal_speed=1.0)}, viscosity=1e-3,
                         turbulence="k-omega-sst", max_iter=1)
    s = SimpleSolver(m, c, backend=be)
    y = s.cell_centres()[:, 1]
    assert np.allclose(s.turb.y, np.minimum(y, 1 - y), atol=1e-12)


@pytest.mark.parametrize("model", ["k-omega-sst", "k-epsilon-realizable", "k-epsilon-standard"])
def test_turbulent_channel_runs_and_reports_fields(model, tmp_path):
    m = cartesian_mesh_2d(20, 8, 4.0, 1.0, names=("in", "out", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"))
    c = UnstructuredCase(boundaries={"in": BoundarySpec("inlet", normal_speed=1.0)}, viscosity=1e-4,
                         turbulence=model, max_iter=150, tol=1e-12)
    s = SimpleSolver(m, c, backend=be)
    for _ in range(150):
        s.step()
    f = s.fields()
    assert np.all(np.isfinite(f["x-velocity"])) and f["viscosity-ratio"].max() > 1.0
    assert f["turb-kinetic-energy"].min() > 0
    assert "y-plus" in f and f["y-plus"].max() > 0
    st = s.state_dict()
    s2 = SimpleSolver(m, c, backend=be)
    s2.load_state(st)
    assert np.allclose(s2.be.to_numpy(s2.turb.k), st["k"])


def test_sst_channel_matches_dns():
    r = run_channel("k-omega-sst")
    assert r["converged"] and abs(r["tau_w"] - 1.0) < 0.01
    assert r["max_abs_err_pct"] < 5.0, r["err_pct"]


def test_kepsilon_channel_with_wall_functions():
    r = run_channel("k-epsilon-standard")
    assert r["converged"] and abs(r["tau_w"] - 1.0) < 0.01
    assert r["max_abs_err_pct"] < 5.0, r["err_pct"]
    assert 20 < r["y_plus_first"] < 40  # the mesh is in the wall-function range
    rr = run_channel("k-epsilon-realizable")
    assert rr["converged"] and rr["max_abs_err_pct"] < 10.0, rr["err_pct"]


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        UnstructuredCase(turbulence="spalart-allmaras-typo")
