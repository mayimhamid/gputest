import numpy as np
import pytest

from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.unstructured import (cartesian_mesh_2d, compute_geometry, match_periodic, split_zone, stretch,
                                         wall_distance)
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries, default_periodic_pairs
from flowtrace.unstructured.presets import (apply_initial_condition, flat_plate, periodic_taylor_green,
                                            radiation_slab, turbulent_channel)
from flowtrace.unstructured.simple import SimpleSolver
from flowtrace.unstructured.turbulence import spalding_utau
from flowtrace.validation.checks import validate_unstructured

be = get_backend("numpy")


def run(s):
    while not s.finished:
        s.step()
    return s


# ---------------------------------------------------------------- mesh tools
def test_stretch_and_split_and_periodic_match():
    x = stretch(30, 2.0, first=1e-3)
    assert np.isclose(x[0], 0) and np.isclose(x[-1], 2.0) and np.isclose(x[1], 1e-3)
    y = stretch(30, 2.0, first=1e-3, both=True)
    assert np.allclose(np.diff(y), np.diff(y)[::-1])
    m = cartesian_mesh_2d(0, 0, xs=np.linspace(0, 1, 6), ys=y, names=("a", "b", "c", "d"),
                          types=("periodic", "periodic", "wall", "wall"))
    m2 = split_zone(m, "c", lambda fc: np.where(fc[:, 0] < 0.5, "front", "plate"), {"front": "symmetry"})
    names = {z.name: (z.type, z.count) for z in m2.face_zones()}
    assert names["front"] == ("symmetry", 2) and names["plate"] == ("wall", 3)
    fa, fb, t = match_periodic(m2, "a", "b")
    assert fa.size == 30 and np.allclose(t, [1.0, 0.0])
    assert default_periodic_pairs(m2, default_boundaries(m2)) == [["a", "b"]]
    g = compute_geometry(m2)
    d = wall_distance(g.cell_centre, m2, m2.faces_of_zone(m2.zone_by_name("d").id))
    assert np.allclose(d, 2.0 - g.cell_centre[:, 1])


def test_spalding_law_limits():
    xp = np
    nu = 1e-5
    y = np.array([1e-6, 1e-3])
    # viscous sublayer: u+ = y+  -> utau^2 = nu ut / y
    ut = np.array([0.01, 10.0])
    utau = spalding_utau(xp, ut, y, nu)
    assert np.isclose(utau[0] ** 2, nu * ut[0] / y[0], rtol=1e-3)
    yp, up = y[1] * utau[1] / nu, ut[1] / utau[1]
    assert yp > 30 and np.isclose(up, np.log(yp) / 0.41 + 5.2, rtol=0.02)


# ---------------------------------------------------------------- periodic + transient
def test_periodic_poiseuille():
    m = cartesian_mesh_2d(3, 20, 1.0, 1.0, names=("left", "right", "bottom", "top"),
                          types=("periodic", "periodic", "wall", "wall"), triangulate=True)
    c = UnstructuredCase(periodic=[["left", "right"]], periodic_pressure_gradient=[-1.0, 0, 0], viscosity=0.1,
                         max_iter=600, tol=1e-9)
    s = run(SimpleSolver(m, c, be))
    xc = s.cell_centres()
    ue = xc[:, 1] * (1 - xc[:, 1]) / (2 * 0.1)
    assert np.abs(s.fields()["x-velocity"] - ue).max() / ue.max() < 5e-3
    assert abs(s.periodic_mass_flow()["left->right"] - 1 / 1.2) < 5e-3


def test_transient_taylor_green_periodic_mesh():
    mesh, case, info = periodic_taylor_green(n=16, t_end=0.5)
    s = SimpleSolver(mesh, case, be)
    apply_initial_condition(s, info)
    run(s)
    assert abs(s.time - 0.5) < 1e-9 and s.time_step == 10
    chk = validate_unstructured(s, info)["checks"][0]
    assert chk["value"] < 0.02, chk


# ---------------------------------------------------------------- SST
def test_sst_channel_matches_dns_bulk_velocity():
    mesh, case, info = turbulent_channel(ny=64)
    s = run(SimpleSolver(mesh, case, be))
    assert s.converged
    checks = {c["name"]: c for c in validate_unstructured(s, info)["checks"]}
    ub = next(c for n, c in checks.items() if n.startswith("Bulk velocity"))
    assert ub["pass"], ub
    bal = checks["Wall shear balances the driving pressure gradient"]
    assert bal["pass"], bal
    f = s.fields()
    assert f["viscosity-ratio"].max() > 20 and f["turb-kinetic-energy"].min() >= 0


def test_sst_flat_plate_skin_friction():
    mesh, case, info = flat_plate(nx=48, ny=40)
    s = run(SimpleSolver(mesh, case, be))
    assert s.converged
    checks = validate_unstructured(s, info)["checks"]
    assert checks[0]["pass"] and checks[1]["pass"], checks
    rep = s.wall_report()["plate"]
    assert rep["yplus_max"] < 1.0 and 0.0025 < rep["cd"] < 0.0031


# ---------------------------------------------------------------- P-1
@pytest.mark.parametrize("tau", [0.2, 2.0])
def test_p1_radiation_slab(tau):
    mesh, case, info = radiation_slab(tau=tau)
    s = run(SimpleSolver(mesh, case, be))
    for c in validate_unstructured(s, info)["checks"]:
        assert c["pass"], c
    assert "incident-radiation" in s.fields()


def test_case_validation_errors():
    with pytest.raises(ValueError):
        UnstructuredCase(radiation="p1", energy=False)
    with pytest.raises(ValueError):
        UnstructuredCase(turbulence="k-epsilon")
    assert UnstructuredCase(turbulence="sst").turbulence == "k-omega-sst"
    m = cartesian_mesh_2d(4, 4, types=("periodic", "periodic", "wall", "wall"))
    with pytest.raises(ValueError):
        SimpleSolver(m, UnstructuredCase(boundaries={"left": BoundarySpec("periodic"),
                                                     "right": BoundarySpec("periodic")}), be)


def test_turbulent_state_roundtrip():
    mesh, case, info = turbulent_channel(ny=32)
    case.max_iter = 30
    a = SimpleSolver(mesh, case, be)
    for _ in range(20):
        a.step()
    st = a.state_dict()
    b = SimpleSolver(mesh, case, be)
    b.load_state(st)
    for _ in range(5):
        a.step()
        b.step()
    assert np.abs(a.fields()["x-velocity"] - b.fields()["x-velocity"]).max() < 1e-10
