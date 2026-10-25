import numpy as np
from scipy.interpolate import griddata

from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.unstructured import cartesian_mesh_2d, extrude_to_3d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
from flowtrace.unstructured.simple import SimpleSolver
from flowtrace.validation.reference import compare_ghia

be = get_backend("numpy")


def run(s):
    while not s.finished:
        s.step()
    return s


def _ghia_err(s):
    xc, f = s.cell_centres(), s.fields()
    y = np.linspace(0.02, 0.98, 49)
    u = griddata(xc[:, :2], f["x-velocity"], np.column_stack([np.full_like(y, 0.5), y]))
    v = griddata(xc[:, :2], f["y-velocity"], np.column_stack([y, np.full_like(y, 0.5)]))
    return compare_ghia(100, y, u, y, v)


def test_simplec_cavity_skewed_triangles():
    m = cartesian_mesh_2d(32, 32, triangulate=True, jitter=0.2)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         max_iter=3000, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    assert s.converged
    e = _ghia_err(s)
    assert e["u_max_err"] < 0.03 and e["v_max_err"] < 0.03, e


def test_simple_algorithm_converges_too():
    m = cartesian_mesh_2d(24, 24)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         algorithm="SIMPLE", max_iter=4000, tol=1e-5)
    assert c.relax_p == 0.3
    s = run(SimpleSolver(m, c, be))
    assert s.converged


def test_channel_mass_balance_and_heat():
    m = cartesian_mesh_2d(60, 12, 5.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"))
    bnd = default_boundaries(m)
    assert bnd["inlet"].kind == "inlet" and bnd["outlet"].kind == "outlet"
    bnd["inlet"] = BoundarySpec("inlet", normal_speed=1.0, value=0.0)
    bnd["bottom"] = BoundarySpec("wall", thermal="temperature", value=1.0)
    c = UnstructuredCase(boundaries=bnd, viscosity=0.02, energy=True, conductivity=0.02, max_iter=1500, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    assert s.converged
    mb = s.mass_balance()
    assert abs(mb["inlet"] + 1.0) < 1e-9 and abs(sum(mb.values())) < 1e-8
    T = s.fields()["temperature"]
    assert -1e-8 <= T.min() and T.max() <= 1.0 + 1e-6


def test_3d_extruded_cavity_is_two_dimensional():
    m = extrude_to_3d(cartesian_mesh_2d(16, 16), 2, 0.2)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, viscosity=0.01,
                         max_iter=400, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    f = s.fields()
    assert np.abs(f["z-velocity"]).max() < 1e-8
    assert f["x-velocity"].min() < -0.15
    r = s.residuals()["scaled"]
    assert r["z-velocity"] < 1e-3
