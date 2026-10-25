"""Periodic zones, rotating walls / MRF zones, conjugate heat transfer."""

import numpy as np
import pytest

from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.fluent_msh import read_fluent_mesh, write_fluent_mesh
from flowtrace.mesh.unstructured import (annulus_mesh, cartesian_mesh_2d, compute_geometry, extract_cells,
                                         make_periodic, set_cell_zone, split_zone)
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.conjugate import ConjugateSolver, make_solver
from flowtrace.unstructured.simple import SimpleSolver

be = get_backend("numpy")


def run(s):
    while not s.finished:
        s.step()
    return s


@pytest.mark.parametrize("tri", [False, True])
def test_periodic_poiseuille_and_fluent_roundtrip(tmp_path, tri):
    m = make_periodic(cartesian_mesh_2d(4, 12, 0.5, 1.0, names=("in", "out", "bottom", "top"), triangulate=tri),
                      "in", "out")
    assert [z.name for z in m.boundary_zones()] == ["bottom", "top"]
    p = tmp_path / "per.msh"
    write_fluent_mesh(m, str(p))
    m2 = read_fluent_mesh(str(p))
    assert len(m2.periodic) == 1 and np.allclose(m2.periodic[0]["translation"], [0.5, 0.0])
    s = run(SimpleSolver(m2, UnstructuredCase(viscosity=0.1, body_force=[1.0, 0, 0], max_iter=6000, tol=1e-9,
                                              relax_u=0.9), backend=be))
    assert s.converged
    xc, f = s.cell_centres(), s.fields()
    assert np.abs(f["x-velocity"] - 5 * xc[:, 1] * (1 - xc[:, 1])).max() < 0.02
    assert np.abs(f["y-velocity"]).max() < 1e-8


def test_make_periodic_rejects_mismatched_zones():
    m = cartesian_mesh_2d(4, 6, names=("in", "out", "bottom", "top"))
    with pytest.raises(ValueError):
        make_periodic(m, "in", "bottom")


def test_split_zone_and_set_cell_zone_roundtrip(tmp_path):
    m = cartesian_mesh_2d(10, 4, 2.0, 1.0)
    m = split_zone(m, "bottom", lambda fc: fc[:, 0] > 1.0, "plate", "wall")
    m = set_cell_zone(m, lambda x: x[:, 0] < 0.6, "block", "solid")
    assert m.zone_by_name("plate").count == 5 and m.zone_by_name("bottom").count == 5
    cz = {z.name: z.count for z in m.cell_zones()}
    assert cz == {"fluid": 28, "block": 12}
    p = tmp_path / "z.msh"
    write_fluent_mesh(m, str(p))
    m2 = read_fluent_mesh(str(p))
    assert {z.name: z.type for z in m2.cell_zones()} == {"fluid": "fluid", "block": "solid"}
    assert compute_geometry(m2).flipped_faces == 0


def test_extract_cells_creates_oriented_interfaces():
    m = set_cell_zone(cartesian_mesh_2d(6, 6), lambda x: x[:, 1] < 0.5, "solid", "solid")
    sub, fmap, cmap = extract_cells(m, m.cell_zone != m.zone_by_name("solid").id)
    assert sub.n_cells == 18 and cmap.size == 18
    iface = sub.zone_by_name("interface-solid")
    assert iface is not None and iface.count == 6
    g = compute_geometry(sub)
    assert g.flipped_faces == 0 and np.isclose(g.total_volume, 0.5)


def _taylor_couette_error(mesh, **kw):
    s = run(SimpleSolver(mesh, UnstructuredCase(viscosity=0.1, max_iter=4000, tol=1e-9, **kw), backend=be))
    xc, f = s.cell_centres(), s.fields()
    r = np.linalg.norm(xc, axis=1)
    ut = (-xc[:, 1] * f["x-velocity"] + xc[:, 0] * f["y-velocity"]) / r
    return np.abs(ut - (-r / 3 + 4 / (3 * r))).max()


def test_rotating_wall_taylor_couette():
    e1 = _taylor_couette_error(annulus_mesh(1, 2, 6, 36), boundaries={"inner": BoundarySpec("wall", rotation={"omega": 1.0})})
    e2 = _taylor_couette_error(annulus_mesh(1, 2, 12, 72), boundaries={"inner": BoundarySpec("wall", rotation={"omega": 1.0})})
    assert e2 < 6e-3 and np.log2(e1 / e2) > 1.8


def test_mrf_zone_gives_the_absolute_solution():
    m = set_cell_zone(annulus_mesh(1, 2, 12, 72), lambda x: np.linalg.norm(x, axis=1) < 1.5, "rotor")
    e = _taylor_couette_error(m, mrf={"rotor": {"omega": 1.0}})
    assert e < 6e-3
    with pytest.raises(ValueError):
        SimpleSolver(m, UnstructuredCase(mrf={"nope": {"omega": 1.0}}), backend=be)


def test_conjugate_composite_wall_is_exact():
    m = cartesian_mesh_2d(30, 3, 3.0, 1.0)
    m = set_cell_zone(m, lambda x: x[:, 0] < 1.0, "a", "solid")
    m = set_cell_zone(m, lambda x: x[:, 0] > 2.0, "b", "solid")
    c = UnstructuredCase(energy=True, conductivity=1.0, solids={"a": {"conductivity": 5.0}, "b": {"conductivity": 0.5}},
                         boundaries={"left": BoundarySpec("wall", thermal="temperature", value=1.0),
                                     "right": BoundarySpec("wall", thermal="temperature", value=0.0)},
                         max_iter=300, tol=1e-10)
    s = run(make_solver(m, c, backend=be))
    assert isinstance(s, ConjugateSolver)
    x, T = s.cell_centres()[:, 0], s.fields()["temperature"]
    q = 1 / 3.2
    Te = np.where(x < 1, 1 - q * x / 5, np.where(x < 2, 1 - q / 5 - q * (x - 1), 1 - 1.2 * q - 2 * q * (x - 2)))
    assert np.abs(T - Te).max() < 1e-10


def test_conjugate_requires_solid_conductivity():
    m = set_cell_zone(cartesian_mesh_2d(4, 4), lambda x: x[:, 0] < 0.5, "s", "solid")
    with pytest.raises(ValueError):
        make_solver(m, UnstructuredCase(energy=True), backend=be)


def test_conjugate_heated_channel_energy_balance():
    m = cartesian_mesh_2d(30, 10, 4.0, 1.25, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"), y0=-0.25)
    m = split_zone(m, "inlet", lambda fc: fc[:, 1] < 0.0, "heater-left", "wall")
    m = split_zone(m, "outlet", lambda fc: fc[:, 1] < 0.0, "heater-right", "wall")
    m = set_cell_zone(m, lambda x: x[:, 1] < 0.0, "heater", "solid")
    c = UnstructuredCase(energy=True, conductivity=0.05, viscosity=0.05, solids={"heater": {"conductivity": 2.0, "source": 4.0}},
                         boundaries={"inlet": BoundarySpec("inlet", normal_speed=1.0, value=0.0),
                                     "outlet": BoundarySpec("outlet")}, max_iter=3000, tol=1e-8)
    s = run(make_solver(m, c, backend=be))
    q = s.heat_flow()
    assert abs(q["fluid-solid interfaces"] - 4.0) < 1e-6
    assert abs(sum(v for k, v in q.items() if k != "fluid-solid interfaces")) < 1e-8
    f = s.fields()
    assert np.all(f["x-velocity"][s.solid] == 0) and f["temperature"].max() > 0
