"""Density-based compressible solver: Sod tube, oblique shock, axisymmetric balance."""

import numpy as np
import pytest

from flowtrace.compressible import CompBC, CompressibleCase, CompressibleSolver
from flowtrace.compressible.riemann import oblique_shock, sample
from flowtrace.mesh.unstructured import cartesian_mesh_2d


def _sod(n, order=2):
    m = cartesian_mesh_2d(n, 2, 1.0, 2.0 / n)
    s = CompressibleSolver(m, CompressibleCase(time="transient", t_end=0.2, cfl=0.5, order=order, backend="numpy"))
    x = s.cell_centres()[:, 0]
    left = x < 0.5
    s.initialize(np.where(left, 1.0, 0.125), np.zeros((x.size, 2)), np.where(left, 1.0, 0.1))
    while not s.finished:
        s.step()
    exact = sample(x, 0.2, (1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
    return float(np.mean(np.abs(s.fields()["density"] - exact[0])))


def test_sod_converges_and_second_order_beats_first():
    e1, e2 = _sod(100), _sod(200)
    assert e2 < 0.006 and e1 / e2 > 1.6  # shocks limit L1 to order ~1
    assert _sod(100, order=1) > 1.4 * e1


def _wedge(theta=10.0, mach=2.0, nx=48, ny=24):
    xs, ys = np.linspace(0, 2, nx + 1), np.linspace(0, 1, ny + 1)
    m = cartesian_mesh_2d(nx, ny, xs=xs, ys=ys, names=("in", "out", "wall", "top"))
    X = m.nodes
    ramp = np.where(X[:, 0] > 0.5, (X[:, 0] - 0.5) * np.tan(np.radians(theta)), 0.0)
    X[:, 1] = ramp + X[:, 1] * (1.0 - ramp)
    T, p = 300.0, 1e5
    u = mach * np.sqrt(1.4 * 287.0 * T)
    free = CompBC(kind="supersonic-inlet", velocity=[u, 0], p=p, T=T)
    case = CompressibleCase(boundaries={"in": free, "top": free, "out": CompBC(kind="pressure-outlet", p=p),
                                        "wall": CompBC(kind="slip-wall")},
                            init_p=p, init_T=T, init_velocity=[u, 0, 0], cfl=2, tol=1e-7, max_iter=2000,
                            backend="numpy")
    s = CompressibleSolver(m, case)
    while not s.finished:
        s.step()
    return s, p


def test_oblique_shock_pressure_ratio():
    s, p = _wedge()
    assert s.converged
    xc, f = s.cell_centres(), s.fields()
    below = (xc[:, 0] > 1.2) & (xc[:, 0] < 1.5) & (xc[:, 1] < 0.2 + (xc[:, 0] - 0.5) * np.tan(np.radians(10)))
    beta, p2p1, _, M2 = oblique_shock(2.0, 10.0)
    assert np.median(f["pressure"][below]) / p == pytest.approx(p2p1, rel=0.01)
    assert np.median(f["mach-number"][below]) == pytest.approx(M2, rel=0.01)


@pytest.mark.parametrize("turbulence", ["laminar", "k-omega-sst"])
def test_axisymmetric_gas_at_rest_stays_at_rest(turbulence):
    """The radial pressure source must balance the radius-weighted face fluxes exactly."""
    m = cartesian_mesh_2d(12, 8, 1.0, 0.5, names=("a", "b", "axis", "c"), warp=0.05)
    case = CompressibleCase(axisymmetric=True, viscous=True, turbulence=turbulence, max_iter=20, backend="numpy",
                            boundaries={"axis": CompBC(kind="symmetry")})
    s = CompressibleSolver(m, case)
    for _ in range(20):
        s.step()
    f = s.fields()
    assert np.abs(f["velocity-magnitude"]).max() < 1e-8
    assert np.ptp(f["pressure"]) < 1e-6 * 101325


def test_native_hllc_matches_array_path():
    from flowtrace import native

    if not native.available():
        pytest.skip(native.status().get("reason", "no native kernels"))
    rng = np.random.default_rng(1)
    for dim in (2, 3):
        nf = 5000
        def state():
            W = np.empty((nf, dim + 2))
            W[:, 0] = rng.uniform(0.1, 2.0, nf)
            W[:, 1:1 + dim] = rng.normal(0, 400.0, (nf, dim))
            W[:, -1] = rng.uniform(1e4, 3e5, nf)
            return W
        n = rng.normal(size=(nf, dim))
        n /= np.linalg.norm(n, axis=1)[:, None]
        A = rng.uniform(0.1, 2.0, nf)
        WL, WR = state(), state()
        m = cartesian_mesh_2d(2, 2)
        s = CompressibleSolver(m, CompressibleCase(backend="numpy"))
        s.dim = dim
        ref = s._hllc(WL, WR, n) * A[:, None]
        got = native.hllc_flux(WL, WR, n, A, 1.4)
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-9 * np.abs(ref).max())


def test_native_and_array_solvers_agree():
    from flowtrace import native

    if not native.available():
        pytest.skip("no native kernels")
    s_nat, _ = _wedge(nx=24, ny=12)
    with native.disabled():
        s_arr, _ = _wedge(nx=24, ny=12)
    assert s_nat.flux_kernel == "native-cpu" and s_arr.flux_kernel == "array"
    assert np.allclose(s_nat.fields()["pressure"], s_arr.fields()["pressure"], rtol=1e-8)


def test_polyhedral_3d_wedge_through_fluent_msh(tmp_path):
    """Hexes agglomerated 2x2x2 into 24-face polyhedra, written to and read back from a
    Fluent .msh, must still give the oblique-shock pressure ratio in 3-D."""
    from flowtrace.mesh.fluent_msh import read_fluent_mesh, write_fluent_mesh
    from flowtrace.mesh.unstructured import agglomerate_cells, extrude_to_3d

    nx, ny, nz, theta = 48, 24, 2, 10.0
    xs, ys = np.linspace(0, 2, nx + 1), np.linspace(0, 1, ny + 1)
    m2 = cartesian_mesh_2d(nx, ny, xs=xs, ys=ys, names=("in", "out", "wall", "top"))
    X = m2.nodes
    ramp = np.where(X[:, 0] > 0.5, (X[:, 0] - 0.5) * np.tan(np.radians(theta)), 0.0)
    X[:, 1] = ramp + X[:, 1] * (1.0 - ramp)
    m3 = extrude_to_3d(m2, nz, 0.1)
    c = np.arange(m3.n_cells)
    c2, k = c % (nx * ny), c // (nx * ny)
    i, j = c2 // ny, c2 % ny
    poly = agglomerate_cells(m3, ((i // 2) * (ny // 2) + j // 2) * nz + k // 2)
    assert poly.n_cells == m3.n_cells // 8
    path = tmp_path / "wedge_poly.msh"
    write_fluent_mesh(poly, str(path))
    mesh = read_fluent_mesh(str(path))
    faces_per_cell = np.bincount(mesh.owner, minlength=mesh.n_cells) + np.bincount(
        mesh.neighbour[mesh.neighbour >= 0], minlength=mesh.n_cells)
    assert faces_per_cell.min() == 24  # 4 coplanar quads on each of 6 sides
    T, p = 300.0, 1e5
    u = 2.0 * np.sqrt(1.4 * 287.0 * T)
    free = CompBC(kind="supersonic-inlet", velocity=[u, 0, 0], p=p, T=T)
    case = CompressibleCase(boundaries={"in": free, "top": free, "out": CompBC(kind="pressure-outlet", p=p),
                                        "wall": CompBC(kind="slip-wall"), "front": CompBC(kind="symmetry"),
                                        "back": CompBC(kind="symmetry")},
                            init_p=p, init_T=T, init_velocity=[u, 0, 0], cfl=2, tol=1e-7, max_iter=2000,
                            backend="numpy")
    s = CompressibleSolver(mesh, case)
    while not s.finished:
        s.step()
    assert s.converged
    xc, f = s.cell_centres(), s.fields()
    below = (xc[:, 0] > 1.2) & (xc[:, 0] < 1.6) & (xc[:, 1] < 0.2 + (xc[:, 0] - 0.5) * np.tan(np.radians(theta)))
    _, p2p1, _, _ = oblique_shock(2.0, theta)
    assert np.median(f["pressure"][below]) / p == pytest.approx(p2p1, rel=0.02)
    assert np.abs(f["z-velocity"]).max() < 1e-6 * u


def test_sst_flat_plate_turbulent_skin_friction():
    """M = 0.5 plate, y+ < 1: SST must give a turbulent wall shear, far above the laminar (Blasius) one."""
    from flowtrace.mesh.unstructured import split_zone, stretch

    T, p = 300.0, 101325.0
    U = 0.5 * np.sqrt(1.4 * 287.0 * T)
    rho, mu = p / (287.0 * T), 1.7894e-5
    xs = np.concatenate([np.linspace(-0.2, 0.0, 9)[:-1], stretch(30, 1.0, first=5e-3)])
    ys = stretch(40, 0.08, first=3e-6)
    m = split_zone(cartesian_mesh_2d(0, 0, xs=xs, ys=ys, names=("in", "out", "bottom", "top")),
                   "bottom", lambda c: c[:, 0] > 0, new_name="plate", new_type="wall")
    bc = {"in": CompBC(kind="velocity-inlet", velocity=[U, 0], T=T, turb_intensity=0.01, turb_visc_ratio=5),
          "out": CompBC(kind="pressure-outlet", p=p), "top": CompBC(kind="pressure-outlet", p=p),
          "bottom": CompBC(kind="symmetry"), "plate": CompBC(kind="wall")}
    s = CompressibleSolver(m, CompressibleCase(boundaries=bc, turbulence="k-omega-sst", init_p=p, init_T=T,
                                               init_velocity=[U, 0, 0], cfl_max=100, max_iter=1500,
                                               backend="numpy"))
    while not s.finished:
        s.step()
    f, xc = s.fields(), s.cell_centres()
    assert np.isfinite(f["viscosity-turb"]).all() and f["viscosity-turb"].max() > 100 * mu
    ncol = len(ys) - 1
    u1 = f["x-velocity"].reshape(-1, ncol)[:, 0]
    y1, x = xc[:, 1].reshape(-1, ncol)[:, 0], xc[:, 0].reshape(-1, ncol)[:, 0]
    i = np.argmin(np.abs(x - 0.7))
    cf = mu * u1[i] / y1[i] / (0.5 * rho * U ** 2)
    re = rho * U * x[i] / mu
    assert cf > 4.0 * 0.664 / np.sqrt(re)  # laminar Blasius value
    assert 0.7 < cf / (0.0592 * re ** -0.2) < 1.3  # turbulent power law
