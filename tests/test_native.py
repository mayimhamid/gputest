"""Native C++/OpenMP (g++) and CUDA kernels against the NumPy/CuPy reference path."""

import numpy as np
import pytest

from flowtrace import native
from flowtrace.hardware.backend import get_backend
from flowtrace.linalg.krylov import bicgstab
from flowtrace.mesh.unstructured import cartesian_mesh_2d, extrude_to_3d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.fvmesh import FVMesh
from flowtrace.unstructured.simple import SimpleSolver

needs_native = pytest.mark.skipif(not native.available(), reason=f"no native kernels: {native.status()['reason']}")
CPU = get_backend("numpy")


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _meshes():
    yield cartesian_mesh_2d(20, 16, triangulate=True, warp=0.05)  # skewed triangles
    yield extrude_to_3d(cartesian_mesh_2d(6, 5, warp=0.05), 4, 0.5)  # 3-D hexahedra


@needs_native
@pytest.mark.parametrize("mesh", list(_meshes()), ids=["tri2d", "hex3d"])
def test_cpu_kernels_match_reference(mesh):
    with native.disabled():
        ref = FVMesh(mesh, CPU)
    nat = FVMesh(mesh, CPU)
    assert ref.nat is None and nat.nat is not None
    rng = np.random.default_rng(0)
    phi, phib = rng.standard_normal(ref.nc), rng.standard_normal(ref.n_bf)
    g = rng.standard_normal((ref.nc, ref.dim))
    F, G = rng.standard_normal(ref.n_if), rng.standard_normal((ref.n_if, ref.dim))
    assert _rel(nat.face_interp(phi), ref.face_interp(phi)) == 0.0
    assert _rel(nat.face_interp_vec(g), ref.face_interp_vec(g)) == 0.0
    assert _rel(nat.gradient(phi, phib), ref.gradient(phi, phib)) < 1e-13
    assert _rel(nat.gradient_gg(phi, phib), ref.gradient_gg(phi, phib)) < 1e-13
    assert _rel(nat.gradient_gg(phi, phib, g), ref.gradient_gg(phi, phib, g)) < 1e-13
    assert _rel(nat.Cdiff_d @ F, ref.Cdiff_d @ F) == 0.0  # same summation order as SciPy
    assert _rel(nat.Cown_d @ G, ref.Cown_d @ G) == 0.0
    d = 4 + rng.random(ref.nc)
    a1, a2 = -rng.random(ref.n_if), -rng.random(ref.n_if)
    assert _rel(nat.matrix(d, a1, a2).toarray(), ref.matrix(d, a1, a2).toarray()) == 0.0


@needs_native
@pytest.mark.parametrize("tol,maxiter", [(0.1, 30), (1e-10, 500), (1e-10, 3)])
def test_native_bicgstab_follows_the_python_algorithm(tol, maxiter):
    fv = FVMesh(cartesian_mesh_2d(24, 24, triangulate=True, warp=0.05), CPU)
    rng = np.random.default_rng(2)
    d = 4 + rng.random(fv.nc)
    A = fv.matrix(d, -rng.random(fv.n_if), -rng.random(fv.n_if))
    b, x0, inv = rng.standard_normal(fv.nc), rng.standard_normal(fv.nc), 1.0 / d
    xr, ir = bicgstab(lambda v: A.csr @ v, b, np, M=lambda v: inv * v, x0=x0, tol=tol, maxiter=maxiter,
                      relative_to_initial=True)
    xn, inn = native.bicgstab_jacobi(A, b, x0, inv, tol, maxiter)
    assert (inn.iterations, inn.converged) == (ir.iterations, ir.converged)
    assert abs(inn.residual - ir.residual) <= 1e-10 * max(ir.residual, 1e-300)
    assert _rel(xn, xr) < 1e-12


def _run_pair(mesh, case):
    with native.disabled():
        ref = SimpleSolver(mesh, case)
        while not ref.finished:
            ref.step()
    nat = SimpleSolver(mesh, case)
    assert nat.fv.nat is not None
    while not nat.finished:
        nat.step()
    return ref, nat


@needs_native
def test_simplec_iterates_with_native_kernels_equal_reference():
    """40 SIMPLEC iterations on skewed triangles (not converged) are unchanged up to round-off."""
    mesh = cartesian_mesh_2d(16, 16, triangulate=True, warp=0.05)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                            max_iter=40, tol=0.0, backend="numpy")
    ref, nat = _run_pair(mesh, case)
    for k in ("x-velocity", "y-velocity", "pressure"):
        assert _rel(nat.fields()[k], ref.fields()[k]) < 1e-10, k


@needs_native
def test_simplec_with_energy_converges_to_the_same_solution():
    """With energy the comparison is made at convergence: this convection-dominated energy system
    is so ill-conditioned for Jacobi-BiCGSTAB that a 1e-15 change of its right-hand side alone
    changes the inner solve by ~1e-2 (measured), but the converged solution cannot depend on it."""
    mesh = cartesian_mesh_2d(16, 16, triangulate=True, warp=0.05)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0], thermal="temperature",
                                                            value=1.0),
                                        "bottom": BoundarySpec("wall", thermal="temperature", value=0.0)},
                            viscosity=0.05, energy=True, conductivity=0.05, max_iter=4000, tol=1e-9, backend="numpy")
    ref, nat = _run_pair(mesh, case)
    assert ref.converged and nat.converged
    for k in ("x-velocity", "y-velocity", "pressure", "temperature"):
        assert _rel(nat.fields()[k], ref.fields()[k]) < 1e-6, k


def test_status_reports_and_disable_context():
    st = native.status()
    assert set(st) >= {"cpu_kernels", "reason"}
    with native.disabled():
        assert FVMesh(cartesian_mesh_2d(4, 4), CPU).nat is None
