import itertools

import numpy as np
import pytest
import scipy.sparse as sp

from flowtrace.hardware.backend import get_backend
from flowtrace.linalg import Axis1D, MGPoisson2D, SeparableSolver, bicgstab, pcg
from flowtrace.linalg.amg import SmoothedAggregationAMG

be = get_backend("numpy")
rng = np.random.default_rng(0)


@pytest.mark.parametrize("lo,hi", list(itertools.product(["neumann", "dirichlet"], repeat=2)))
@pytest.mark.parametrize("sigma", [0.0, 2.5])
def test_fast_transforms_match_dense_eig(lo, hi, sigma):
    axes = [Axis1D(17, 0.1, lo, hi), Axis1D(12, 0.07, "dirichlet", "neumann")]
    f = rng.standard_normal((17, 12))
    if sigma == 0 and lo == hi == "neumann":
        pass  # second axis has a Dirichlet end: never singular
    fast = SeparableSolver(axes, be, methods="auto")
    eig = SeparableSolver(axes, be, methods="eig")
    u1, u2 = fast.solve(f, sigma), eig.solve(f, sigma)
    assert np.abs(u1 - u2).max() < 1e-12
    assert np.abs(eig.apply(u2, sigma) - f).max() < 1e-10
    assert fast.methods[0] != "eig"


def test_node_dirichlet_and_periodic():
    s = SeparableSolver([Axis1D(15, 0.2, "dirichlet", "dirichlet", "node"), Axis1D(16, 0.3, "periodic", "periodic")], be)
    f = rng.standard_normal((15, 16))
    u = s.solve(f, 1.0)
    assert s.methods == ["dst1", "fft"]
    assert np.abs(s.apply(u, 1.0) - f).max() < 1e-10


def test_singular_neumann_gives_zero_mean():
    s = SeparableSolver([Axis1D(20, 0.1), Axis1D(10, 0.2)], be)
    f = rng.standard_normal((20, 10))
    f -= f.mean()
    u = s.solve(f)
    assert abs(u.mean()) < 1e-12
    assert np.abs(s.apply(u) - f).max() < 1e-10


def test_nonuniform_axis():
    faces = np.cumsum(np.r_[0, rng.uniform(0.5, 1.5, 24)])
    s = SeparableSolver([Axis1D(24, faces=faces, lo="dirichlet", hi="neumann"), Axis1D(9, 0.3)], be)
    f = rng.standard_normal((24, 9))
    assert s.methods[0] == "eig"
    assert np.abs(s.apply(s.solve(f, 0.3), 0.3) - f).max() < 1e-9


def _conductances(nx, ny, right_dirichlet=True):
    cx = np.ones((nx + 1, ny))
    cy = np.ones((nx, ny + 1))
    cx[0] = 0.0
    cx[-1] = 2.0 if right_dirichlet else 0.0
    cy[:, 0] = cy[:, -1] = 0.0
    return cx, cy


def test_multigrid_matches_separable():
    nx, ny = 64, 48
    h = 1.0 / nx
    cx, cy = _conductances(nx, ny)
    mg = MGPoisson2D(cx, cy, 0.0, np.ones((nx, ny), bool), be)
    f = rng.standard_normal((nx, ny))
    x, info = mg.solve(f * h * h, tol=1e-11)
    assert info.converged and info.iterations < 15
    ref = SeparableSolver([Axis1D(nx, h, "neumann", "dirichlet"), Axis1D(ny, h)], be).solve(f)
    assert np.abs(x - ref).max() / np.abs(ref).max() < 1e-8


def test_multigrid_singular_with_obstacle():
    nx, ny = 64, 32
    cx, cy = _conductances(nx, ny, right_dirichlet=False)
    act = np.ones((nx, ny), bool)
    act[20:30, 10:20] = False
    mg = MGPoisson2D(cx, cy, 0.0, act, be)
    f = rng.standard_normal((nx, ny))
    x, info = mg.solve(f, tol=1e-10)
    assert info.converged
    b = mg.project(f * act)
    assert np.abs(mg.apply(x) * act - b).max() < 1e-7
    assert np.all(x[~act] == 0)


def _laplacian(n):
    T = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    I = sp.identity(n)
    return (sp.kron(T, I) + sp.kron(I, T)).tocsr()


def test_amg_pcg_and_bicgstab():
    A = _laplacian(40)
    b = rng.standard_normal(A.shape[0])
    amg = SmoothedAggregationAMG(A, be)
    assert amg.n_levels >= 2
    x, info = pcg(lambda v: A @ v, b, np, M=amg, tol=1e-10, maxiter=100)
    assert info.converged and info.iterations < 30
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-9
    N = A + sp.diags(rng.uniform(0, 0.5, A.shape[0])) + 0.3 * sp.eye(A.shape[0], k=1)
    x, info = bicgstab(lambda v: N @ v, b, np, M=lambda v: v / N.diagonal(), tol=1e-10, maxiter=500)
    assert info.converged
