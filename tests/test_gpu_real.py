"""The same GPU checks as test_gpu_paths.py, but on a real CUDA device (skipped without one)."""

import numpy as np
import pytest

from flowtrace.hardware import backend as backend_mod


def _real_gpu():
    try:
        import cupy  # noqa: F401

        if getattr(cupy, "__fake__", False):
            return None
        be = backend_mod.Backend("cupy")
        return be
    except Exception:
        return None


GPU = _real_gpu()
pytestmark = pytest.mark.skipif(GPU is None, reason="no usable CUDA GPU / CuPy")
cpu = backend_mod.get_backend("numpy") if GPU is not None else None


@pytest.mark.parametrize("preset,params", [("cavity", {"n": 48}), ("natural_convection", {"n": 32}),
                                           ("cylinder", {"n_per_d": 6}), ("rayleigh_benard", {"nx": 32, "ny": 16}),
                                           ("channel", {"nx": 64, "ny": 16})])
def test_mac_gpu_matches_cpu_fp64(preset, params):
    from flowtrace.structured.case import make_preset
    from flowtrace.structured.solver import MACSolver

    c = make_preset(preset, **params)
    c.precision = "fp64"
    a = MACSolver(c, backend=cpu, dtype=np.float64)
    b = MACSolver(c, backend=GPU, dtype=np.float64)
    for _ in range(25):
        a.step()
        b.step()
    assert np.abs(a.U - GPU.to_numpy(b.U)).max() < 1e-9
    assert np.abs(a.V - GPU.to_numpy(b.V)).max() < 1e-9
    assert b.max_divergence() < 1e-8


@pytest.mark.parametrize("scheme", ["quick", "vanleer", "upwind", "central"])
@pytest.mark.parametrize("preset,params", [("cavity", {"n": 32}), ("natural_convection", {"n": 24}),
                                           ("cylinder", {"n_per_d": 5}), ("rayleigh_benard", {"nx": 32, "ny": 16})])
def test_fused_cuda_kernels_match_vectorised_cupy(preset, params, scheme):
    from flowtrace.structured import cuda_kernels
    from flowtrace.structured.case import make_preset
    from flowtrace.structured.solver import MACSolver

    if not cuda_kernels.available():
        pytest.skip(f"CUDA kernels not compiled: {cuda_kernels.failure_reason()}")
    c = make_preset(preset, **params)
    c.scheme = scheme
    a = MACSolver(c, backend=GPU, dtype=np.float64)
    b = MACSolver(c, backend=GPU, dtype=np.float64)
    b.use_cuda = False
    assert a.use_cuda
    for _ in range(15):
        a.step()
        b.step()
    assert float(abs(a.U - b.U).max()) < 1e-11 and float(abs(a.V - b.V).max()) < 1e-11
    if a.energy:
        assert float(abs(a.T - b.T).max()) < 1e-11


def test_mac_gpu_fp32_consumer_default():
    from flowtrace.structured.case import lid_driven_cavity
    from flowtrace.structured.solver import MACSolver, resolve_dtype

    c = lid_driven_cavity(n=64)
    dt = resolve_dtype(c, GPU)
    s = MACSolver(c, backend=GPU)
    for _ in range(100):
        s.step()
    assert s.U.dtype == dt
    assert s.max_divergence() < (1e-3 if dt == np.float32 else 1e-10)


def test_separable_solver_gpu():
    from flowtrace.linalg import Axis1D, SeparableSolver

    axes = [Axis1D(32, 0.1, "neumann", "dirichlet"), Axis1D(24, 0.2, "dirichlet", "dirichlet", "node")]
    f = np.random.default_rng(0).standard_normal((32, 24))
    s_gpu = SeparableSolver(axes, GPU)
    s_cpu = SeparableSolver(axes, cpu)
    assert np.abs(GPU.to_numpy(s_gpu.solve(GPU.asarray(f), 1.5)) - s_cpu.solve(f, 1.5)).max() < 1e-10


def test_simplec_gpu_matches_cpu():
    from flowtrace.mesh.unstructured import cartesian_mesh_2d
    from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
    from flowtrace.unstructured.simple import SimpleSolver

    m = cartesian_mesh_2d(16, 16, triangulate=True, warp=0.05)
    # tight inner pressure tolerance: the GPU uses FlowTrace's SA-AMG, the CPU pyamg, so
    # loosely converged corrections would make the iterates (not the solution) differ
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         max_iter=60, tol=1e-12, pressure_tol=1e-9)
    a = SimpleSolver(m, c, backend=cpu)
    b = SimpleSolver(m, c, backend=GPU)
    for _ in range(60):
        a.step()
        b.step()
    assert np.abs(a.fields()["x-velocity"] - b.fields()["x-velocity"]).max() < 1e-6


def test_planner_measures_gpu():
    from flowtrace.hardware.planner import plan_structured
    from flowtrace.structured.case import lid_driven_cavity

    plan, solver = plan_structured(lid_driven_cavity(n=128), autotune=True, steps=2)
    assert "cupy" in plan.measured_ms_per_step
    assert plan.backend in plan.measured_ms_per_step
