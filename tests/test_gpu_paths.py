"""Run every GPU-only branch against a NumPy-backed fake CuPy (see fake_gpu.py)."""

import numpy as np
import pytest

import fake_gpu
from flowtrace.hardware import backend as backend_mod
from flowtrace.hardware import probe as probe_mod


@pytest.fixture()
def gpu():
    fake_gpu.install()
    backend_mod._cache.clear()
    probe_mod._CACHED = None
    try:
        yield backend_mod.get_backend("cupy")
    finally:
        fake_gpu.uninstall()
        backend_mod._cache.clear()
        probe_mod._CACHED = None


def test_probe_and_backend_see_the_gpu(gpu):
    assert gpu.is_gpu and gpu.device == "gpu"
    prof = probe_mod.probe_system()
    g = prof.best_gpu
    assert g is not None and g.usable and g.tensor_cores and g.rt_cores
    assert not g.is_datacenter_fp64 and g.mem_bandwidth_gbs > 200
    assert gpu.mem_info() == (6 << 30, 8 << 30)
    assert not gpu.supports_transform("dct", 4) and gpu.supports_transform("dct", 2)
    assert "cupy" in backend_mod.available_backends()


def test_separable_solver_falls_back_to_gemm_on_gpu(gpu):
    from flowtrace.linalg import Axis1D, SeparableSolver

    axes = [Axis1D(16, 0.1, "neumann", "dirichlet"), Axis1D(12, 0.2, "dirichlet", "dirichlet", "node")]
    s_gpu = SeparableSolver(axes, gpu)
    s_cpu = SeparableSolver(axes, backend_mod.get_backend("numpy"))
    assert s_gpu.methods == ["eig", "eig"] and s_cpu.methods == ["dct4", "dst1"]
    f = np.random.default_rng(0).standard_normal((16, 12))
    assert np.abs(gpu.to_numpy(s_gpu.solve(gpu.asarray(f), 1.5)) - s_cpu.solve(f, 1.5)).max() < 1e-12


@pytest.mark.parametrize("preset", ["cavity", "natural_convection", "cylinder", "rayleigh_benard"])
def test_mac_solver_gpu_matches_cpu(gpu, preset):
    from flowtrace.structured.case import make_preset
    from flowtrace.structured.solver import MACSolver

    params = {"cavity": {"n": 32}, "natural_convection": {"n": 32}, "cylinder": {"n_per_d": 6},
              "rayleigh_benard": {"nx": 32, "ny": 16}}[preset]
    c = make_preset(preset, **params)
    c.precision = "fp64"
    a = MACSolver(c, backend=backend_mod.get_backend("numpy"))
    b = MACSolver(c, backend=gpu)
    for _ in range(12):
        a.step()
        b.step()
    assert np.abs(a.U - gpu.to_numpy(b.U)).max() < 1e-9
    r = b.residuals()
    assert all(np.isfinite(v) for v in r["raw"].values())
    f = b.fields()
    assert all(isinstance(v, np.ndarray) and type(v) is np.ndarray for v in f.values())


def test_gpu_defaults_to_fp32_on_consumer_card(gpu):
    from flowtrace.structured.case import lid_driven_cavity
    from flowtrace.structured.solver import MACSolver, resolve_dtype

    c = lid_driven_cavity(n=32)
    assert resolve_dtype(c, gpu) == np.float32
    s = MACSolver(c, backend=gpu)
    for _ in range(20):
        s.step()
    assert s.U.dtype == np.float32 and s.max_divergence() < 1e-3


def test_simple_solver_on_gpu_matches_cpu(gpu):
    from flowtrace.mesh.unstructured import cartesian_mesh_2d
    from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
    from flowtrace.unstructured.simple import SimpleSolver

    m = cartesian_mesh_2d(12, 12, triangulate=True)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         max_iter=40, tol=1e-12, energy=False)
    a = SimpleSolver(m, c, backend=backend_mod.get_backend("numpy"))
    b = SimpleSolver(m, c, backend=gpu)
    for _ in range(40):
        a.step()
        b.step()
    fa, fb = a.fields(), b.fields()
    assert np.abs(fa["x-velocity"] - fb["x-velocity"]).max() < 1e-6
    assert "FlowTrace SA-AMG" in b.describe()["pressure_solver"]


def test_planner_and_engine_with_gpu(gpu, tmp_path):
    from flowtrace.hardware.planner import plan_structured
    from flowtrace.runtime.engine import EventHub, Job
    from flowtrace.structured.case import lid_driven_cavity

    plan, solver = plan_structured(lid_driven_cavity(n=64), autotune=True, steps=1)
    assert "cupy" in plan.measured_ms_per_step
    assert any(p.stage == "Tensor Cores" for p in plan.placements) if plan.backend == "cupy" else True
    job = Job({"kind": "structured", "preset": "cavity", "params": {"n": 32},
               "overrides": {"backend": "cupy", "max_steps": 40}, "fmg": False}, EventHub(), str(tmp_path))
    job.start()
    job.join()
    assert job.status in ("finished", "converged"), job.error
    files = job.export_package(str(tmp_path / "out"))
    assert open(files["ip"]).readline().strip() == "3"


def test_sst_and_p1_on_gpu_match_cpu(gpu):
    from flowtrace.unstructured.presets import radiation_slab, turbulent_channel
    from flowtrace.unstructured.simple import SimpleSolver

    for make in (lambda: turbulent_channel(ny=24), lambda: radiation_slab(nx=12)):
        mesh, case, _ = make()
        # the stand-in GPU runs the NumPy code path, so compare with the CPU *reference* path: the
        # native C++ kernels round differently, and 15 unconverged SST iterations amplify that
        # (measured 4.5e-12 -> 1.1e-5 in k); native vs reference is covered by test_native.py
        from flowtrace import native

        with native.disabled():
            a = SimpleSolver(mesh, case, backend=backend_mod.get_backend("numpy"))
        b = SimpleSolver(mesh, case, backend=gpu)
        for _ in range(15):
            a.step()
            b.step()
        fa, fb = a.fields(), b.fields()
        for k in fa:
            assert np.allclose(fa[k], fb[k], rtol=1e-6, atol=1e-9), k
