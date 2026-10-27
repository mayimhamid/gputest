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


def test_auto_precision_is_fp64_unless_fp32_is_shown_admissible(gpu):
    """No FP32 without evidence: 'auto' builds FP64; the planner adds an FP32 candidate on a
    consumer GPU only when FP32 can reach the case's steady tolerance (u32 L/(U dt) <= tol)."""
    from flowtrace.hardware.planner import plan_structured
    from flowtrace.structured.case import lid_driven_cavity, taylor_green
    from flowtrace.structured.solver import MACSolver, fp32_admissible, fp32_round_off_bound, resolve_dtype

    c = lid_driven_cavity(n=32, backend="cupy")  # steady_tol 1e-6: FP32 would stall above it
    assert c.precision == "auto" and resolve_dtype(c, gpu) == np.float64
    plan, s = plan_structured(c, steps=1)
    assert s.dtype == np.float64 and "would stall above the steady tolerance" in plan.precision_reason
    assert any(p.stage == "Numerical precision" for p in plan.placements)
    assert not fp32_admissible(c, s.dt)[0] and fp32_round_off_bound(c, s.dt) > c.steady_tol

    loose = lid_driven_cavity(n=32, backend="cupy")
    loose.steady_tol = 1e-3  # FP32 can meet this: it becomes a timed candidate
    plan, s = plan_structured(loose, steps=1)
    assert plan.precision_reason.startswith("FP32 admissible") and "FP32" in " ".join(plan.notes)

    tg = taylor_green(n=32, backend="cupy")  # transient: no accuracy target to test FP32 against
    plan, s = plan_structured(tg, steps=1)
    assert s.dtype == np.float64 and plan.precision_reason.startswith("FP64: transient")

    forced = lid_driven_cavity(n=32, backend="cupy")
    forced.precision = "fp32"  # an explicit choice is honoured, with a warning
    plan, s = plan_structured(forced, steps=1)
    assert s.dtype == np.float32 and plan.precision_reason == "user choice (fp32)"
    assert any("exceeds steady_tol" in n for n in plan.notes)
    auto = plan_structured(lid_driven_cavity(n=32), steps=1)[0]  # every candidate's decision is in the plan
    assert "would stall" in auto.precision_by_backend["cupy"] and "CPU" in auto.precision_by_backend["numpy"]
    f32 = MACSolver(forced, backend=gpu)
    for _ in range(20):
        f32.step()
    assert f32.U.dtype == np.float32 and f32.max_divergence() < 1e-3


def test_gpu_report_compares_equal_precision(gpu):
    """The speed table times every backend in FP64; the GPU's FP32 time is a separate, labelled entry."""
    from flowtrace.hardware.gpu_report import render_markdown, speed

    table = speed(["numpy", "cupy"], sizes=(16,))["16"]
    assert table["numpy"]["precision"] == "float64" and table["cupy"]["precision"] == "float64"
    assert "ms_per_step_fp32" in table["cupy"] and "ms_per_step_fp32" not in table["numpy"]
    assert table["planner_choice"].endswith("(fp64)")  # cavity: FP32 cannot meet its steady tolerance
    md = render_markdown({"machine": {}, "backends": ["numpy", "cupy"], "micro": {}, "speed": {"16": table}})
    assert "(fp64); FP32 " in md and "not a like-for-like speed-up" in md


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


def test_compressible_solver_on_gpu_matches_cpu(gpu):
    """Density-based solver (array path, SST on) through the CuPy branches vs the NumPy run."""
    from flowtrace.compressible import CompBC, CompressibleCase, CompressibleSolver
    from flowtrace.mesh.unstructured import cartesian_mesh_2d

    m = cartesian_mesh_2d(16, 10, 1.0, 0.2, names=("in", "out", "wall", "top"))
    U = 100.0
    bc = {"in": CompBC(kind="velocity-inlet", velocity=[U, 0], T=300.0),
          "out": CompBC(kind="pressure-outlet", p=101325.0), "top": CompBC(kind="symmetry"),
          "wall": CompBC(kind="wall")}
    out = []
    for be in (gpu, backend_mod.get_backend("numpy")):
        case = CompressibleCase(boundaries=bc, turbulence="k-omega-sst", init_velocity=[U, 0, 0], max_iter=30)
        s = CompressibleSolver(m, case, backend=be)
        if be is gpu:
            assert s.flux_kernel == "array"  # the fake CuPy never loads the real CUDA kernels
        for _ in range(30):
            s.step()
        out.append(s.fields())
    for k in ("pressure", "x-velocity", "viscosity-turb"):
        assert np.allclose(out[0][k], out[1][k], rtol=1e-9, atol=1e-12)


def test_solver_diagnostics_on_gpu_match_cpu(gpu):
    """SimpleSolver.diagnostics() through the CuPy branches (device reductions, one host
    transfer) equals the NumPy evaluation of the very same iteration. (The two backends'
    *iterates* differ legitimately: the GPU path uses FlowTrace's SA-AMG, the CPU path pyamg.)"""
    from flowtrace import native
    from flowtrace.mesh.unstructured import cartesian_mesh_2d
    from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
    from flowtrace.unstructured.diagnostics import summary
    from flowtrace.unstructured.simple import SimpleSolver

    mesh = cartesian_mesh_2d(12, 12, warp=0.05)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, energy=True, max_iter=50)
    with native.disabled():
        g = SimpleSolver(mesh, case, backend=gpu)
        for _ in range(8):
            g.step()
        c = SimpleSolver(mesh, case, backend=backend_mod.get_backend("numpy"))
    h = gpu.to_numpy

    def host(v):
        if hasattr(v, "indptr"):  # a device CSR matrix with the solver's fixed pattern
            return c.fv.matrix_like(np.asarray(h(v.data)))
        if isinstance(v, (list, tuple)):
            return type(v)(host(x) for x in v)
        if isinstance(v, dict):
            return {k: host(x) for k, x in v.items()}
        if hasattr(v, "shape"):
            return np.asarray(h(v))
        return v

    c._capture = host(g._capture)
    dg, dc = g.diagnostics(), c.diagnostics()
    assert dg["available"] and dg["health"]["finite"] and "energy" in dg
    a, b = summary(dg), summary(dc)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-300), k
    assert dg["momentum"]["x-velocity"]["solve"] == dc["momentum"]["x-velocity"]["solve"]
