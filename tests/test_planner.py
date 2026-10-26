"""Measured backend placement for the unstructured solver, and the automatic SIMPLEC relaxation."""

import numpy as np
import pytest

import fake_gpu
from flowtrace.hardware import backend as backend_mod
from flowtrace.hardware import probe as probe_mod
from flowtrace.hardware.planner import plan_unstructured, time_iterations
from flowtrace.mesh.unstructured import cartesian_mesh_2d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.simple import SimpleSolver, auto_relax_p


def _cavity(n=12, backend="auto", **kw):
    mesh = cartesian_mesh_2d(n, n)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                            max_iter=10 ** 6, tol=0.0, backend=backend, **kw)
    return mesh, case


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


def test_explicit_backend_is_honoured_without_timing():
    mesh, case = _cavity(backend="numpy")
    plan, solver = plan_unstructured(mesh, case)
    assert plan.backend == "numpy" and plan.reason == "user choice"
    assert not plan.measured_ms_per_iteration and solver.iteration == 0


def test_gpu_not_timed_when_cpu_beats_its_floor(gpu):
    mesh, case = _cavity()
    cal = {"cpu_us_per_cell": 1.0, "gpu_floor_ms": 1e6}
    plan, solver = plan_unstructured(mesh, case, calibration=cal)
    assert plan.device == "cpu" and "cupy" in plan.skipped and "not timed" in plan.skipped["cupy"]
    assert list(plan.measured_ms_per_iteration) == [plan.backend]
    assert solver.be.device == "cpu"


def test_gpu_never_initialised_when_cached_floor_rules_it_out(gpu):
    backend_mod._cache.clear()  # forget the device the fixture created
    mesh, case = _cavity()
    plan, _ = plan_unstructured(mesh, case, calibration={"cpu_us_per_cell": 1.0, "gpu_floor_ms": 1e6})
    assert plan.device == "cpu"
    assert not any(name == "cupy" for name, _ in backend_mod._cache)


def test_both_timed_and_fastest_kept(gpu):
    mesh, case = _cavity()
    plan, solver = plan_unstructured(mesh, case, calibration={"cpu_us_per_cell": 1.0, "gpu_floor_ms": 0.0}, max_steps=3)
    t = plan.measured_ms_per_iteration
    assert "cupy" in t and len(t) == 2
    assert plan.backend == min(t, key=t.get) and solver.be.name == plan.backend
    assert plan.timed_iterations[plan.backend] == 3


def test_cost_model_without_autotune(gpu):
    mesh, case = _cavity()
    p_small, s = plan_unstructured(mesh, case, autotune=False, calibration={"cpu_us_per_cell": 1.0, "gpu_floor_ms": 50.0})
    assert p_small.device == "cpu" and s.iteration == 0 and "cost model" in p_small.reason
    p_big, _ = plan_unstructured(mesh, case, autotune=False, calibration={"cpu_us_per_cell": 1e3, "gpu_floor_ms": 50.0})
    assert p_big.backend == "cupy"


def test_kept_solver_continues_exactly():
    """The iterations spent timing are real iterations: continuing the kept solver gives the
    same field as an untimed solver run for the same total count."""
    mesh, case = _cavity(backend="numpy")
    _, a = plan_unstructured(mesh, case)
    ts = time_iterations(a, 4, warmup=1)
    assert len(ts) == 4 and a.iteration == 5
    b = SimpleSolver(mesh, case, backend=backend_mod.get_backend("numpy"))
    for _ in range(5):
        b.step()
    assert np.array_equal(a.U, b.U) and np.array_equal(a.p, b.p)


def test_auto_relax_p_initial_value():
    assert auto_relax_p(0.0) == 1.0 and auto_relax_p(5.0) == 1.0 and auto_relax_p(45.0) == 0.8


def test_guard_recovers_simplec_on_45_degree_sheared_mesh():
    """Uniformly sheared quads (45 deg on every face, as in GAMBIT block meshes around bluff
    bodies): SIMPLEC diverges at alpha_p = 0.8 and 0.5. The guard rolls back to a healthy
    snapshot and lowers alpha_p until the iteration converges."""
    def run(relax_p):
        mesh = cartesian_mesh_2d(24, 24)
        mesh.nodes[:, 0] = mesh.nodes[:, 0] + mesh.nodes[:, 1]
        case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.1,
                                max_iter=3000, tol=1e-6, backend="numpy", relax_p=relax_p)
        s = SimpleSolver(mesh, case)
        while not s.finished and np.isfinite(s.residuals()["raw"].get("continuity", 0.0)):
            s.step()
        return s

    fixed = run(0.8)  # a user-set relaxation is left alone: diverges as before
    assert not fixed.converged and not fixed.guard_events
    s = run(None)
    assert s.fv.non_orthogonality_deg > 40
    assert s.converged and s.guard_events and s.relax_p < 0.5
    assert np.isfinite(s.fields()["velocity-magnitude"]).all()
    assert s.describe()["stability_rollbacks"] == s.guard_events
