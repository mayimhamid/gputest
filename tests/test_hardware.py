from flowtrace.hardware.backend import available_backends, get_backend
from flowtrace.hardware.memory import estimate_structured, estimate_unstructured
from flowtrace.hardware.planner import plan_structured
from flowtrace.hardware.probe import probe_system
from flowtrace.structured.case import cylinder_wake, lid_driven_cavity


def test_probe_never_fails():
    p = probe_system()
    assert p.logical_cores >= 1 and p.ram_total > 0
    d = p.to_dict()
    assert "gpus" in d and "libraries" in d


def test_backends():
    names = available_backends()
    assert "numpy" in names
    be = get_backend("numpy")
    assert be.supports_transform("dct", 2)
    assert be.mem_info()[1] > 0


def test_memory_estimates():
    e = estimate_structured(1024, 1024, True, 8, "fft", True, False, "cpu", 1 << 40)
    assert e.fits and e.required > 1024 * 1024 * 8 * 10
    e = estimate_structured(1 << 14, 1 << 14, True, 8, "fft", True, False, "gpu", 4 << 30)
    assert not e.fits and any("fp32" in a for a in e.advice)
    u = estimate_unstructured(100000, 300000, 3, 8, True, "cpu", 1 << 40)
    assert u.fits and u.required > 0


def test_planner_measures_and_explains():
    plan, solver = plan_structured(lid_driven_cavity(re=100, n=64), autotune=True, steps=2)
    assert plan.backend in available_backends()
    assert plan.measured_ms_per_step
    stages = [p.stage for p in plan.placements]
    assert "Pressure projection" in stages and plan.fmg == [2]
    assert solver.step_count >= 4
    plan, _ = plan_structured(cylinder_wake(n_per_d=4), autotune=False)
    assert "MG-PCG" in plan.pressure


def test_unstructured_memory_estimate_is_an_upper_bound():
    """The pre-run estimate must not be below what a run really allocates (CPU, tracemalloc)."""
    import tracemalloc

    from flowtrace.hardware.memory import estimate_unstructured
    from flowtrace.mesh.unstructured import cartesian_mesh_2d
    from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
    from flowtrace.unstructured.simple import SimpleSolver

    mesh = cartesian_mesh_2d(96, 96)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, max_iter=100, tol=0,
                            backend="numpy")
    tracemalloc.start()
    s = SimpleSolver(mesh, case)
    for _ in range(12):
        s.step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    est = estimate_unstructured(mesh.n_cells, mesh.n_faces, mesh.dim, 8, False, "cpu", 0)
    assert est.required >= peak
