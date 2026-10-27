"""SimpleSolver.diagnostics(): definitions checked against the systems the iteration solved.

Every assertion follows from an algebraic identity of the SIMPLE/SIMPLEC iteration or from
the stopping rule of its linear solvers (see flowtrace/unstructured/diagnostics.py); none
is a tuned threshold.
"""

import math

import numpy as np
import pytest

from flowtrace.mesh.unstructured import cartesian_mesh_2d, extrude_to_3d
from flowtrace.unstructured import diagnostics as D
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.conjugate import make_solver
from flowtrace.unstructured.simple import SimpleSolver

EPS = np.finfo(np.float64).eps


def _cavity(n=16, warp=0.05, **kw):
    kw.setdefault("max_iter", 1000)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, backend="numpy", **kw)
    return SimpleSolver(cartesian_mesh_2d(n, n, warp=warp), case)


def _duct(energy=True, **kw):
    m = cartesian_mesh_2d(24, 8, 3.0, 1.0, names=("in", "out", "bottom", "top"), warp=0.03)
    case = UnstructuredCase(boundaries={"in": BoundarySpec("inlet", velocity=[1.0, 0.0], value=1.0),
                                        "out": BoundarySpec("outlet"),
                                        "bottom": BoundarySpec("wall", thermal="temperature", value=0.0),
                                        "top": BoundarySpec("wall")}, energy=energy, backend="numpy", **kw)
    return SimpleSolver(m, case)


def _run(s, n):
    for _ in range(n):
        s.step()
    return s


def test_unavailable_before_the_first_iteration_and_after_a_restore():
    s = _cavity()
    assert s.diagnostics()["available"] is False
    _run(s, 3)
    assert s.diagnostics()["available"] is True
    s.load_state(s.state_dict())
    assert s.diagnostics()["available"] is False  # a restored state is not the result of an iteration


def test_captured_system_reproduces_the_existing_scaled_residual():
    """Matrix/RHS consistency: the captured A, b and old velocity give exactly the momentum
    residual the solver reports (sum|b - A u^n| / max(sum|a_P u^n|, floor))."""
    s = _run(_cavity(), 12)
    cap = s._capture
    A, U = cap["A"], cap["U_old"]
    diag = A.diagonal()
    num = [np.abs(cap["b"][k] - A @ U[:, k]).sum() for k in range(2)]
    den = [np.abs(diag * U[:, k]).sum() for k in range(2)]
    floor = 1e-3 * max(max(den), 1e-300)
    for k, name in enumerate(("x-velocity", "y-velocity")):
        assert num[k] / max(den[k], floor) == pytest.approx(s.residuals()["scaled"][name], rel=1e-12)


def test_relaxed_and_assembled_systems_share_the_presolve_residual():
    """b_r - A_r u^n = b - A u^n algebraically; only round-off may separate them."""
    s = _run(_cavity(), 8)
    cap = s._capture
    for k in range(2):
        u0 = cap["U_old"][:, k]
        r_pre = cap["b"][k] - cap["A"] @ u0
        r_rel = cap["br"][k] - cap["Ar"] @ u0
        scale = np.abs(cap["br"][k]).max() + np.abs(cap["Ar"].data).max() * np.abs(u0).max()
        assert np.abs(r_rel - r_pre).max() <= 64 * EPS * scale


def test_solved_system_residual_matches_the_linear_solver():
    """The explicitly computed reduction ||b_r - A_r u*|| / ||b_r - A_r u^n|| equals BiCGSTAB's
    recurrence residual up to round-off, and meets its tolerance (0.1) when it converged."""
    s = _run(_cavity(), 20)
    d = s.diagnostics()
    for comp in ("x-velocity", "y-velocity"):
        sol = d["momentum"][comp]["solve"]
        lin = sol["linear_solver"]
        assert sol["true_reduction_l2"] == pytest.approx(lin["reported_residual"], rel=1e-6)
        if lin["converged"]:
            assert sol["true_reduction_l2"] < 0.1 * (1 + 1e-6)


def test_corrected_imbalance_is_the_pressure_correction_residual():
    """net outflow of the corrected fluxes = m* + A_p p' = -(rhs - A_p p') of A_p p' = -m*."""
    s = _run(_duct(energy=False), 15)
    cap = s._capture
    corrected = s.fv.net_outflow(cap["F"], cap["Fb"])
    residual = -cap["mass"] - cap["Ap"] @ cap["pp"]
    assert np.abs(corrected + residual).max() <= 1e-12 * np.abs(cap["mass"]).max()
    d = s.diagnostics()
    pc = d["pressure_correction"]
    assert d["continuity"]["corrected"]["abs_l2"] == pytest.approx(pc["abs_l2"], rel=1e-9)
    # the PCG stopping rule bounds the corrected imbalance by pressure_tol times the predictor one
    assert pc["relative_l2"] == pytest.approx(pc["linear_solver"]["reported_residual"], rel=1e-6)
    if pc["linear_solver"]["converged"]:
        assert d["continuity"]["corrected"]["abs_l2"] <= s.case.pressure_tol * d["continuity"]["predictor"]["abs_l2"] \
            * (1 + 1e-6)


def test_global_boundary_imbalance_is_the_sum_of_cell_imbalances():
    s = _run(_duct(energy=False), 30)
    c = s.diagnostics()["continuity"]
    g = c["global_boundary"]
    assert g["throughflow"] > 0 and g["relative"] >= 0
    assert g["signed"] == pytest.approx(c["corrected"]["sum_of_cells"], abs=1e-12 * g["throughflow"])
    # a zero global sum does not imply local balance: the local metric is reported separately
    assert c["corrected"]["local_max_relative"] > 0


def test_zero_scale_system_reports_zero_not_nan():
    """No driving at all: every term is exactly zero, so residuals are exactly zero (0/0 = 0)."""
    case = UnstructuredCase(boundaries={}, backend="numpy", max_iter=10)
    s = _run(SimpleSolver(cartesian_mesh_2d(6, 6), case), 2)
    d = s.diagnostics()
    for comp in ("x-velocity", "y-velocity"):
        for stage in ("pre_solve", "solve"):
            rep = d["momentum"][comp][stage]
            assert rep["row_scaled"] == 0.0 and rep["active_rows"] == 0 and rep["excluded_rows"] == s.nc
            assert rep["nonfinite_rows"] == 0
    assert d["continuity"]["predictor"]["local_max_relative"] == 0.0
    assert d["continuity"]["global_boundary"] == {"signed": 0.0, "throughflow": 0.0, "relative": 0.0}
    assert d["field_change"]["velocity"] == 0.0 and d["field_change"]["pressure"] == 0.0
    assert d["health"]["finite"]


def test_small_scale_rows_are_excluded_against_a_shared_reference():
    """Synthetic rows: exclusion uses gamma_K times the largest scale over *all* components."""
    xp, u, K = np, D.unit_roundoff(np.float64), 6
    s_big = np.array([1.0, 2.0, 4.0])
    s_tiny = np.array([1e-20, 0.0, 3e-16])  # all at or below gamma_6 * 4 ~ 2.7e-15
    r_big = np.array([1e-3, -4e-3, 0.0])
    r_tiny = np.array([1e-20, 0.0, -3e-16])
    sc = D._Scalars(xp, lambda a: a)
    D._row_set(sc, xp, "t", [("big", r_big, s_big), ("tiny", r_tiny, s_tiny)], K, u)
    vals = sc.fetch()
    centres = np.zeros((3, 2))
    big, tiny = D._row_report(vals, "t", "big", centres), D._row_report(vals, "t", "tiny", centres)
    assert big["row_scaled"] == pytest.approx(2e-3) and big["worst_cell"] == 1 and big["active_rows"] == 3
    assert tiny["active_rows"] == 0 and tiny["excluded_rows"] == 3 and tiny["row_scaled"] == 0.0
    assert tiny["excluded_abs_max"] == pytest.approx(3e-16)
    assert vals["t|s_ref"] == 4.0
    assert D.gamma(K, u) == pytest.approx(K * u / (1 - K * u))


def test_ratio_convention():
    assert D.ratio(0.0, 0.0) == 0.0 and D.ratio(1.0, 0.0) == math.inf and D.ratio(1.0, 4.0) == 0.25
    assert D.ratio(float("nan"), 1.0) == math.inf and D.ratio(1.0, float("inf")) == math.inf


def test_nonfinite_values_are_counted_and_fail_thresholds():
    s = _run(_cavity(), 5)
    cap = s._capture
    cap["b"][0][3] = np.nan  # test-only corruption of the captured right-hand side
    cap["U_new"][7, 1] = np.inf
    d = s.diagnostics()
    pre = d["momentum"]["x-velocity"]["pre_solve"]
    assert pre["nonfinite_rows"] == 1 and pre["row_scaled"] == math.inf
    assert d["health"]["finite"] is False and d["health"]["nonfinite_values"]["velocity"] == 1
    assert D.summary(d)["finite"] is False


def test_diverging_run_and_stability_guard_rollback_are_flagged():
    def sheared(relax_p):
        mesh = cartesian_mesh_2d(24, 24)
        mesh.nodes[:, 0] = mesh.nodes[:, 0] + mesh.nodes[:, 1]
        case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.1,
                                max_iter=3000, tol=1e-6, backend="numpy", relax_p=relax_p)
        return SimpleSolver(mesh, case)

    s = sheared(None)  # automatic relaxation: the guard rolls back
    while not s.guard_events and not s.finished:
        s.step()
    assert s.guard_events
    d = s.diagnostics()
    assert d["available"] and d["rolled_back"] is True
    assert d["field_change"]["velocity"] is None  # the discarded iteration has no meaningful change
    s.step()
    assert s.diagnostics()["rolled_back"] is False

    f = sheared(0.8)  # a user-set relaxation diverges until values stop being finite
    with np.errstate(all="ignore"):
        while np.isfinite(f.residuals()["raw"].get("continuity", 0.0)) and not f.finished:
            f.step()
        d = f.diagnostics()
    assert d["available"] and not d["health"]["finite"]


def test_energy_diagnostics_and_field_change_definition():
    s = _run(_duct(), 10)
    d = s.diagnostics()
    e = d["energy"]
    assert e["pre_solve"]["active_rows"] > 0 and e["solve"]["linear_solver"]["iterations"] >= 0
    cap = s._capture
    dU = np.linalg.norm(cap["U_new"] - cap["U_old"], axis=1).max()
    U_ref = max(np.linalg.norm(cap["U_new"], axis=1).max(), np.linalg.norm(cap["U_old"], axis=1).max(),
                np.linalg.norm(s.ub_fixed, axis=1).max())
    assert d["field_change"]["velocity"] == pytest.approx(dU / U_ref, rel=1e-12)
    p_ref = max(np.ptp(cap["p_new"]), s.case.density * U_ref ** 2)
    assert d["field_change"]["pressure"] == pytest.approx(np.abs(cap["p_new"] - cap["p_old"]).max() / p_ref,
                                                          rel=1e-12)
    T_old, T_new = cap["energy"]["T_old"], cap["energy"]["T_new"]
    T_ref = max(T_new.max(), 1.0) - min(T_new.min(), 0.0)  # fixed walls at 0, inlet at 1
    assert d["field_change"]["temperature"] == pytest.approx(np.abs(T_new - T_old).max() / T_ref, rel=1e-12)


def test_diagnostics_converge_towards_zero_with_the_outer_iteration():
    """Not a monotonicity requirement per iteration: compare an early and a converged state."""
    s = _cavity(n=12, tol=1e-8)
    _run(s, 5)
    early = D.summary(s.diagnostics())
    while not s.finished:
        s.step()
    late = D.summary(s.diagnostics())
    assert s.converged
    for k in ("momentum_pre_row_scaled", "continuity_predictor_local_max", "velocity_change"):
        assert late[k] < 1e-3 * early[k]


def test_turbulence_and_conjugate_solvers_expose_diagnostics():
    m = cartesian_mesh_2d(20, 8, 3.0, 1.0, names=("in", "out", "bottom", "top"))
    case = UnstructuredCase(boundaries={"in": BoundarySpec("inlet", velocity=[1.0, 0.0]), "out": BoundarySpec("outlet"),
                                        "bottom": BoundarySpec("wall"), "top": BoundarySpec("wall")},
                            viscosity=1e-4, turbulence="k-omega-sst", backend="numpy", max_iter=50)
    s = _run(SimpleSolver(m, case), 5)
    ch = s.diagnostics()["field_change"]
    assert ch["k"] is not None and ch["omega"] is not None

    from flowtrace.mesh.unstructured import set_cell_zone

    mesh = set_cell_zone(cartesian_mesh_2d(12, 12, names=("left", "right", "bottom", "top")),
                         lambda x: x[:, 1] < 0.25, "slab", "solid")
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0]),
                                        "bottom": BoundarySpec("wall", thermal="temperature", value=1.0)},
                            energy=True, solids={"slab": {"conductivity": 1.0}}, backend="numpy", max_iter=20)
    cs = make_solver(mesh, case)
    _run(cs, 3)
    d = cs.diagnostics()
    assert d["available"] and "scope" in d and "energy" not in d  # the conjugate solver owns the energy equation


def test_quasi_2d_flow_on_a_3d_mesh_is_finite_and_defined():
    m = extrude_to_3d(cartesian_mesh_2d(8, 8), 2, 0.25)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, backend="numpy",
                            max_iter=20)
    s = _run(SimpleSolver(m, case), 6)
    d = s.diagnostics()
    w = d["momentum"]["z-velocity"]["pre_solve"]
    assert w["nonfinite_rows"] == 0 and 0.0 <= w["row_scaled"] <= 1.0
    assert d["health"]["finite"]


def test_divergence_report_names_the_guard_state_and_the_skew():
    from flowtrace.runtime.engine import divergence_context

    def sheared(**kw):
        mesh = cartesian_mesh_2d(24, 24)
        mesh.nodes[:, 0] = mesh.nodes[:, 0] + mesh.nodes[:, 1]  # 45 degrees on every face
        case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.1,
                                max_iter=3000, backend="numpy", **kw)
        return SimpleSolver(mesh, case)

    def diverge(s):
        with np.errstate(all="ignore"):
            while np.isfinite(s.residuals()["raw"].get("continuity", 0.0)) and not s.finished:
                s.step()
        return s

    fixed = diverge(sheared(relax_p=0.8))
    msg = divergence_context(fixed)
    assert "set explicitly" in msg and "45.0 deg" in msg and "cos^2" in msg
    s = sheared()
    s.GUARD_MIN_RELAX = 0.45  # the guard reaches its floor before the stable 0.29
    s.GUARD_FALLBACK = False  # ... and may not fall back to SIMPLE
    diverge(s)
    assert s.guard_exhausted_at is not None and s.guard_events
    assert "could not go lower" in divergence_context(s) and s.describe()["stability_guard_exhausted_at"]


def test_pressure_matrix_meets_the_conditions_for_cg():
    from flowtrace.unstructured.diagnostics import pressure_matrix_properties

    s = _cavity(n=10)
    assert pressure_matrix_properties(s) is None  # nothing assembled yet
    _run(s, 3)
    pm = pressure_matrix_properties(s)
    assert pm["cg_applicable"] and pm["symmetric"] and pm["singular"]  # closed cavity: pressure fixed only up to a constant
    d = _run(_duct(energy=False), 3)
    pm = pressure_matrix_properties(d)
    # interior rows have margin exactly 0 (sum of off-diagonals = diagonal), so only round-off may go below it
    assert pm["cg_applicable"] and not pm["singular"] and pm["min_dominance_margin"] >= -64 * D.unit_roundoff(np.float64)


def test_job_reports_divergence_and_input_errors_in_one_line(tmp_path):
    """A diverged run and an input error end the job with a one-line, actionable log entry (the
    traceback is kept for unexpected faults and FLOWTRACE_DEBUG=1); the CLI maps them to exit
    statuses 1 and 2."""
    from flowtrace.runtime.engine import EventHub, Job

    mesh = cartesian_mesh_2d(24, 24)
    mesh.nodes[:, 0] = mesh.nodes[:, 0] + mesh.nodes[:, 1]  # 45 degrees on every face
    case = {"boundaries": {"top": {"kind": "wall", "velocity": [1.0, 0.0]}}, "viscosity": 0.1, "max_iter": 3000,
            "backend": "numpy", "relax_p": 0.8}
    hub = EventHub()
    q = hub.subscribe()
    job = Job({"kind": "unstructured", "mesh_id": "m", "case": case}, hub, str(tmp_path), meshes={"m": mesh})
    job.start()
    job.join()
    errors = []
    while not q.empty():
        ev = q.get()
        if ev.get("type") == "log" and ev.get("level") == "error":
            errors.append(ev["msg"])
    assert job.status == "error" and isinstance(job.exception, FloatingPointError)
    assert len(errors) == 1 and "Traceback" not in errors[0]
    assert "diverged" in errors[0] and "non-orthogonality 45.0 deg" in errors[0] and "set explicitly" in errors[0]


def test_cli_reports_an_unavailable_backend_in_one_line():
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("CuPy is installed here, so the GPU backend may well be available")
    r = subprocess.run([sys.executable, "-m", "flowtrace", "run", "cavity", "--n", "16", "--backend", "cupy",
                        "--max-steps", "2"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and "flowtrace run: error: BackendUnavailable" in r.stderr
    assert "Traceback" not in r.stdout + r.stderr
