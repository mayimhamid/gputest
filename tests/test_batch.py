"""Wall-force report and the concurrent batch runner."""

import os

import numpy as np

from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.fluent_h5 import read_any_mesh
from flowtrace.mesh.unstructured import cartesian_mesh_2d, make_periodic
from flowtrace.runtime.batch import BatchJob, run_batch
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
from flowtrace.unstructured.simple import SimpleSolver

HERE = os.path.dirname(os.path.abspath(__file__))
DUCT = os.path.join(HERE, "..", "examples", "heated_duct_tri.msh")


def test_wall_forces_balance_the_driving_body_force():
    """Periodic channel driven by a body force G: at steady state the walls carry G x area exactly."""
    mesh = make_periodic(cartesian_mesh_2d(8, 16), "left", "right")
    case = UnstructuredCase(boundaries={"bottom": BoundarySpec("wall"), "top": BoundarySpec("wall")}, viscosity=0.1,
                            body_force=[1.0, 0.0, 0.0], max_iter=5000, tol=1e-10, backend="numpy")
    s = SimpleSolver(mesh, case)
    while not s.finished:
        s.step()
    f = s.wall_forces()
    assert set(f) == {"bottom", "top"}
    assert abs(f["bottom"]["total"][0] + f["top"]["total"][0] - 1.0) < 1e-6
    assert abs(f["bottom"]["pressure"][0]) < 1e-9  # walls parallel to the flow: shear only


def _duct_case(mu, iters, tol=1e-12):
    return {"viscosity": mu, "max_iter": iters, "tol": tol}


def test_batch_runs_concurrently_and_matches_single_runs(tmp_path):
    # converged solutions: unconverged iterates differ by ~1e-7 between processes (BLAS threading
    # changes the summation order), a converged solution does not depend on that
    jobs = [BatchJob(name=f"duct_mu{mu}", mesh=DUCT, case=_duct_case(mu, 3000, tol=1e-8), out_dir=str(tmp_path))
            for mu in (0.05, 0.02)]
    res = run_batch(jobs, cpu_workers=2, use_gpu=False, log=lambda m: None)
    assert [r["name"] for r in res["results"]] == ["duct_mu0.05", "duct_mu0.02"]
    mesh, _ = read_any_mesh(DUCT)
    for job, r in zip(jobs, res["results"]):
        assert r["error"] is None and r["device"] == "cpu" and r["converged"]
        assert os.path.exists(r["ip"]) and os.path.exists(r["npz"])
        s = SimpleSolver(mesh, UnstructuredCase(boundaries=default_boundaries(mesh), backend="numpy", **job.case),
                         backend=get_backend("numpy"))
        while not s.finished:
            s.step()
        assert s.converged
        for zone, f in s.wall_forces().items():
            assert np.allclose(r["wall_forces"][zone]["total"], f["total"], rtol=1e-6, atol=1e-10)


def test_batch_reports_a_failing_case_without_stopping():
    jobs = [BatchJob(name="ok", mesh=DUCT, case=_duct_case(0.05, 5)),
            BatchJob(name="bad", mesh=DUCT, case={"boundaries": {"no-such-zone": {"kind": "wall"}}, "max_iter": 5})]
    res = run_batch(jobs, cpu_workers=2, use_gpu=False, log=lambda m: None)
    ok, bad = res["results"]
    assert ok["error"] is None and "no-such-zone" in bad["error"]
