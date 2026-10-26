import json
import math

import numpy as np
import pytest

from flowtrace.validation.gci import gci
from flowtrace.validation.mms import exact, run_mac, run_simplec, sources


def test_manufactured_solution_is_consistent():
    # the exact field is divergence free and vanishes on the walls
    h = 1e-5
    x, y = np.random.default_rng(0).uniform(0.1, 0.9, (2, 50))
    e = exact(x, y)
    ux = (exact(x + h, y)["u"] - exact(x - h, y)["u"]) / (2 * h)
    vy = (exact(x, y + h)["v"] - exact(x, y - h)["v"]) / (2 * h)
    assert np.abs(ux + vy).max() < 1e-6
    t = np.linspace(0, 1, 11)
    for xx, yy in ((t, 0 * t), (t, 0 * t + 1), (0 * t, t), (0 * t + 1, t)):
        w = exact(xx, yy)
        assert np.abs(w["u"]).max() < 1e-12 and np.abs(w["v"]).max() < 1e-12 and np.abs(w["T"]).max() < 1e-12
    # the source reproduces the momentum residual computed by finite differences
    nu = 0.3
    def lap(f):
        return ((exact(x + h, y)[f] - 2 * e[f] + exact(x - h, y)[f])
                + (exact(x, y + h)[f] - 2 * e[f] + exact(x, y - h)[f])) / h ** 2
    def d(f, ax):
        return ((exact(x + h, y)[f] - exact(x - h, y)[f]) if ax == 0 else (exact(x, y + h)[f] - exact(x, y - h)[f])) / (2 * h)
    fu = e["u"] * d("u", 0) + e["v"] * d("u", 1) + d("p", 0) - nu * lap("u")
    assert np.abs(fu - sources(x, y, nu, 0.1)["u"]).max() < 1e-3


def test_mms_mac_second_order():
    r = run_mac((16, 32), backend="numpy")
    for q in ("u", "v", "p", "T"):
        assert r.orders[q][-1] > 1.9, (q, r.orders)


def test_mms_simplec_second_order_on_warped_triangles():
    r = run_simplec((10, 20, 40), triangulate=True, warp=0.05, energy=False)
    assert r.orders["u"][-1] > 1.8 and r.orders["v"][-1] > 1.8, r.orders
    assert r.orders["p"][-1] > 1.1, r.orders


def test_least_squares_pressure_force_is_inconsistent_on_triangles():
    # documents why the Green-Gauss face-pressure force is the default
    gg = run_simplec((10, 20), triangulate=True, energy=False)
    ls = run_simplec((10, 20), triangulate=True, energy=False, pressure_gradient="least-squares")
    assert gg.errors["u"][-1] < 0.3 * ls.errors["u"][-1]


def test_gci_recovers_exact_order_and_limit():
    for r in (2.0, 1.5):
        h = np.array([1.0, r, r * r]) * 0.01
        phi = 3.0 + 5.0 * h ** 2
        g = gci(phi[0], phi[1], phi[2], h[0], h[1], h[2])
        assert abs(g.observed_order - 2.0) < 1e-8
        assert abs(g.extrapolated - 3.0) < 1e-10
        assert abs(g.asymptotic_ratio - 1.0) < 1e-6
    with pytest.raises(ValueError):
        gci(1.0, 1.0, 2.0, 1, 2, 4)


def test_benchmark_runner_and_report(tmp_path):
    from flowtrace.validation.benchmarks import REGISTRY, run_benchmarks

    assert {"cavity_re100_mac", "mms_mac", "gci_natural_convection"} <= set(REGISTRY)
    data = run_benchmarks(ids=["projection_divergence", "taylor_green_order"], out_dir=str(tmp_path), verbose=False)
    assert data["summary"] == {"total": 2, "passed": 2}
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["machine"]["flowtrace"]
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Taylor-Green" in html
    try:  # plots need the optional [report] extra
        import matplotlib  # noqa: F401

        assert "<svg" in html
    except ImportError:
        pass
    md = (tmp_path / "validation_table.md").read_text(encoding="utf-8")
    assert "`taylor_green_order`" in md and "pass" in md
    # a partial rerun merges into the existing results.json
    run_benchmarks(ids=["projection_divergence"], out_dir=str(tmp_path), verbose=False, report=False)
    assert len(json.loads((tmp_path / "results.json").read_text())["results"]) == 2


def test_failing_benchmark_is_reported_not_raised(tmp_path):
    from flowtrace.validation import benchmarks as B

    @B.benchmark("_broken", "broken on purpose", "test", tiers=())
    def _broken(ctx):
        raise RuntimeError("boom")

    try:
        data = B.run_benchmarks(ids=["_broken"], verbose=False, report=False)
        assert data["results"][0]["error"].startswith("RuntimeError") and not data["results"][0]["passed"]
    finally:
        B.REGISTRY.pop("_broken")


TRANSCRIPT = """
Welcome to ANSYS Fluent (synthetic transcript written for this test)
  iter  continuity  x-velocity  y-velocity     time/iter
      1  1.0000e+00  2.1000e-03  1.9000e-03  0:00:03   99
      2  6.2000e-01  1.5000e-03  1.2000e-03  0:00:02   98
    ...
    137  9.8000e-06  1.1000e-07  1.0000e-07  0:00:00   0
!  137 solution is converged
"""


def test_parse_transcript():
    from flowtrace.fluent.warmstart_study import parse_transcript

    r = parse_transcript(TRANSCRIPT)
    assert r["iterations"] == 137 and r["converged"] and r["first_iteration"] == 1
    assert r["final_residuals"]["continuity"] == pytest.approx(9.8e-6)
    assert parse_transcript("no residual table here")["iterations"] is None


def test_ip_interpolation_onto_another_mesh(tmp_path):
    from flowtrace.fluent.export import write_interpolation_file
    from flowtrace.fluent.interpolate import interpolate_ip_file

    g = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    p = tmp_path / "s.ip"
    write_interpolation_file(str(p), pts, {"x-velocity": 2 * pts[:, 0] + pts[:, 1], "pressure": pts[:, 1]})
    tgt = np.array([[0.33, 0.71], [0.5, 0.5], [1.2, 0.5]])  # last point is outside the hull
    f = interpolate_ip_file(str(p), tgt)
    assert np.allclose(f["x-velocity"][:2], 2 * tgt[:2, 0] + tgt[:2, 1])
    assert np.isclose(f["x-velocity"][2], 2.5)  # nearest source point (1, 0.5)


def test_warm_start_proxy_small(tmp_path, monkeypatch):
    from flowtrace.fluent import warmstart_study as W

    monkeypatch.setitem(W.CASES, "tiny", {"title": "tiny cavity", "preset": "cavity", "params": {"re": 100, "n": 32},
                                          "target": {"kind": "tri", "n": 16}})
    r = W.run_proxy_case("tiny", str(tmp_path / "tiny"), backend="numpy", tol=1e-5)
    assert r.cold_converged and r.warm_converged
    assert r.warm_iterations < r.cold_iterations
    assert r.max_velocity_difference <= W.FIELD_TOL
    assert r.cold_iterations_tol and r.cold_iterations <= r.cold_iterations_tol + 10
    for f in ("flowtrace.msh", "flowtrace.ip", "cold.jou", "warm.jou", "case.json"):
        assert (tmp_path / "tiny" / f).exists(), f
    jou = (tmp_path / "tiny" / "warm.jou").read_text()
    assert "/file/interpolate/read-data" in jou and "convergence-criteria" in jou
    assert "/file/interpolate" not in (tmp_path / "tiny" / "cold.jou").read_text()
    md = W.render_markdown({"proxy": [dict(r.__dict__, speedup=r.speedup)],
                            "fluent": [{"case": "tiny", "status": "not-run", "message": "no licence",
                                        "cold_iterations": None, "warm_iterations": None, "speedup": None}],
                            "tolerance": 1e-5, "machine": {}})
    assert "Not measured" in md and "_placeholder_" in md
    assert not math.isnan(r.speedup)
