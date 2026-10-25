import numpy as np
import pytest

from flowtrace.hardware.backend import available_backends, get_backend
from flowtrace.structured.case import (StructuredCase, backward_facing_step, channel_flow, cylinder_wake,
                                       lid_driven_cavity, natural_convection, rayleigh_benard, taylor_green)
from flowtrace.structured.sequencing import coarse_case, prolong
from flowtrace.structured.solver import G, MACSolver
from flowtrace.validation.checks import strouhal, validate_structured
from flowtrace.validation.reference import compare_ghia

be = get_backend("numpy")


def run(s, k=10):
    while not s.finished:
        s.step()
        if s.step_count % k == 0:
            s.residuals()
    return s


def test_taylor_green_second_order():
    errs = []
    for n in (16, 32, 64):
        c = taylor_green(n=n, re=10.0, t_end=0.5, cfl=0.25)
        s = run(MACSolver(c, backend=be))
        errs.append(validate_structured(s)["checks"][0]["value"])
        assert s.max_divergence() < 1e-12
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.8), rates


def test_cavity_re100_matches_ghia():
    s = run(MACSolver(lid_driven_cavity(re=100, n=48), backend=be))
    assert s.converged
    (y, u), (x, v) = s.centerline()
    e = compare_ghia(100, y, u, x, v)
    assert e["u_max_err"] < 0.01 and e["v_max_err"] < 0.015, e


def test_natural_convection_nusselt():
    s = run(MACSolver(natural_convection(ra=1e4, n=48), backend=be))
    assert s.converged
    assert abs(s.nusselt("left") - 2.243) / 2.243 < 0.015
    assert abs(s.nusselt("left") - s.nusselt("right")) < 1e-3


def test_channel_poiseuille():
    s = run(MACSolver(channel_flow(re=50, nx=128, ny=16, length=8.0), backend=be))
    chk = validate_structured(s)["checks"][0]
    assert chk["pass"], chk


def test_obstacle_divergence_free_and_masks():
    s = MACSolver(cylinder_wake(n_per_d=8, t_end=2.0), backend=be)
    run(s)
    assert s.max_divergence() < 1e-6
    # no flow through the obstacle surface: faces with exactly one solid neighbour
    solid = s.solid_h
    U = s.U[G:G + s.nx + 1, G:G + s.ny]
    surf = np.zeros_like(U, dtype=bool)
    surf[1:-1] = solid[1:] ^ solid[:-1]
    assert np.all(U[surf] == 0)
    assert np.all(np.isnan(s.fields(["x-velocity"])["x-velocity"][solid]))


def test_step_and_periodic_cases_run():
    for c in (backward_facing_step(re=100, n_per_h=6, max_steps=50), rayleigh_benard(nx=48, ny=16, t_end=0.01)):
        s = run(MACSolver(c, backend=be))
        assert np.isfinite(s.kinetic_energy())
        assert s.max_divergence() < 1e-6


@pytest.mark.skipif("numba" not in available_backends(), reason="numba not installed")
@pytest.mark.parametrize("make", [lambda: lid_driven_cavity(re=400, n=32), lambda: natural_convection(ra=1e5, n=32),
                                  lambda: cylinder_wake(n_per_d=6), lambda: rayleigh_benard(nx=32, ny=16)])
@pytest.mark.parametrize("scheme", ["quick", "vanleer", "upwind", "central"])
def test_numba_kernels_match_numpy(make, scheme):
    c = make()
    c.scheme = scheme
    a = MACSolver(c, backend=get_backend("numpy"))
    b = MACSolver(c, backend=get_backend("numba"))
    for _ in range(15):
        a.step()
        b.step()
    assert np.abs(a.U - b.U).max() < 1e-9
    assert np.abs(a.V - b.V).max() < 1e-9
    if a.energy:
        assert np.abs(a.T - b.T).max() < 1e-9


def test_fp32_runs():
    c = lid_driven_cavity(re=100, n=32, precision="fp32", max_steps=50)
    s = run(MACSolver(c, backend=be))
    assert s.U.dtype == np.float32
    assert s.max_divergence() < 1e-3


def test_checkpoint_roundtrip():
    c = natural_convection(ra=1e4, n=24, max_steps=30)
    a = MACSolver(c, backend=be)
    for _ in range(20):
        a.step()
    st = a.state_dict()
    b = MACSolver(c, backend=be)
    b.load_state(st)
    for _ in range(5):
        a.step()
        b.step()
    assert np.abs(a.U - b.U).max() < 1e-12 and np.abs(a.T - b.T).max() < 1e-12


def test_fmg_prolongation_is_divergence_free():
    c = lid_driven_cavity(re=100, n=64)
    coarse = MACSolver(coarse_case(c, 2), backend=be)
    for _ in range(200):
        coarse.step()
    fine = MACSolver(c, backend=be)
    prolong(coarse, fine)
    assert fine.max_divergence() < 1e-12
    assert fine.kinetic_energy() > 0.5 * coarse.kinetic_energy()


def test_case_json_roundtrip():
    c = cylinder_wake()
    d = c.to_dict()
    c2 = StructuredCase.from_dict(d)
    assert c2.to_dict() == d


def test_strouhal_helper():
    t = np.linspace(0, 100, 4000)
    assert abs(strouhal(t, np.sin(2 * np.pi * 0.165 * t)) - 0.165) < 0.005
