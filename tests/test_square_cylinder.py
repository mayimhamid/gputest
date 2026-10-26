"""Regression on a real GAMBIT/Fluent mesh: square cylinder in a channel, Re = 40 (steady).

tests/data/SquareCylinder-Coarse.msh is the coarsest of the project author's four-mesh
independence set (5 824 quads, D = 4, inlet 4.5 D upstream, side walls 6 D from the axis).
The reference values were computed with this code base and agree with the finer meshes of the
set within the grid-convergence band (Finest: Cd 2.161 / 1.943, Lr/D 2.324 / 2.831; see
docs/validation/square-cylinder.md). The test guards against regressions, it is not a validation.
"""

import os

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator

from flowtrace.mesh.fluent_h5 import read_any_mesh
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
from flowtrace.unstructured.simple import SimpleSolver

MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SquareCylinder-Coarse.msh")
D, X_REAR = 4.0, -18.0


def test_zones_are_detected_from_the_fluent_file():
    mesh, fmt = read_any_mesh(MESH)
    assert fmt == "fluent-msh" and mesh.dim == 2 and mesh.n_cells == 5824
    b = default_boundaries(mesh)
    assert b["inlet"].kind == "inlet" and b["outlet"].kind == "outlet"
    assert b["BluffWalls"].kind == "wall" and b["SideWalls"].kind == "wall"


@pytest.mark.parametrize("sides,cd,lr", [("wall", 2.12276, 2.3485), ("symmetry", 1.90654, 2.8977)])
def test_square_cylinder_re40(sides, cd, lr):
    mesh, _ = read_any_mesh(MESH)
    bnd = default_boundaries(mesh)
    if sides == "symmetry":
        bnd["SideWalls"] = BoundarySpec("symmetry")
    s = SimpleSolver(mesh, UnstructuredCase(boundaries=bnd, viscosity=0.1, max_iter=3000, tol=1e-7, backend="numpy"))
    while not s.finished:
        s.step()
    assert s.converged
    f = s.wall_forces()["BluffWalls"]["total"]
    q = 0.5 * 1.0 * 1.0 ** 2 * D
    assert abs(f[0] / q - cd) < 2e-4
    assert abs(f[1] / q) < 1e-6  # symmetric flow: no lift
    mb = s.mass_balance()
    assert abs(sum(mb.values())) < 1e-8 * abs(mb["inlet"])
    ip = LinearNDInterpolator(s.cell_centres(), s.fields()["x-velocity"])
    xs = np.linspace(X_REAR + 1e-3, 30.0, 6000)
    u = ip(xs, np.zeros_like(xs))
    k = np.nonzero((u[:-1] < 0) & (u[1:] >= 0))[0][0]
    x0 = xs[k] - u[k] * (xs[k + 1] - xs[k]) / (u[k + 1] - u[k])
    assert abs((x0 - X_REAR) / D - lr) < 5e-3
