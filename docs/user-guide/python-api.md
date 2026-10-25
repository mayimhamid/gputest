# Python API and case files

Everything the CLI does is available from Python. Cases are plain dataclasses that round-trip
through JSON, so the dashboard, the CLI and scripts all share them.

## Structured solver

```python
from flowtrace.structured.case import lid_driven_cavity
from flowtrace.structured.solver import MACSolver
from flowtrace.hardware.backend import get_backend

s = MACSolver(lid_driven_cavity(re=400, n=128), backend=get_backend("auto"))
while not s.finished:
    s.step()
    if s.step_count % 10 == 0:
        s.residuals()            # also drives the steady-state test
(y, u), (x, v) = s.centerline()
fields = s.fields()              # cell-centred NumPy arrays (x-velocity, pressure, vorticity, ...)
```

## Unstructured solver

```python
from flowtrace.mesh.fluent_h5 import read_any_mesh
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.conjugate import make_solver

mesh, fmt = read_any_mesh("duct.cas.h5")
case = UnstructuredCase(
    boundaries={"inlet": BoundarySpec("inlet", normal_speed=10.0, turb_intensity=0.05),
                "outlet": BoundarySpec("outlet"),
                "heater": BoundarySpec("wall", thermal="flux", value=5e3)},
    density=1.2, viscosity=1.8e-5, energy=True, conductivity=0.026, cp=1006.0,
    turbulence="k-omega-sst", max_iter=5000, tol=1e-5)
solver = make_solver(mesh, case)       # ConjugateSolver if the case has solid zones
while not solver.finished:
    solver.step()
print(solver.mass_balance(), solver.heat_flow())
```

### Boundary kinds

| `BoundarySpec.kind` | Fluent zone types mapped to it | Data |
|---|---|---|
| `wall` | wall | `velocity` or `rotation={"omega", "origin", "axis"}`; `thermal` = adiabatic / temperature / flux with `value` |
| `inlet` | velocity-inlet, mass-flow-inlet | `velocity` vector or `normal_speed`; `value` = temperature; turbulence `turb_intensity`, `turb_viscosity_ratio` or `k`, `omega`, `epsilon` |
| `outlet` | pressure-outlet, outflow, … | static `pressure`; backflow takes `value` and the turbulence values |
| `pressure` | pressure-inlet, inlet-vent | static `pressure`, inflow allowed |
| `symmetry` | symmetry, axis | — |

### Zones

```python
from flowtrace.mesh.unstructured import make_periodic, set_cell_zone, split_zone

make_periodic(mesh, "inlet", "outlet")                       # translational periodicity
mesh = set_cell_zone(mesh, lambda xc: xc[:, 1] < 0, "plate", "solid")
case.solids = {"plate": {"conductivity": 16.0, "source": 1e5}}  # conjugate heat transfer
case.mrf = {"rotor": {"omega": 157.0, "origin": [0, 0, 0], "axis": [0, 0, 1]}}
```

## Mesh generation helpers

`cartesian_mesh_2d` (quads/triangles, graded `xs`/`ys`, smooth `warp`), `annulus_mesh`,
`extrude_to_3d`, `flowtrace.fluent.convert.masked_block_mesh` (graded blocks minus solid cells)
and `flowtrace.mesh.grading` (geometric node distributions for a target first-cell height).

## Verification tools

```python
from flowtrace.validation.mms import run_simplec
from flowtrace.validation.gci import gci

r = run_simplec((16, 32, 64), triangulate=True, warp=0.05)
print(r.orders)                  # observed order per quantity
g = gci(phi_fine, phi_medium, phi_coarse, h_fine, h_medium, h_coarse)
print(g.observed_order, g.extrapolated, g.gci_fine)
```
