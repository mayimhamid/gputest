# Validation report

Every number on this page was produced by

```bash
flowtrace benchmark --tier full --out benchmarks/results
```

on the machine named under the table, in FP64 on the CPU. The suite writes `results.json`, a
self-contained [HTML report with plots](report.html), a [PDF](report.pdf) and the table below.
Each check states its acceptance criterion. Failing checks are reported as failing, and
criteria are not adjusted after the fact.

## Results

--8<-- "benchmarks/results/validation_table.md"

## What the checks cover

| Category | Benchmarks | Reference |
|---|---|---|
| Verification (exact answers) | `taylor_green_order`, `mms_mac`, `mms_simplec_quads`, `mms_simplec_tri`, `projection_divergence`, `simplec_mass_balance`, `periodic_poiseuille`, `taylor_couette`, `cht_composite_wall`, `cht_heated_channel` | exact / manufactured / analytic solutions |
| Grid convergence | `gci_natural_convection`, `gci_cavity` | Celik et al. (2008) procedure |
| Laminar validation | `cavity_re100_*`, `cavity_re1000_simplec`, `natural_convection_*`, `channel_re50_mac` | Ghia et al. (1982), de Vahl Davis (1983), Poiseuille |
| Turbulence | `channel_retau395_*`, `flat_plate_*`, `bfs_driver_seegmiller_sst` | MKM DNS (1999), White / Schultz-Grunow, Driver & Seegmiller (1985) |
| Fluent workflow | `warmstart_proxy` | FlowTrace SIMPLEC cold vs `.ip` warm start ([details](warmstart.md)) |

## Failing checks

Two of the 70 acceptance checks fail in the committed run. Both are reported as measured, and
their criteria were fixed before the run.

* **Realizable k-ε, flat plate (`flat_plate_kepsilon`).** Over 10⁶ < Re_x < 10⁷, Cf is on
  average 7.0 % below Schultz-Grunow (criterion: within 6 %) and 11.4 % below White (within
  12 %, passes). The wall-function mesh puts the first cell at y⁺ ≈ 56. SST on a wall-resolved
  mesh is 5.3 % and 9.8 % below the same correlations. The cause has not been isolated yet:
  candidates are the scalable wall function at this y⁺ and the model itself.
* **Warm start, cylinder at Re = 40 (`warmstart_proxy`).** The warm run needs 1,085 iterations
  to reach the converged field, the cold run 700 (×0.65; criterion > 1). This is a property of
  the test meshes, not a solver fault. The pre-solve mesh resolves the cylinder by 12
  staircase cells per diameter and the target mesh by 9. The converged recirculation lengths
  behind the body differ accordingly: 1.87 D on the source mesh (the FlowTrace pre-solve gives
  1.79 D there, within one cell) and 2.33 D on the target mesh
  (`benchmarks/warmstart_cylinder_wake.py`, [table](warmstart.md#the-cylinder-case)). The interpolated field is the
  right answer on the wrong mesh. The wake has to grow by a quarter, which SIMPLEC does more
  slowly than it develops the wake from rest. A warm start pays off when the pre-solve mesh
  resolves the geometry the way the target mesh does, and on this mesh pair it does not.

## Known deviations

* **Realizable k-ε, channel:** the core velocity is over-predicted (centreline U⁺ 7.2 % high)
  with wall functions, while the log layer is correct. Standard k-ε (3.5 %) and SST (4.1 %)
  are within 5 % of the DNS. Use SST or standard k-ε for internal flows dominated by the core.
* **SST, flat plate:** Cf is 9.8 % below White's correlation and 5.3 % below Schultz-Grunow.
  The two correlations differ from each other by about 5 % at these Reynolds numbers.
* **SST, backward-facing step:** the reattachment length is 6.61 H, 5.6 % longer than measured.
  That is typical of SST on grids of this size; the inflow boundary layer is developed from a
  uniform inlet over 110 step heights.
* **Pressure on triangles:** the observed order of the pressure error on warped triangles is
  1.2 for the finest grid pair (velocity and temperature: 2.0). The acceptance threshold for
  this quantity is 1.0.

## Reproduce a single check

```bash
flowtrace benchmark --list
flowtrace benchmark --only channel_retau395_sst --out my_results
```
