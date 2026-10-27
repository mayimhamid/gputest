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

Outside the benchmark suite, with their own scripts and committed results:

| Check | Script | Result |
|---|---|---|
| Readers and writers against files written by Fluent | `benchmarks/fluent_files_check.py` | [7 of 7 properties](../user-guide/fluent.md#checked-against-files-written-by-fluent) |
| FP32 against FP64 (accuracy and convergence) | `benchmarks/precision_fp32.py` | [FP32 reaches the tolerance in 1 of 6 cases](../user-guide/gpu.md#precision) |
| Memory estimate against measured peaks | `benchmarks/memory_estimate.py` | [upper bound in all 11 cases](../user-guide/gpu.md#memory) |
| SIMPLE/SIMPLEC stability against mesh skew | `benchmarks/skew_envelope.py` | [stable α_p ≈ cos²θ](../theory/simplec.md#skewed-meshes-measured-stability-envelope) |
| Planner placement and overhead | `benchmarks/planner_unstructured.py` | [planner study](planner.md) |

## Failing checks

Two of the 70 acceptance checks fail in the committed run (FlowTrace 0.2.0; the re-run on the
release code is in progress). Both are reported as measured, and their criteria were fixed
before the run.

* **Realizable k-ε, flat plate (`flat_plate_kepsilon`).** Over 10⁶ < Re_x < 10⁷, Cf is on
  average 7.0 % below Schultz-Grunow (criterion: within 6 %) and 11.4 % below White (within
  12 %, passes). The wall-function mesh puts the first cell at y⁺ ≈ 56. SST on a wall-resolved
  mesh is 5.3 % and 9.8 % below the same correlations. **Cause found since:** the wall
  function paired κ = 0.41 with Fluent's E = 9.793, a combination that matches neither Fluent
  nor the classical log law. With Fluent's κ = 0.4187 (now the code's value) the same case is
  4.96 % below Schultz-Grunow. The remaining deviation depends on y⁺ (−7.3 % at y⁺ ≈ 30, −2.1 %
  at y⁺ ≈ 105; [turbulence theory](../theory/turbulence.md)).
* **Warm start, cylinder at Re = 40 (`warmstart_proxy`).** The warm run needs 1,085 iterations
  to reach the converged field, the cold run 700 (×0.65; criterion > 1). This is a property of
  the test meshes, not a solver fault. The pre-solve mesh resolves the cylinder by 12
  staircase cells per diameter and the target mesh by 9. The converged recirculation lengths
  behind the body differ accordingly: 1.87 D on the source mesh (the FlowTrace pre-solve gives
  1.79 D there, within one cell) and 2.33 D on the target mesh
  (`benchmarks/warmstart_cylinder_wake.py`, [table](warmstart.md#the-cylinder-case)). The interpolated field is the
  right answer on the wrong mesh. The wake has to grow by a quarter, which SIMPLEC does more
  slowly than it develops the wake from rest. A warm start pays off when the pre-solve mesh
  resolves the geometry the way the target mesh does, and on this mesh pair it does not. A
  pre-registered test confirmed this: with matched meshes the warm start saves ×1.58 (8 cells/D)
  and ×2.04 (12 cells/D) ([details](warmstart.md#the-cylinder-case)).

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
