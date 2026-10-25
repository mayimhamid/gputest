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

## Known deviations

* **Realizable k-ε, channel:** the core velocity is over-predicted (centreline U⁺ about 7 % high)
  with wall functions, while the log layer is correct. Standard k-ε and SST are within 5 %. Use
  SST or standard k-ε for internal flows dominated by the core.
* **SST, flat plate:** Cf is about 10 % below White's correlation and 3–4 % below Schultz-Grunow.
  The two correlations differ from each other by about 6 % at these Reynolds numbers.
* **SST, backward-facing step:** the reattachment length is longer than measured. That is typical
  of SST on grids of this size; the inflow boundary layer is developed from a uniform inlet over
  110 step heights.

## Reproduce a single check

```bash
flowtrace benchmark --list
flowtrace benchmark --only channel_retau395_sst --out my_results
```
