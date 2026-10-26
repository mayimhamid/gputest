# Ansys Fluent workflow

FlowTrace is meant to sit **in front of** Fluent. It computes an initial field that Fluent
interpolates onto its own mesh and continues from. Fluent remains the solver of record.

## 1. From an existing Fluent case

```bash
flowtrace fluent-warmstart my_case.cas.h5 --turbulence k-omega-sst \
    --bc inlet=inlet:12.5 --bc outlet=outlet --rho 1.225 --mu 1.79e-5 --out ws/
```

1. FlowTrace reads the mesh from the case (`.msh`, `.cas`, `.cas.h5` or `.msh.h5`) and prints a
   mesh quality report. It stops if the mesh is below Fluent's recommendations; override with
   `--allow-bad-mesh`.
2. It solves with its pressure-based solver. Zones you do not name in `--bc` get defaults from
   their Fluent zone types.
3. It writes `ws/flowtrace.ip`, plus `ws/flowtrace.dat.h5` when the input was HDF5.
4. `ws/warmstart.jou` and `ws/warmstart_pyfluent.py` open **your original case**, so Fluent keeps
   the physics you set up. They initialise, interpolate `flowtrace.ip`, iterate and save.

```bash
fluent 3ddp -g -t8 -i ws/warmstart.jou     # batch
python ws/warmstart_pyfluent.py            # PyFluent
```

In the GUI the equivalent is: read your case → *Initialize* → *File › Interpolate › Read and
Interpolate* `flowtrace.ip` → *Run Calculation*.

`--setup-physics` also writes the viscous and energy model settings into the scripts. Use it when
the case file has none.

## 2. From a FlowTrace case

`flowtrace run ... --export out/`, `flowtrace solve ... --export out/` or the dashboard's
*Fluent* tab write a hand-off package: `flowtrace.msh` (named zones), `flowtrace.ip`,
`run_fluent.jou`, `run_pyfluent.py` and a CSV of cell values.

## File formats

| File | Read | Write | Status |
|---|---|---|---|
| `.msh` / legacy `.cas` (ASCII, binary, gzip) | ✓ | ✓ (`.msh`) | round-trip tested; sample meshes in `examples/` |
| Periodic pairs (section 18) | ✓ | ✓ | round-trip tested |
| `.cas.h5` / `.msh.h5` (CFF mesh) | ✓ | ✓ | layout of the public VTK reader; **not yet checked against Fluent-written files** |
| `.dat.h5` (CFF cell data, `SV_*` names) | ✓ | ✓ | as above |
| `.ip` interpolation (v3 ASCII) | ✓ | ✓ | the verified warm-start route |
| `.prof` point profile | — | ✓ | boundary profiles |
| Journal (`.jou`), PyFluent (`.py`) | — | ✓ | TUI/Settings-API names can differ between releases |

`flowtrace h5-inspect` prints the full HDF5 tree. If FlowTrace cannot read a Fluent-written
`.cas.h5`, please open an issue and attach that output; no geometry is needed.

## Measuring the benefit

`flowtrace warmstart-study` runs five cases (cavity, channel, step, cylinder, heated cavity). It:

* **measures** the iteration savings with FlowTrace's own SIMPLEC solver as a proxy for Fluent's
  pressure-based solver (cold start vs. the `.ip` field interpolated onto a different mesh);
* runs cold and warm Fluent starts through PyFluent if a licensed Fluent is available. Otherwise
  it writes `cold.jou`/`warm.jou` for each case, and `flowtrace parse-transcript cold.trn warm.trn`
  counts the iterations afterwards.

Results without Fluent stay **labelled placeholders**. See the [study page](../validation/warmstart.md).

!!! tip "Comparing iteration counts fairly"
    Fluent normalises the continuity residual by its largest value in the first five
    iterations. A warm start begins with a small imbalance, so its *normalised* residual looks
    worse than it is. Compare absolute residuals, or a physical monitor such as drag, Nusselt
    number or mass imbalance. FlowTrace's study counts iterations until the field itself is
    within 10⁻⁴ of the converged solution.
