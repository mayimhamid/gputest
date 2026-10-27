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
| `.msh` / legacy `.cas` (ASCII, binary, gzip) | ✓ | ✓ (`.msh`) | round-trip tested; 3-D face orientation tested on a hexahedron written by hand in Fluent's convention; sample meshes in `examples/` |
| Periodic pairs (section 18) | ✓ | ✓ | round-trip tested |
| `.cas.h5` / `.msh.h5` (CFF mesh) | ✓ | ✓ | **read checked against Fluent-written files** (below); written files reproduce Fluent's node order and c0/c1 face by face; Fluent reading a FlowTrace-written file is not yet checked |
| `.dat.h5` (CFF cell data, `SV_*` names) | ✓ | ✓ | read checked against a Fluent-written solution (below); written files not yet read by Fluent |
| `.ip` interpolation (v3 ASCII) | ✓ | ✓ | the verified warm-start route |
| `.prof` point profile | — | ✓ | boundary profiles |
| Journal (`.jou`), PyFluent (`.py`) | — | ✓ | TUI/Settings-API names can differ between releases |

`flowtrace h5-inspect` prints the full HDF5 tree. If FlowTrace cannot read a Fluent-written
`.cas.h5`, please open an issue and attach that output; no geometry is needed.

### Checked against files written by Fluent

`benchmarks/fluent_files_check.py` downloads the PyFluent *mixing elbow* example, published
by Ansys in `ansys/example-data` under the MIT licence (about 9 MB, SHA-256 verified). That is
a 3-D polyhedral mesh as `.msh.h5` in inches, its case as `.cas.h5` in metres, and a converged
solution as `.dat.h5`. The script then tests seven properties, each with its criterion fixed
in `flowtrace/validation/fluent_files.py`. The same check runs as the opt-in test
`tests/test_fluent_files.py`, and in CI.

--8<-- "benchmarks/results/fluent_files_check.md"

The check found two bugs, both fixed in this release:

* **3-D face orientation.** Fluent stores a 3-D face with the right-hand normal of its node
  order pointing *toward* c0. FlowTrace reversed every 3-D face on reading and on writing. Its
  geometry repair hid the reversal in its own solutions, but files written for Fluent had every
  face reversed. 2-D files were not affected.
* **Skewness on polyhedra.** Equiangle skewness is defined for standard element shapes only. On
  the mixing elbow's polyhedra it gave false critical values, and `flowtrace quality` refused
  Fluent's own mesh. Polyhedral cells are now judged by orthogonal quality, as in Fluent.

3-D `.msh`/`.cas.h5` files written by FlowTrace 0.3.0 or earlier still read. The repair flips
their faces, and the quality report notes it. Re-export them before giving them to Fluent.

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
