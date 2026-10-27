# Fluent warm-start study

**Question:** how many iterations does a FlowTrace initial field save a pressure-based solver,
and does that save time once the pre-solve is paid for?

## Method

For five cases (lid-driven cavity Re = 400, developing channel Re = 100, backward-facing step
Re = 200, steady cylinder Re = 40, heated cavity Ra = 10⁵):

1. FlowTrace's structured solver computes the flow (with FMG). The solution is written as a
   Fluent interpolation file (`.ip`), and the mesh as `.msh` with named zones.
2. **Proxy measurement (always runs).** The `.ip` file is interpolated onto a *different*
   unstructured mesh: triangles, or a 0.75× coarser quad mesh. FlowTrace's SIMPLEC solver,
   from the same algorithm family as Fluent's default pressure-based solver, then runs twice.
   The cold run starts from rest and is converged tightly (scaled residuals of 10⁻⁷); its
   final field is the reference answer. The warm run starts from the interpolated field. **Both
   runs are counted until the velocity (and temperature) field is within 10⁻⁴ (relative, max
   norm) of that answer.** This measures what matters, the solution, and is immune to
   residual-normalisation artefacts. The usual "iterations until the scaled residuals reach
   10⁻⁵" are listed as well. **Both runs are also timed on equal terms:** solver steps only,
   until the same field tolerance. The warm path is charged with the FlowTrace pre-solve and
   the interpolation. The ratio of the two is the wall-time saving, reported for information,
   with no acceptance criterion.
3. **Fluent measurement (needs a licence).** The same cold/warm pair in Ansys Fluent through
   PyFluent, or by running `cold.jou`/`warm.jou` and `flowtrace parse-transcript cold.trn warm.trn`.

## Results

--8<-- "benchmarks/fluent_warmstart/results.md"

## Reading the numbers

* **The cylinder case is slower with a warm start (×0.65), and the benchmark reports it as a
  failure.** See [below](#the-cylinder-case).
* The proxy says how much a Fluent-like segregated solver gains from the field. It is **not** a
  Fluent measurement. Fluent's coupled solver, its AMG settings and its pseudo-transient options
  will change the ratio.
* The FlowTrace pre-solve time is listed next to the savings, so the net benefit can be judged.
  The pre-solve is worthwhile when the expensive solver's iterations cost more than the
  FlowTrace solve, which is the usual case for large Fluent meshes and licensed core-hours.
* The "warm to tol." column shows the pitfall described in
  [Fluent workflow](../user-guide/fluent.md#measuring-the-benefit). Judged by Fluent-style
  continuity normalisation, a warm start can look slower than it is: its first iterations
  already have a small imbalance, so it is normalised by a small number. A dash means the warm
  run reached the converged field before the normalised residuals reached the tolerance.

### The cylinder case

The pre-solve resolves the cylinder with 12 staircase cells per diameter, the target mesh with
9. Converged on each grid, the flow has different recirculation lengths behind the body:

--8<-- "benchmarks/results/warmstart_cylinder_wake.md"

The interpolated field is the right answer on the wrong mesh. On the target mesh the wake has
to grow by about a quarter, and SIMPLEC does that more slowly than it develops the wake from
rest.

**A pre-registered test of that explanation.** If the mesh mismatch is the cause, a target
mesh that resolves the cylinder like the pre-solve should restore the saving. The prediction
was written down before the runs: fewer warm than cold iterations on a matched target, and no
saving, or a much smaller one, on a mismatched target. Same case, same field tolerance, and
FlowTrace's SIMPLEC as the target solver:

| Pre-solve | Target mesh | Target cells | Cold | Warm | Saved |
|---|---|---:|---:|---:|---:|
| 8 cells/D | same resolution (scale 1.0) | 8,144 | 300 | 190 | ×1.58 |
| 8 cells/D | coarser (scale 0.75, 6 cells/D) | 4,580 | 306 | 260 | ×1.18 |
| 12 cells/D | same resolution (scale 1.0) | 18,324 | 558 | 274 | ×2.04 |
| 12 cells/D | coarser (scale 0.75, 9 cells/D): the study's case | 10,308 | 700 | 1,085 | ×0.65 |

The prediction held at both resolutions. On a target mesh that resolves the cylinder like the
pre-solve, the warm start keeps its benefit: ×1.58 at 8 cells/D and ×2.04 at 12 cells/D. A
coarser target lowers the saving (×1.18 at 8 → 6 cells/D) or turns it into a loss (×0.65 at
12 → 9 cells/D). The failing check is therefore a property of this mesh pair, not of the solver
or of the interpolation. It stays in the benchmark as a failing check, and its criterion is not
changed. The 12 → 9 cells/D row is the committed full-tier run; the other rows come from
`flowtrace.fluent.warmstart_study.run_proxy_case` with the target scale changed.

These runs shared the machine with other jobs, so only their iteration counts are reported;
their timings are not.

## Running the Fluent part

```bash
pip install ansys-fluent-core
flowtrace warmstart-study --out benchmarks/fluent_warmstart        # proxy + Fluent (licence needed)
```

Without PyFluent, run Fluent in batch in each case folder and count the iterations:

```bash
fluent 2ddp -g -t4 -i cold.jou
fluent 2ddp -g -t4 -i warm.jou
flowtrace parse-transcript cold.trn warm.trn
```

The journals contain the model and solver settings but not the boundary-condition values,
because those TUI commands differ between Fluent releases. `case.json` lists the values, and
`run_fluent_study` (PyFluent) sets them automatically.
