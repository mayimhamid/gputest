# Fluent warm-start study

**Question:** how many iterations does a FlowTrace initial field save a pressure-based solver?

## Method

For five cases (lid-driven cavity Re = 400, developing channel Re = 100, backward-facing step
Re = 200, steady cylinder Re = 40, heated cavity Ra = 10⁵):

1. FlowTrace's structured solver computes the flow (with FMG). The solution is written as a
   Fluent interpolation file (`.ip`), and the mesh as `.msh` with named zones.
2. **Proxy measurement (always runs).** The `.ip` file is interpolated onto a *different*
   unstructured mesh: triangles, or a 0.75× coarser quad mesh. FlowTrace's SIMPLEC solver,
   from the same algorithm family as Fluent's default pressure-based solver, then runs twice.
   The cold run starts from rest and stops at scaled residuals of 10⁻⁵. The warm run starts
   from the interpolated field and stops when it reaches the cold run's *absolute* final
   residuals. It is also checked that both runs converge to the same solution.
3. **Fluent measurement (needs a licence).** The same cold/warm pair in Ansys Fluent through
   PyFluent, or by running `cold.jou`/`warm.jou` and `flowtrace parse-transcript cold.trn warm.trn`.

## Results

--8<-- "benchmarks/fluent_warmstart/results.md"

## Reading the numbers

* The proxy says how much a Fluent-like segregated solver gains from the field. It is **not** a
  Fluent measurement. Fluent's coupled solver, its AMG settings and its pseudo-transient options
  will change the ratio.
* The FlowTrace pre-solve time is listed next to the savings, so the net benefit can be judged.
  The pre-solve is worthwhile when the expensive solver's iterations cost more than the
  FlowTrace solve, which is the usual case for large Fluent meshes and licensed core-hours.
* The normalised-residual count (in brackets) shows the pitfall described in
  [Fluent workflow](../user-guide/fluent.md#measuring-the-benefit). Judged by Fluent's default
  continuity normalisation, a warm start can look slower even when it is not.

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
