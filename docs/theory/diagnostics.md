# Solver diagnostics

The Fluent-style scaled residuals answer one question: is the linearised momentum system
satisfied, on average, by the field the iteration started from? They do not say how well the
linear solves did their job, whether the mass fluxes the iteration hands on balance in every
cell, or whether anything became non-finite. `SimpleSolver.diagnostics()` reports these
separately, each with the exact system and state it is evaluated on. The normative definitions
are in the docstring of `flowtrace/unstructured/diagnostics.py`; this page explains them.

```python
s = SimpleSolver(mesh, case)
for _ in range(100):
    s.step()
d = s.diagnostics()                 # full report of the latest outer iteration
from flowtrace.unstructured.diagnostics import summary
summary(d)                          # headline numbers (also in the dashboard events)
```

The solver keeps references to the systems of its latest iteration (no copies, and released at
the start of the next one, so the peak memory of an iteration does not grow). The metrics are
computed only when asked for, with all reductions on the compute device and one transfer to the
host. One evaluation costs 0.20–0.28 of an outer iteration (measured on a 4-core CPU at 4k and
65k cells); the dashboard and `flowtrace solve` evaluate them every ≥ 10 iterations and at the
end of the run, so they cost at most about 3 % of the run time.

## Where in the iteration each quantity is evaluated

| Step of iteration *n* | System or state | Reported as |
|---|---|---|
| 1. Momentum assembled at \((U^n, p^n, F^n)\) | \(A u = b\); \(b\) holds the pressure force, body forces, sources, boundary and deferred-correction terms | `momentum.<component>.pre_solve`: \(b - A u^n\) |
| 2. Under-relaxed system given to Jacobi-BiCGSTAB | \(A_r = A\) with diagonal \(a_P/\alpha_u\); \(b_r = b + \frac{1-\alpha_u}{\alpha_u} a_P u^n\) | — (note \(b_r - A_r u^n = b - A u^n\)) |
| 3. Linear solve returns \(u^*\) | same system | `momentum.<component>.solve`: \(b_r - A_r u^*\), plus the solver's iterations and recurrence residual |
| 4. Rhie–Chow predictor fluxes \(F^*\) | cell imbalance \(m^*\) | `continuity.predictor` |
| 5. Pressure correction \(A_p p' = -m^*\) (AMG-PCG) | | `pressure_correction` |
| 6. Corrected fluxes \(F^{n+1}\) | cell imbalance \(m^* + A_p p'\) = minus the pressure-correction residual | `continuity.corrected` |
| 7. \(U^{n+1}\), \(p^{n+1}\) | | `field_change` |
| 8. Energy (when this solver owns it) | as momentum | `energy.pre_solve`, `energy.solve` |

Two identities follow from the algorithm and are checked by `tests/test_diagnostics.py`: the
relaxed and the assembled system have the same residual at \(u^n\), and the corrected cell
imbalance is exactly minus the residual of the pressure-correction equation. The corrected
imbalance is therefore reduced by the pressure tolerance (`pressure_tol`, 0.1 by default) per
iteration, not to zero; it vanishes as the outer iteration converges.

## Row-scaled (componentwise) residual

For a system \(Ax=b\) at \(x\):

\[
r_i = b_i - \sum_j A_{ij}x_j, \qquad s_i = |b_i| + \sum_j |A_{ij}|\,|x_j|, \qquad
R_\text{row} = \max_{i\ \text{active}} \frac{|r_i|}{s_i}.
\]

\(R_\text{row}\) is the Oettli–Prager componentwise backward error: the smallest relative
perturbation of the individual entries of \(A\) and \(b\) for which \(x\) is the exact solution
(Higham, *Accuracy and Stability of Numerical Algorithms*, 2002, Thm. 7.3). It lies between 0 and
1, has no units, and needs no reference value chosen by hand. It measures how well the
**discrete system** is satisfied. It is not the error of the solution against the continuous
problem, and it is not independent of the mesh.

`|A|` uses the assembled matrix entries, so two faces between the same pair of cells (possible
on polyhedral meshes) are summed before the absolute value is taken. Every row is a
finite-volume balance: boundary conditions enter through the diagonal and the right-hand side,
so there are no identity rows to skip.

**Rows with a negligible scale.** A row is excluded when
\(s_i \le \gamma_K\, S_\text{ref}\), with \(S_\text{ref}\) the largest \(s_i\) of the equation
set (all velocity components share it), \(\gamma_K = Ku/(1-Ku)\) the bound on the rounding error
of a \(K\)-term sum, \(u\) the unit round-off of the working precision (\(2^{-53}\) in FP64) and
\(K\) the largest number of terms in a row. The whole magnitude of such a row is below the rounding
error of evaluating the largest rows, so its ratio is not resolvable. This keeps, for example,
the round-off-level \(w\) equation of a 2-D flow on a 3-D mesh out of the maximum. No dimensional
constant is involved. Excluded rows are counted (`excluded_rows`) and their largest absolute
residual is reported (`excluded_abs_max`).

**Zero and non-finite values.** \(0/0\) is reported as 0 and \(x/0\) for \(x>0\) as infinity. A
system whose every term is exactly zero (no driving at all) therefore reports 0 everywhere.
Rows with a non-finite residual or scale are counted (`nonfinite_rows`) and set the metric to
infinity, which JSON outputs show as `null`.

## Local and global continuity

| Quantity | Definition | Units |
|---|---|---|
| cell imbalance \(m_i\) | \(\sum_f F_f\), outward positive | kg/s (per unit depth in 2-D) |
| `local_max_relative` | \(\max_i |m_i| / \sum_f |F_f|\) over the cell's faces: the componentwise backward error of the cell's mass balance; same exclusion rule, \(K\) = most faces of a cell | – |
| `l1_over_face_flux` | \(\sum_i |m_i| / \sum_\text{faces} |F|\) | – |
| `global_boundary.signed` | \(G = \sum_\text{boundary faces} F_b\), positive = net outflow | kg/s |
| `global_boundary.relative` | \(|G| / \sum_b |F_b|\); 0 for a closed domain with \(G=0\) | – |

Translational periodic faces are interior faces, so they enter the local balances and cancel from
the global sum. Since \(G = \sum_i m_i\) exactly, a zero global sum is necessary but **not
sufficient** for local conservation.

The older `continuity_local_max` value of the run result is kept unchanged: it is
\(\max_i |m_i|\) divided by (\(\sum\) of all \(|F|\) and \(|F_b|\)) / number of cells, on the final
fluxes.

## Field change per iteration and health

`field_change.velocity` is \(\max_i |U^{n+1}_i - U^n_i| / U_\text{ref}\) with \(U_\text{ref}\)
the largest velocity magnitude in \(U^n\), \(U^{n+1}\) or on velocity boundaries;
`pressure` is \(\max_i |p^{n+1}_i - p^n_i| / \max(\text{range}(p^{n+1}), \rho U_\text{ref}^2)\);
`temperature` is relative to the range of \(T^{n+1}\) together with the fixed boundary
temperatures; turbulence quantities are relative to their own maximum. Under-relaxation keeps
these small even far from convergence: they are not error estimates. They are `null` for an
iteration the stability guard rolled back (`rolled_back: true`).

`health.nonfinite_values` counts non-finite entries of velocity, pressure, face and boundary
mass fluxes, temperature and turbulence fields after the iteration.

## Where to find them

* Python: `SimpleSolver.diagnostics()` (also on `ConjugateSolver`, for the fluid sub-mesh).
* Dashboard and `flowtrace solve`: the headline numbers ride on the residual events every
  ≥ 10 iterations, the run log ends with a `[diagnostics]` line, and the run result and the HTML
  run report contain the full table.
* `run_batch`: each pressure-based case result carries `diagnostics` (the headline numbers).
