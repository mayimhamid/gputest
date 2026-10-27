# Pressure-based unstructured solver (SIMPLE / SIMPLEC)

`flowtrace.unstructured.simple.SimpleSolver` follows the algorithm behind Fluent's default
pressure-based segregated solver. It uses collocated, cell-centred finite volumes on arbitrary
polygons and polyhedra.

## Mesh data

Faces store owner \(P\) and neighbour \(N\) with the area vector \(\mathbf S_f\) pointing from
\(P\) to \(N\) (outwards on boundaries), as in Fluent. Left-handed faces are re-oriented on
import. For each interior face:

* \(\mathbf d = \mathbf x_N - \mathbf x_P\). For periodic pairs, \(\mathbf x_N\) is seen through
  the translation.
* Over-relaxed decomposition \(\mathbf S = \alpha\mathbf d + \mathbf T\) with
  \(\alpha = \mathbf S\cdot\mathbf S/\mathbf S\cdot\mathbf d\).
* Linear interpolation weight from the projection of the face centre on \(\mathbf d\).

## Discretisation

* **Gradients:** least squares (inverse-distance weighted) with boundary face values.
  Zero-gradient boundary values are reconstructed along the face with the lagged cell gradient,
  \(\phi_b = \phi_P + \nabla\phi_P\cdot\mathbf d_{b,t}\), which keeps skewed boundary cells
  consistent.
* **Diffusion:** implicit \(\Gamma_f\alpha(\phi_N-\phi_P)\) plus the explicit non-orthogonal
  correction \(\Gamma_f\,\overline{\nabla\phi}_f\cdot\mathbf T\).
* **Convection:** implicit first-order upwind with a deferred, bounded second-order upwind
  correction.
* **Pressure force in momentum:** Green–Gauss on face pressures,
  \((\nabla p)_P = V_P^{-1}\sum_f p_f\mathbf S_f\), with linearly interpolated, skewness-corrected
  \(p_f\). This is the discrete adjoint of the face-flux divergence (it is also Fluent's choice).
  With a least-squares cell gradient instead, the manufactured-solution error stalled at O(1) on
  triangles. That option remains available as `pressure_gradient="least-squares"` for
  comparison.

## Pressure–velocity coupling

Momentum is solved with under-relaxation \(\alpha_u\) (BiCGSTAB, Jacobi preconditioned) to give
\(\mathbf u^*\). Face mass fluxes use Rhie–Chow interpolation:

\[
F_f = \rho\Big[\overline{\mathbf u^*}_f\cdot\mathbf S_f
  - \overline{d}_f\,\alpha_f\big(p_N - p_P - \overline{\nabla p}_f\cdot\mathbf d\big)\Big],
\qquad d = \frac{V}{a_P - \sum a_{nb}}\ (\text{SIMPLEC}),\ \frac{V}{a_P}\ (\text{SIMPLE}).
\]

Both pressure terms are projected on the same orthogonal direction \(\alpha\mathbf d\). The
pressure correction \(\sum_f \rho\,\overline d_f\alpha_f(p'_P-p'_N) = -\dot m_P\) is solved by
AMG-preconditioned CG. pyamg is used on the CPU, and FlowTrace's smoothed-aggregation AMG,
applied with cuSPARSE, on the GPU. The inner tolerance defaults to 0.1, Fluent's AMG default;
this was measured to leave outer iterations unchanged and save 15–50 % of the wall time.

**Pressure under-relaxation.** SIMPLE uses 0.3. SIMPLEC needs none on orthogonal meshes (1.0).
The pressure-correction equation keeps only the orthogonal part \(\alpha\) of each face, though,
and on non-orthogonal meshes the neglected part accumulates. On smoothly warped 96² and 128²
meshes (24.6° non-orthogonality) SIMPLEC at 1.0 was measured to diverge, while 0.8 converged.
The default is therefore automatic: 1.0 when the maximum non-orthogonality is ≤ 5°, otherwise
0.8. This costs 2–6 % more iterations where 1.0 would also have worked, and it follows Fluent's
own advice to use skewness correction or a more conservative factor on skewed meshes. An
explicit `relax_p` always takes precedence.

## Skewed meshes: measured stability envelope

The pressure-correction equation keeps only the orthogonal part of its Laplacian
(\(\alpha = \mathbf S\cdot\mathbf S/\mathbf S\cdot\mathbf d\)); the non-orthogonal remainder is
neglected, as in Fluent's default and in OpenFOAM without non-orthogonal correctors. A
controlled sequence shows what that costs: lid-driven cavity, Re = 10, 24 × 24 quads sheared
uniformly to the angle θ (so every face has non-orthogonality θ), 4,000 iterations at most.

| θ | min. orthogonal quality | α_p = 1 | 0.8 | 0.5 | 0.33 | 0.2 | automatic + guard | SIMPLE (α_p 0.3) |
|---|---|---|---|---|---|---|---|---|
| 0°–20° | ≥ 0.94 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 30° | 0.87 | ✗ | ✗ (late) | ✓ | ✓ | ✓ | ✓ (1 rollback, 0.48) | ✗ |
| 40° | 0.77 | ✗ | ✗ | ✓ (slow) | ✓ | ✓ | ✓ (1, 0.48) | ✗ |
| 45° | 0.71 | ✗ | ✗ | ✗ (late) | ✓ | ✓ | ✓ (2, 0.29) | ✗ |
| 50° | 0.64 | ✗ | ✗ | ✗ | ✓ (slow) | ✓ | ✓ (2, 0.29) | ✗ |
| 55° | 0.57 | ✗ | ✗ | ✗ | stalls | ✓ | ✓ (3, 0.17) | ✗ |
| 60° | 0.50 | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ (3, 0.17) | ✗ |

The largest stable α_p follows \(\cos^2\theta\) at every angle (0.75, 0.59, 0.50, 0.41, 0.33,
0.25): the neglected part grows like \(\tan^2\theta\) relative to the retained one. Plain SIMPLE
diverges as well, so the cause is the neglected non-orthogonal coupling, not SIMPLEC's
consistent coefficients. Meshes whose skew alternates in sign from face to face (warped quads,
triangles) tolerate far more: warped triangles at 43° converge at 0.8.

The stability guard (automatic `relax_p` only) recovered every case in the table. It stops at
`relax_p` = 0.1, which by the cos² law covers uniform skew up to about 70°. Beyond that, or with
an explicit `relax_p`, a diverging run now ends with an error naming the mesh non-orthogonality,
the guard's state and the cos² estimate. Non-orthogonal correctors for the pressure equation (the
principled fix) are not implemented.

## Residuals and convergence

Scaled residuals as in Fluent: momentum and energy use
\(\sum|r| / \sum|a_P\phi_P|\); continuity uses \(\sum|\dot m|\) divided by its largest value in
the first five iterations. For flows that start without any imbalance (periodic channels, warm
starts), that denominator is floored at \(10^{-3}\sum|F|\), so convergence is not judged against
round-off. `residuals()["raw"]` reports the absolute values.

Where they are evaluated: the momentum and energy residuals belong to the assembled,
*unrelaxed* system at the field the iteration started from (before its linear solve); the
continuity residual is the imbalance of the Rhie–Chow *predictor* fluxes, before the pressure
correction. When every velocity component is exactly zero in every cell (a cavity started from
rest, first iteration) \(\sum|a_P\phi_P| = 0\) and the momentum value is the numerator divided by
a \(10^{-300}\) floor, a very large number (\(\sim 10^{302}\)); convergence is only tested from the
sixth iteration on. The [solver diagnostics](diagnostics.md) separate the pre-solve, solved and
corrected states and are defined in that case.

## Energy

\(\nabla\cdot(F T) = \nabla\cdot\big((k/c_p + \mu_t/Pr_t)\nabla T\big) + q'''/c_p\), with fixed
temperature, heat flux or adiabatic walls, inlet temperatures and outlet backflow values.
`heat_flow()` reports conduction plus enthalpy flux per boundary zone; at steady state the values
sum to zero, which is checked by the benchmarks.

## Verification

Manufactured solution on smoothly warped quads and triangles (`mms_simplec_quads`,
`mms_simplec_tri`): observed order about 2 for velocity and temperature; pressure 2.0 on quads and
between first and second order on triangles: 1.2 for the finest grid pair of the full tier,
1.4 on the coarser grids of the quick tier.
Global mass and energy balances hold to ~1e-12.
