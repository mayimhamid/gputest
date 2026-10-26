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

## Residuals and convergence

Scaled residuals as in Fluent: momentum and energy use
\(\sum|r| / \sum|a_P\phi_P|\); continuity uses \(\sum|\dot m|\) divided by its largest value in
the first five iterations. For flows that start without any imbalance (periodic channels, warm
starts), that denominator is floored at \(10^{-3}\sum|F|\), so convergence is not judged against
round-off. `residuals()["raw"]` reports the absolute values.

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
