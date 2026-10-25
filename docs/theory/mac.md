# Structured MAC solver

`flowtrace.structured.solver.MACSolver` solves the 2-D incompressible Navier–Stokes equations
with an optional Boussinesq energy equation on a uniform staggered (MAC) grid:

\[
\frac{\partial \mathbf u}{\partial t} + \nabla\cdot(\mathbf u\mathbf u) = -\nabla p + \nu\nabla^2\mathbf u
 + \boldsymbol\beta\,(T - T_\mathrm{ref}) + \mathbf f,\qquad \nabla\cdot\mathbf u = 0,
\]
\[
\frac{\partial T}{\partial t} + \nabla\cdot(\mathbf u T) = \kappa\nabla^2 T + f_T .
\]

\(\boldsymbol\beta = \rho\beta\mathbf g\) is the buoyancy vector, and \(\mathbf f, f_T\) are
optional source fields (`set_sources`), used for manufactured solutions and spatially varying
body forces.

## Discretisation

* **Staggered arrangement.** \(u\) lives on x-faces, \(v\) on y-faces, \(p, T\) at cell centres.
  Discrete divergence and gradient are exact adjoints, so pressure cannot checkerboard and the
  projection is exact to round-off.
* **Convection** is in conservative flux form. Face values come from central, first-order
  upwind, QUICK or van Leer TVD interpolation. Boundary faces use the boundary value.
* **Time integration.** Adams–Bashforth 2 (variable step) for convection. Diffusion is either
  explicit (AB2) or Crank–Nicolson in delta form,
  \((\sigma - L_0)\,\delta u = \tfrac{2}{\nu}R\), solved exactly by the separable fast solver.
  It switches to Crank–Nicolson only when explicit diffusion would limit the time step.
* **Time-step control.** The CFL number on the local velocity, the AB2 diffusion limit and, for
  upwind-biased schemes, the combined AB2 stability bound
  \(\Delta t\,\big(2|u|/\Delta x + 2|v|/\Delta y + 4\nu(\Delta x^{-2}+\Delta y^{-2})\big) \le 0.9\).
  Without the combined bound, a channel at CFL 0.5 with explicit diffusion was measured to
  oscillate indefinitely near the outlet.
* **Projection.** Incremental pressure correction,
  \(-L\phi = -\nabla\!\cdot\mathbf u^*/\Delta t\), \(\mathbf u = \mathbf u^* - \Delta t\nabla\phi\),
  \(p \mathrel{+}= \phi\). It is solved by a separable direct solver (per axis: DCT/DST/FFT or a
  dense eigen-decomposition applied as GEMM, chosen by timing both), or by multigrid-preconditioned
  CG when immersed obstacles make the operator non-separable.

At a steady state the incremental projection reproduces the discrete steady equations exactly,
so steady results do not depend on \(\Delta t\). That is why the manufactured-solution test can
measure the spatial order.

## Boundary conditions

| Kind | Normal velocity | Tangential velocity | Pressure | Temperature |
|---|---|---|---|---|
| wall | 0 | ghost reflection to the wall speed | Neumann | fixed, flux or adiabatic |
| inlet | profile (uniform/parabolic) | 0 | Neumann | fixed |
| outlet | convective \(\partial_t u + U_c \partial_n u = 0\) | zero gradient | Dirichlet 0 | zero gradient / inflow value |
| symmetry | 0 | zero gradient | Neumann | zero gradient |
| periodic | periodic | periodic | periodic | periodic |

Immersed obstacles are blocked cells (staircase). Velocities on blocked faces are zero, ghost
values are reflected, and isothermal obstacles impose their temperature through ghost cells.

## Steady problems: FMG

Like Fluent's FMG initialisation, a steady case is first converged on 1/4 and 1/2 of the
resolution. Each solution is interpolated (and re-projected to be divergence-free) onto the next
grid.

## Verification

* Taylor–Green vortex: second-order L∞ error against the exact solution
  (`taylor_green_order`).
* Manufactured solution with walls and energy: observed order 2.0 for u, v, p and T
  (`mms_mac`; see [verification](verification.md)).
* Divergence after projection: ~1e-14 (fast solver), ~1e-11 (multigrid).
