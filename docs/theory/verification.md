# Verification methodology

FlowTrace separates **verification** ("are the equations solved correctly?") from
**validation** ("are they the right equations for this flow?").

## Method of Manufactured Solutions

A smooth field satisfying every boundary condition of the unit square is prescribed. Its
Navier–Stokes and energy residual is added as a source, and the discrete solution is compared
with the field on a sequence of grids:

\[
\psi = \sin^2(\pi x)\sin^2(\pi y),\quad u = \partial_y\psi,\ v = -\partial_x\psi,\quad
p = \cos\pi x\cos\pi y,\quad T = \sin\pi x\sin\pi y .
\]

The velocity is divergence-free and vanishes on all walls, the pressure has zero mean and zero
normal derivative there, and T vanishes on the walls. The sources
\(\mathbf f = (\mathbf u\cdot\nabla)\mathbf u + \nabla p - \nu\nabla^2\mathbf u\) and
\(f_T = \mathbf u\cdot\nabla T - \kappa\nabla^2T\) are evaluated analytically
(`flowtrace.validation.mms`).

The observed order between successive grids is \(p = \ln(e_{2h}/e_h)/\ln 2\), using L2 errors.

**Mesh sequences matter.** The SIMPLEC tests use a *smooth* distortion
(\(\mathbf x \mathrel{+}= 0.05\sin 2\pi x\sin 2\pi y\)) of quads and triangles. Refining such a
mesh is a systematic refinement. Randomly jittered meshes that are re-randomised at every level
are not, and cell-centred schemes are known to lose formal order on them, so they are not used
to claim an order.

### What MMS found

On triangles the SIMPLEC velocity error stalled at about 0.04 under refinement. Stepping through
the operators showed that diffusion and momentum alone were second order. The fault was the
pairing of a least-squares cell pressure gradient (in momentum) with the face-flux divergence
(in continuity): the two are not adjoint. Switching the momentum pressure force to the
Green–Gauss face-pressure form restored order 2.0 for velocity. This bug was invisible in the
lid-driven cavity validation, which agreed with Ghia et al. to 0.007.

## Grid Convergence Index

Following Celik et al. (2008), for three grids with ratios \(r_{21}, r_{32}\):
observed order \(p\) (fixed-point iteration for non-constant ratio), Richardson extrapolation
\(\phi_{ext} = (r^p\phi_1-\phi_2)/(r^p-1)\), and the fine-grid index
\(GCI = 1.25\,|(\phi_1-\phi_2)/\phi_1|/(r^p-1)\). The asymptotic-range check uses
\(GCI_{32}/(r^pGCI_{21})\approx 1\). Implemented in `flowtrace.validation.gci` and applied to the
natural-convection Nusselt number and the cavity centreline minimum.

## Acceptance criteria

Each benchmark states its criterion next to the result (for example "within 2 % of de Vahl Davis",
or "observed order > 1.8"). Criteria are set from the reference's own uncertainty and typical
published model accuracy. They are never tuned after the fact to make a result pass: a failing
benchmark is reported as failing.
