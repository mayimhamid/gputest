# Conjugate heat transfer, rotating frames, periodic zones

## Conjugate heat transfer

`ConjugateSolver` (created by `make_solver` when the case has `solids`) splits the mesh:

* **Flow** (momentum, pressure, turbulence) is solved on the fluid cell zones only. Faces between
  fluid and solid cells become boundary zones named `interface-<solid zone>`, treated as no-slip
  walls.
* **Energy** is solved on *all* cells:
  \(\nabla\cdot(F T) = \nabla\cdot(\Gamma\nabla T) + q'''/c_p\), with \(\Gamma = k/c_p\) per cell
  (plus \(\mu_t/Pr_t\) in turbulent fluid). Mass fluxes are zero in solids, and face diffusivities
  are harmonic means,
  \(\Gamma_f = \big((1-w)/\Gamma_P + w/\Gamma_N\big)^{-1}\).

The harmonic mean makes the heat flux continuous across material interfaces, and it is exact for
1-D composite walls (verified to 4e-15 by `cht_composite_wall`). `heat_flow()` also reports the
heat crossing all fluid–solid interfaces.

```python
case.solids = {"heater": {"conductivity": 16.0, "source": 2e5}}   # W/(m K), W/m^3
```

## Rotating walls

`BoundarySpec("wall", rotation={"omega": 10.0, "origin": [0, 0, 0], "axis": [0, 0, 1]})` sets the
wall velocity to \(\boldsymbol\omega\times(\mathbf x-\mathbf x_0)\) in the absolute frame.

## Moving reference frames (MRF)

For a cell zone rotating with \(\boldsymbol\Omega\), FlowTrace uses the absolute-velocity
formulation, as Fluent does:

\[
\nabla\cdot(\rho\,\mathbf u_r\otimes\mathbf u) + \rho\,\boldsymbol\Omega\times\mathbf u = -\nabla p + \nabla\cdot\boldsymbol\tau,
\qquad \nabla\cdot(\rho\mathbf u) = 0,\qquad \mathbf u_r = \mathbf u - \boldsymbol\Omega\times\mathbf r .
\]

The convecting mass flux is \(F - \rho(\boldsymbol\Omega\times\mathbf r_f)\cdot\mathbf S_f\);
continuity uses the absolute flux. Walls bounding an MRF zone move with it unless they have their
own velocity or rotation. An MRF interface should be a surface of revolution, where the frame
velocity is tangential.

Verification: Taylor–Couette flow between cylinders, solved with a rotating wall and with an MRF
inner ring. Both converge to the exact profile \(u_\theta = Ar + B/r\) at second order
(`taylor_couette`).

## Translational periodicity

`make_periodic(mesh, "a", "b")` pairs two boundary zones whose faces coincide after a translation
(detected or given). Paired faces become interior faces. The neighbour cell centre is seen
through the translation, so gradients, interpolation and fluxes need no special cases. Pairs are
read from and written to Fluent's `.msh` section 18.

Fully developed flow is driven by `body_force` (the mean pressure gradient), as in Fluent's
periodic flow with a specified pressure gradient. Verified against Poiseuille flow on quads and
triangles (`periodic_poiseuille`) and used for the turbulent channel.
