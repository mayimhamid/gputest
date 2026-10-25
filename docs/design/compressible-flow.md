# Design: compressible flow in FlowTrace

Status: **proposal** (not implemented). Written before any code, as the roadmap asks.

## Goal

Extend the pressure-based unstructured solver to compressible flow. The target is subsonic to
moderately supersonic external and internal aerodynamics (M ≲ 3), the regime Fluent covers with its
pressure-based solver and the *ideal gas* density law. The warm-start workflow must still work: FlowTrace
produces an initial field that Fluent can interpolate (`.ip` with pressure, velocity, temperature
and density).

Out of scope for the first version: hypersonics, real-gas equations of state, combustion, and
density-based (coupled implicit) solvers.

## Why pressure-based

FlowTrace already has a validated SIMPLEC solver (MMS second order on warped triangles, Rhie–Chow
with a Green–Gauss face-pressure force, AMG pressure correction). The standard extension to
all-speed flow keeps that structure. It is also what Fluent's pressure-based solver does, so the
two tools can share initial fields and boundary conditions one to one.

## Equations

Conservative variables are kept in primitive form, as in the incompressible solver:

* continuity  `∂ρ/∂t + ∇·(ρu) = 0`
* momentum    `∂(ρu)/∂t + ∇·(ρu⊗u) = −∇p + ∇·τ`
* energy (total enthalpy) `∂(ρE)/∂t + ∇·(ρuH) = ∇·(k∇T) + ∇·(τ·u)`, with `H = h + |u|²/2`, `h = c_p T`
* equation of state `ρ = p_abs / (R T)` with `p_abs = p + p_operating` (Fluent convention)

## Pressure-correction for compressible flow

The key change is the pressure-correction equation. The face mass flux responds to pressure
through both the velocity and the density:

```
F' = ρ_f* (d_f α_f) (p'_P − p'_N)  +  (F*/ρ_f*) (∂ρ/∂p)_f p'_upwind
```

The second term is convective in `p'` (with `∂ρ/∂p = 1/(R T)` for an ideal gas). It turns the
pressure equation from Poisson-type (elliptic) into convection–diffusion type. That allows shocks
to be captured and removes the low-Mach stiffness of density-based solvers.

Consequences for the code:

1. `fv.matrix` already assembles non-symmetric face coefficients. The pressure-correction matrix
   gains an upwind convective part, so AMG-PCG is replaced by **AMG-preconditioned BiCGSTAB**
   (the existing `bicgstab` plus SA-AMG as preconditioner; pyamg works as well).
2. The density update `ρ = (p + p_op)/(R T)` happens after each correction. Face densities are
   upwinded (second order with a limiter), not linearly interpolated, so the scheme stays monotone
   at shocks.
3. Under-relaxation: density 1.0, pressure 0.3–0.5 for SIMPLE, SIMPLEC with consistent `d`.

## Energy

* Total-enthalpy form so that stagnation conditions are conserved exactly in inviscid flow. This
  is checked by an inviscid nozzle (isentropic Mach–area relation) and a constant-H test.
* Viscous heating `∇·(τ·u)` and pressure work enter as explicit sources.
* Temperature from `h` (constant c_p first; NASA polynomials later).

## Boundary conditions (Fluent names)

| FlowTrace kind | Fluent equivalent | Specified |
|---|---|---|
| `pressure-inlet` | pressure-inlet | total pressure, total temperature, flow direction |
| `mass-flow-inlet` | mass-flow-inlet | mass flux, total temperature |
| `pressure-outlet` | pressure-outlet | static pressure (subsonic); extrapolated when supersonic |
| `pressure-far-field` | pressure-far-field | Mach, static p, T, direction (Riemann invariants) |
| `wall` | wall | no-slip, thermal BC as now |

Supersonic outlets switch automatically when the face Mach number is above 1, as Fluent does.

## Turbulence

The SST and k-ε implementations carry over with density inside the transport terms (`ρk`, `ρω`).
Two additions are needed:

* Compressibility corrections are off by default. Fluent's SST has none by default; the
  Sarkar/Wilcox dilatation corrections are options.
* Wall functions with variable density use the van Driest-transformed log law, also Fluent's
  default for compressible wall functions.

## Discretisation details

* Convective fluxes: second-order upwind with the existing Barth–Jespersen-type bound (the
  `clip` of the high-order face value). Shocks need it; MMS checks that smooth regions stay at
  second order.
* Pseudo-transient relaxation (local CFL ~ 5–50) for robustness on shocked cases, implemented
  as an implicit `ρV/Δt_local` diagonal term in every equation.
* GPU: no new kernel types. Everything is the same face-gather/scatter plus sparse algebra, so
  the CuPy path is unchanged.

## Verification and validation plan

These are the acceptance criteria before the feature is advertised.

1. **MMS** (compressible, subsonic, smooth): observed order ≥ 1.8 for ρ, u, p, T on warped
   triangles.
2. **Quasi-1D converging–diverging nozzle**, inviscid: Mach distribution against the isentropic
   relation, normal-shock position against the analytic value for a given back pressure.
3. **Sod-like steady shock (2D normal shock in a duct)**: Rankine–Hugoniot jump conditions
   within 1 %.
4. **Supersonic wedge (M = 2, 10°)**: oblique-shock angle and pressure ratio against
   θ–β–M theory.
5. **NASA TMR RAE 2822 case 9** (transonic airfoil, SST): Cp and Cf against experiment.
   Mesh-heavy, so full tier only.
6. **Warm start into Fluent** (needs a licence): iteration savings for the RAE 2822 case,
   measured with `flowtrace warmstart-study`.

## Data exchange

* `.ip` gains `density` and `total-temperature`. The `.prof` writer gains `total-pressure` and
  `total-temperature` profiles for pressure-inlets.
* The journal/PyFluent generator sets `materials/fluid/air/density = ideal-gas`, the operating
  pressure and the compressible BC types.

## Effort estimate

| Work package | Estimate |
|---|---|
| Density/EOS plumbing, total-enthalpy energy equation | 1 week |
| Compressible pressure correction + BiCGSTAB-AMG | 1–2 weeks |
| Compressible BCs (pressure inlet/outlet, far field) | 1 week |
| Verification suite (MMS, nozzle, shock, wedge) | 1 week |
| RAE 2822 validation + documentation | 1 week |

## Risks

* Pressure–density coupling at high Mach can stall SIMPLEC. Mitigation: pseudo-transient
  terms, and a coupled pressure–velocity solve as a later option.
* AMG on non-symmetric pressure matrices: smoothed aggregation is less robust there.
  Mitigation: use pyamg's root-node or AIR (approximate ideal restriction) solvers when they are
  installed.
