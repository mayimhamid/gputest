# Turbulence models

RANS models for the unstructured solver (`flowtrace.unstructured.turbulence`). The Boussinesq
eddy viscosity \(\mu_t\) enters momentum as \(\mu+\mu_t\) on faces, plus the explicit
\(\nabla\cdot(\mu_t(\nabla\mathbf u)^T)\) term. The isotropic \(\tfrac23\rho k\) part is absorbed
into the pressure. The energy equation uses \(k/c_p + \mu_t/Pr_t\) with \(Pr_t = 0.85\).

## k-ω SST (`k-omega-sst`)

Menter, Kuntz & Langtry (2003), the form used by Fluent's *SST k-omega* and NASA's Turbulence
Modeling Resource:

\[
\nabla\cdot(\rho\mathbf u k) = \nabla\cdot\big((\mu+\sigma_k\mu_t)\nabla k\big) + \tilde P_k - \beta^*\rho k\omega,
\]
\[
\nabla\cdot(\rho\mathbf u \omega) = \nabla\cdot\big((\mu+\sigma_\omega\mu_t)\nabla \omega\big) + \gamma\rho S^2
 - \beta\rho\omega^2 + 2(1-F_1)\rho\sigma_{\omega2}\frac{\nabla k\cdot\nabla\omega}{\omega},
\]
\[
\mu_t = \frac{\rho a_1 k}{\max(a_1\omega, S F_2)},\qquad
\tilde P_k = \min(\mu_t S^2,\ 10\beta^*\rho k\omega).
\]

Coefficients are blended with \(F_1\) between the inner set (σk = 0.85, σω = 0.5, β = 0.075) and
the outer set (σk = 1, σω = 0.856, β = 0.0828), with β\* = 0.09 and a₁ = 0.31. \(F_1\) and
\(F_2\) use the wall distance, computed exactly as the point-to-segment distance in 2-D. The
cross-diffusion term is applied implicitly when it is negative.

**Wall treatment (y⁺-insensitive).** The near-wall cell value
\(\omega_P = \sqrt{\omega_{vis}^2+\omega_{log}^2}\) with
\(\omega_{vis} = 6\nu/(\beta_1 y^2)\) and \(\omega_{log} = \sqrt{k}/(C_\mu^{1/4}\kappa y)\). The
wall shear comes from Spalding's law of the wall, solved for \(u_\tau\) by Newton iteration. It
is exactly laminar for y⁺ < 5 and the log law (κ = 0.41, B = 5.2) for y⁺ > 30, so the model is
accurate on wall-resolved meshes and consistent on coarser ones.

## Realizable k-ε (`k-epsilon-realizable`)

Shih et al. (1995), with Fluent's constants:

\[
\nabla\cdot(\rho\mathbf u\varepsilon) = \nabla\cdot\big((\mu+\mu_t/\sigma_\varepsilon)\nabla\varepsilon\big)
 + \rho C_1 S\varepsilon - \rho C_2\frac{\varepsilon^2}{k+\sqrt{\nu\varepsilon}},\qquad
C_1 = \max\!\Big(0.43, \frac{\eta}{\eta+5}\Big),\ \eta = \frac{Sk}{\varepsilon},
\]

with C₂ = 1.9, σk = 1, σε = 1.2, and
\(\mu_t = \rho C_\mu k^2/\varepsilon\), \(C_\mu = 1/(A_0 + A_s k U^*/\varepsilon)\), A₀ = 4.04,
\(A_s = \sqrt6\cos\phi\), \(\phi = \tfrac13\arccos(\sqrt6 W)\).

## Standard k-ε (`k-epsilon-standard`)

Launder & Spalding (1974): C_μ = 0.09, C₁ = 1.44, C₂ = 1.92, σk = 1, σε = 1.3.

**Wall functions (both k-ε models).** Scalable standard wall functions, as Fluent's *Scalable
Wall Functions*. With \(u^* = C_\mu^{1/4}k_P^{1/2}\) and \(y^* = \max(\rho u^* y_P/\mu,\ 11.225)\):
the wall shear is \(\tau_w = \rho\kappa u^* U_P/\ln(Ey^*)\); the wall-cell production is
\(G_k = \tau_w^2/(\kappa\rho u^* y_P)\); and the near-wall dissipation is fixed at
\(\varepsilon_P = C_\mu^{3/4}k_P^{3/2}/(\kappa y_P)\).

## Boundary conditions

Inlets and outlet backflow take \(k = \tfrac32(IU)^2\) and ω or ε from the viscosity ratio
\(\mu_t/\mu\) (Fluent defaults I = 5 %, ratio 10), or explicit values. Outlets and symmetry
planes are zero-gradient.

## Numerics and robustness

* k and ω/ε use first-order upwind by default (`turb_scheme`) and Jacobi-BiCGSTAB. Destruction
  terms are implicit and values are bounded below.
* Under-relaxation defaults are SST 0.8 and k-ε 0.5. **This deviates from Fluent's 0.8 for k-ε:**
  with the segregated update used here, k-ε at 0.8 was measured to oscillate, and to laminarise
  completely on a 24-cell channel. 0.5 converged on every case tested.
* High-Re k-ε has a laminar fixed point (k = 0). Start turbulent cases from a turbulent initial
  field (`turb_init`) or the inlet values.

## Validation summary

| Case | Model | Result (full tier) | Reference |
|---|---|---|---|
| Channel Re_τ = 392 | SST, y⁺ ≈ 0.25 | U⁺ within 4.1 % at all 16 DNS points | MKM 1999 DNS |
| Channel Re_τ = 392 | standard k-ε, WF | within 3.5 % for y⁺ ≥ 30 | MKM 1999 DNS |
| Channel Re_τ = 392 | realizable k-ε, WF | within 7.6 % for y⁺ ≥ 30 (core over-predicted) | MKM 1999 DNS |
| Flat plate Re_L = 5×10⁶ | SST | Cf 3–4 % below Schultz-Grunow, ~10 % below White | correlations |
| Backward-facing step | SST | x_r/H = 6.61 | 6.26 ± 0.10 (Driver & Seegmiller) |

The numbers above were measured with `flowtrace benchmark --tier full`. The authoritative,
regenerated values are in the [validation report](../validation/index.md).
