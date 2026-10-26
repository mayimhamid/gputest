# Density-based compressible solver

`flowtrace.compressible` solves the Euler or Navier–Stokes equations of an ideal gas on any
Fluent mesh: 2-D, 3-D (any polyhedra), or 2-D axisymmetric with x the axis and y the radius,
as Fluent uses. It is a separate solver from the pressure-based SIMPLEC one, close to Fluent's
*density-based implicit* solver. The [design note](../design/compressible-flow.md) proposed a
pressure-based extension instead; the density-based route was chosen because the target cases
(nozzles, rocket plumes, supersonic external flow) are shock-dominated and Fluent's reference
results for them come from its density-based solver.

## Discretisation

| Part | Method |
|---|---|
| Unknowns | conservative `Q = (ρ, ρu, ρE)` at cell centres |
| Inviscid flux | HLLC (Toro 2009, ch. 10), wave speeds from Roe averages (Einfeldt) |
| Second order | MUSCL on primitive `(ρ, u, p)` with least-squares gradients and a Venkatakrishnan (default, `K = 5`) or Barth–Jespersen limiter; a face falls back to first order when the reconstruction would make ρ or p negative |
| Viscous flux | constant or Sutherland viscosity, face gradients from averaged cell gradients with a directional correction, Fourier heat flux at constant Prandtl number |
| Axisymmetric | radius-weighted face areas and volumes, plus the source `p − 2μ(v/r − ∇·u/3)` in the radial momentum equation |
| Steady time stepping | implicit Euler with a local time step: the first-order Rusanov Jacobian is applied matrix-free with Jacobi sweeps, CFL ramped from `cfl` up to `cfl_max` while the residual falls |
| Transient | SSP-RK3 with a global time step |
| Turbulence | k-ω SST (Menter 2003), see below |

Every operation is a face or cell loop, so the same code runs on NumPy and CuPy arrays.

## Boundary conditions

Ghost states on boundary faces; Fluent zone types map to these kinds automatically.

| Kind | Fluent | Specified |
|---|---|---|
| `pressure-inlet` | pressure-inlet | `p0`, `T0`, `direction`; `p` while the inflow is supersonic |
| `pressure-outlet` | pressure-outlet | static `p` (subsonic; extrapolated when supersonic), `T` for backflow |
| `far-field` | pressure-far-field | `mach`, `p`, `T`, `direction` (Riemann invariants) |
| `velocity-inlet` | velocity-inlet | `velocity`, `T` |
| `supersonic-inlet` | — | `velocity`, `p`, `T` |
| `wall` | wall | no-slip (slip in Euler runs); adiabatic or `wall_T` |
| `slip-wall`, `symmetry` | symmetry, axis | zero normal velocity |

## k-ω SST

The SST model is loosely coupled: after each mean-flow iteration, `k` and `ω` take one implicit
step with the same local time step, first-order upwind convection from the HLLC mass fluxes,
central diffusion and point-implicit destruction, relaxed with Jacobi sweeps. Production is
`μ_t S² − ⅔ρk∇·u`, limited to `10 β* ρkω`; in axisymmetric cases `S` includes the hoop strain
`v/r`. Walls use the low-Reynolds treatment (`k = 0`, `ω = 60ν/(β₁y²)` in the wall cells), so the
first cell should sit at y⁺ ≈ 1. Turbulent heat flux uses `Pr_t = 0.85`. Inflow `k` and `ω`
come from each zone's `turb_intensity` and `turb_visc_ratio`.

## Native kernels

The HLLC flux, the most expensive loop, has a C++/OpenMP kernel (`ft_hllc_flux` in
`fvm_kernels.cpp`) and a CUDA kernel (same name in `fvm_kernels.cu`). The solver uses them
automatically in FP64 (`describe()["flux_kernel"]`: `native-cpu`, `native-cuda` or `array`). On
1M 3-D faces with 4 threads, the C++ kernel takes 0.037 s per call against 3.25 s for the
vectorised NumPy expression. `tests/test_compressible.py` checks that the C++ kernel agrees with
the array path to 1e-12; `tests/test_gpu_real.py` does the same for the CUDA kernel on a real GPU.

## Verification

All from `tests/test_compressible.py` unless stated otherwise.

| Case | Measured | Reference |
|---|---|---|
| Sod tube, t = 0.2, density L1 error | 6.0e-3 (100 cells), 3.2e-3 (200 cells): order 0.9 | exact Riemann solution; order → 1 with shocks |
| First vs second order (Sod, 100 cells) | first-order error > 1.4 × second-order | — |
| Oblique shock, M = 2, 10° wedge, 80×40 | p₂/p₁ = 1.7067 | 1.7066 (θ–β–M) |
| Same on 24-face polyhedra, 3-D, via `.msh` | within 2 % | 1.7066 |
| Axisymmetric gas at rest (warped mesh, laminar and SST) | \|u\| < 1e-8 after 20 iterations | exact balance |
| SST flat plate, M = 0.5, Re_L = 1.1e7, y⁺ ≈ 0.6 | Cf 9–13 % below White's correlation, 5–10 % below 0.0592 Re⁻⁰·² (60×50 cells) | White; power law |

The flat-plate deficit is consistent with the coarse streamwise grid and the laminar run-up of
SST at 1 % free-stream turbulence; a refined-grid run is in the changelog.

## Not yet validated

* **FFF-7 against Fluent.** The comparison needs the Fluent case and its results, which are not
  in this repository. Run the same mesh with `flowtrace compressible` and compare wall pressure
  and Mach-number profiles with Fluent's exports.
* **The rocket mesh.** Its 3.19M polyhedral cells read in 1.5 s, and polyhedral meshes are
  tested end to end on a smaller case, but the full case has not been run with this solver.
* **CUDA kernels.** They compile through NVRTC at first use and are tested by
  `tests/test_gpu_real.py`, which needs a CUDA GPU; this release was built without one.
* Convergence on high-aspect-ratio boundary-layer meshes is slow (point-Jacobi relaxation).
  A line-implicit or GMRES smoother is the next step.

## Usage

```bash
flowtrace compressible nozzle.msh --case nozzle.json --export out/
```

`nozzle.json` holds `CompressibleCase` settings and per-zone boundaries, for example:

```json
{"axisymmetric": true, "turbulence": "k-omega-sst", "cfl_max": 50,
 "boundaries": {"inlet":  {"kind": "pressure-inlet", "p0": 3.0e6, "T0": 3000, "p": 2.9e6},
                "outlet": {"kind": "pressure-outlet", "p": 101325},
                "axis":   {"kind": "symmetry"}}}
```

`out/` receives `<mesh>.ip` (pressure, velocity, temperature, density, k, ω) for Fluent's
*File → Interpolate* and `<mesh>.npz`. Many cases run concurrently on every core and the GPU with
`BatchJob(..., solver="density-based")` (see [batch runs](../user-guide/cli.md)).
