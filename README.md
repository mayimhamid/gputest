<div align="center">

# FlowTrace

**A verified, hardware-aware CFD solver and warm-start companion for Ansys Fluent**

![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Backends](https://img.shields.io/badge/backends-NumPy%20%7C%20Numba%20%7C%20CuPy%2FCUDA-blue)
![Tests](https://img.shields.io/badge/tests-168%20passing-brightgreen)
![Benchmarks](https://img.shields.io/badge/validation-68%2F70%20checks%20pass-yellowgreen)

[English](README.md) · [فارسی](README.fa.md) · [Documentation](docs/index.md) · [Validation report](docs/validation/index.md) · [Changelog](CHANGELOG.md)

</div>

FlowTrace solves incompressible flow, heat transfer and RANS turbulence. It measures the CPU
and GPU of the machine it runs on and places each stage of the solve where it is fastest. It
also reads and writes the files Ansys Fluent uses.

Its main job is to **compute a good initial field quickly**, which Fluent then interpolates onto
its own mesh and continues from (a *warm start*). Everything it claims is backed by a script in
this repository.

> Ansys and Fluent are registered trademarks of ANSYS, Inc. FlowTrace is an independent
> open-source project. It is compatible with Fluent's file formats, but it is not affiliated with
> or endorsed by ANSYS, Inc.

## Highlights

* **Verified, not only validated.** Manufactured solutions on warped meshes show second-order
  convergence for velocity and temperature on quads and triangles. They also exposed a real
  pressure-force bug that the cavity validation had missed.
* **Checked against the classic references.** These include Ghia's cavity, de Vahl Davis's
  natural convection and the Moser–Kim–Mansour channel DNS (SST within 4.1 %). They also cover
  flat-plate skin-friction correlations and the Driver & Seegmiller step. **68 of 70 acceptance
  checks pass.** The two failures are listed below, not hidden.
* **A companion for Fluent.** FlowTrace writes Fluent's own `.ip` interpolation file, plus a
  journal and a PyFluent script that open *your* case. In the proxy study the FlowTrace field
  cut the iterations to the converged solution by ×1.4 to ×2.4 in four of five cases. In the
  fifth it made the run slower; the reason is below.
* **Measured, not assumed, hardware placement.** The planner times real solver steps. On a
  laptop RTX 4060 the structured solver runs a 1024² step about 30× faster than the 16-thread
  CPU path (FP32 on the GPU, FP64 on the CPU). The unstructured solver gains from the GPU only above about 16k cells, so small
  cases stay on the CPU.
* **Fluent file formats.** `.msh`/`.cas`, CFF `.cas.h5`/`.dat.h5`, `.ip` and `.prof`. The HDF5
  files are round-trip tested but not yet checked against Fluent-written files.
* **Offline dashboard** in English and Persian, with live residuals, the execution plan and
  exportable run reports.

## Quick start

**Windows:** double-click `start_flowtrace.bat`. It creates a private `.venv`, installs
FlowTrace, adds the CUDA backend if an NVIDIA GPU is present, and opens the dashboard.

**Any platform** (Python 3.9+; prebuilt CuPy/Numba wheels exist for 3.10–3.13):

```bash
pip install -e ".[cpu,amg,h5,report]"   # NumPy/SciPy + Numba, pyamg, h5py, matplotlib
pip install -e ".[gpu]"                 # optional: NVIDIA GPU (CuPy for CUDA 12 with headers)
flowtrace probe                         # what FlowTrace sees on this machine
flowtrace                               # dashboard at http://127.0.0.1:8765
```

## Typical workflows

```bash
# Pre-solve an existing Fluent case and continue it in Fluent
flowtrace fluent-warmstart my_case.cas.h5 --turbulence k-omega-sst --bc inlet=inlet:12.5 --out ws/
fluent 3ddp -g -t8 -i ws/warmstart.jou            # or: python ws/warmstart_pyfluent.py

# Solve a Fluent mesh in FlowTrace (laminar/RANS, heat transfer, solids, MRF, periodic)
flowtrace solve duct.msh --bc inlet=inlet:1.0:T300 --bc heater=wall::q5000 --energy \
    --turbulence k-omega-sst --solid plate:16:2e5 --export out/

# Built-in structured cases; the planner measures CPU vs GPU first
flowtrace run cavity --re 1000 --n 512

# Check a mesh against Fluent's quality recommendations
flowtrace quality my_case.msh

# Re-run the verification and validation suite yourself
flowtrace benchmark --tier quick          # ~2 min;  --tier full: everything below, ~40 min
```

## Validation at a glance

All numbers below come from `flowtrace benchmark --tier full` (CPU, FP64). It was run on
26 October 2026 on a laptop (AMD Ryzen 9 7940HS, 8 cores/16 threads, 15 GB RAM, Windows 11,
Python 3.13) and took 37 minutes. The full table, with all 70 acceptance checks, plots and
the cases not shown here, is in the [validation report](docs/validation/index.md). The raw
results are in [`benchmarks/results/`](benchmarks/results/).

| Case | Reference | Quantity | FlowTrace | Criterion | |
|---|---|---|---|---|---|
| Lid-driven cavity Re = 100, SIMPLEC, skewed triangles (60°) | Ghia et al. 1982 | max centreline deviation u / v | 0.0056 / 0.0089 | < 0.015 / 0.02 | pass |
| Lid-driven cavity Re = 1000, SIMPLEC 128² | Ghia et al. 1982 | max centreline deviation u / v | 0.0053 / 0.0100 | < 0.02 | pass |
| Natural convection Ra = 10⁵, MAC + FMG | de Vahl Davis 1983 | Nu, hot wall | 4.569 (ref. 4.519) | within 2 % | pass |
| Natural convection Ra = 10⁴, 3 grids | de Vahl Davis 1983 | Richardson-extrapolated Nu; GCI | 2.245 (ref. 2.243); 0.08 % | within 1 % | pass |
| Channel Re_τ = 392, k-ω SST, y⁺ ≈ 0.25 | Moser, Kim & Mansour 1999 (DNS) | max \|U⁺ error\|, y⁺ ≥ 1 | 4.1 % | < 5 % | pass |
| Channel Re_τ = 392, standard k-ε, wall functions | Moser, Kim & Mansour 1999 | max \|U⁺ error\|, y⁺ ≥ 51 | 3.5 % | < 5 % | pass |
| Channel Re_τ = 392, realizable k-ε, wall functions | Moser, Kim & Mansour 1999 | max \|U⁺ error\|, y⁺ ≥ 51 | 7.6 % | < 10 % | pass |
| Flat plate Re_L = 5×10⁶, k-ω SST | Schultz-Grunow correlation | mean Cf deviation, 10⁶ < Re_x < 10⁷ | −5.3 % | within 6 % | pass |
| Flat plate Re_L = 5×10⁶, realizable k-ε | Schultz-Grunow correlation | mean Cf deviation, 10⁶ < Re_x < 10⁷ | −7.0 % | within 6 % | **FAIL** |
| Backward-facing step, k-ω SST | Driver & Seegmiller 1985 | reattachment length x_r/H | 6.61 (ref. 6.26) | within 10 % | pass |
| Taylor–Couette flow, rotating wall / MRF zone | exact solution | observed order of accuracy | 2.01 / 2.00 | > 1.8 | pass |
| Conjugate heat transfer, heated channel | energy balance | heat through the fluid–solid interface | 4 W/m (exact 4) | within 10⁻⁶ | pass |
| Warm start, cylinder Re = 40 | cold start | iterations cold / warm | ×0.65 | > 1 | **FAIL** |

**The two failures.** With wall functions at y⁺ ≈ 56, the realizable k-ε model under-predicts
flat-plate skin friction by 7.0 %, against a 6 % criterion. SST on a wall-resolved mesh passes;
the cause has not been isolated. The cylinder warm start is slower because the pre-solve and the
target mesh resolve the cylinder differently, so their converged wakes differ by about a quarter.
Criteria were fixed before the run and were not relaxed afterwards. Both failures are analysed
in the [validation report](docs/validation/index.md#failing-checks).

**Code verification.** Observed order of accuracy of the L2 error (finest grid pair), from
manufactured solutions on systematically refined, smoothly warped meshes:

| Solver and mesh | u | v | p | T |
|---|---|---|---|---|
| MAC (structured), walls + energy | 2.01 | 2.01 | 2.00 | 2.00 |
| SIMPLEC, warped quadrilaterals | 2.01 | 2.01 | 1.99 | 1.98 |
| SIMPLEC, warped triangles | 2.03 | 2.02 | **1.20** | 2.00 |

Pressure on triangles converges more slowly than second order. Its acceptance threshold is 1.0,
and the docs say so rather than claiming second order.

The manufactured solutions found a real bug that the cavity validation had missed. With the
least-squares pressure gradient previously used in momentum, the error on triangles stopped
decreasing. The Green–Gauss face-pressure force, which Fluent also uses, fixed it. See the
[verification methodology](docs/theory/verification.md).

## Warm start: how many iterations does it save?

FlowTrace's structured solver pre-solves each case and writes a Fluent `.ip` file. The file is
interpolated onto a *different* mesh, and FlowTrace's SIMPLEC solver then stands in for Fluent's
pressure-based solver. Both runs are counted until the velocity (and temperature) field is
within 10⁻⁴ of the converged solution:

| Case | Target cells | Cold start | Warm start | Saved | Pre-solve |
|---|---:|---:|---:|---:|---:|
| Lid-driven cavity, Re = 400 | 3,200 | 1,162 | 488 | ×2.4 | 8.2 s |
| Developing channel, Re = 100 | 1,152 | 68 | 47 | ×1.4 | 0.4 s |
| Backward-facing step, Re = 200 | 5,616 | 772 | 418 | ×1.8 | 224 s |
| Cylinder in a channel, Re = 40 | 10,308 | 700 | 1,085 | **×0.65** | 332 s |
| Heated cavity, Ra = 10⁵ | 3,200 | 316 | 158 | ×2.0 | 4.0 s |

How to read this:

* **The cylinder gets slower.** The pre-solve resolves the cylinder with 12 cells per diameter,
  the target mesh with 9. Their converged wakes differ (1.87 D vs 2.33 D of recirculation), so
  the interpolated field must be corrected, and that takes longer than starting from rest.
  A warm start helps when both meshes resolve the geometry alike.
* **The savings are in iterations, not yet in wall time.** On these small 2-D cases the
  pre-solve takes longer than the SIMPLEC run it shortens (step: 224 s against 8 s for the whole
  cold run). The case for a warm start is a large Fluent mesh, where one iteration costs far
  more. That has not been measured here.
* **The Fluent column has not been run** (no licence). `flowtrace warmstart-study` writes the
  cold/warm journals and runs them through PyFluent when Fluent is available. See the
  [study page](docs/validation/warmstart.md).

## Hardware

Measured on the same laptop, with an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). Its FP64 rate is
1/64 of FP32, so FlowTrace runs the structured solver's fields in FP32 on it. In FP64, GPU results
agree with the CPU to within 7×10⁻¹⁴ ([GPU log](docs/validation/gpu.md)).

**Structured (MAC) solver**, lid-driven cavity Re = 1000, ms per time step
(`flowtrace gpu-report`):

| Grid | NumPy (FP64) | Numba, 16 threads (FP64) | CuPy fused kernels (FP32) | Planner picks |
|---|---:|---:|---:|---|
| 64² | 0.38 | 0.48 | 1.79 | NumPy |
| 128² | 0.91 | 0.84 | 1.69 | Numba |
| 256² | 3.71 | 2.43 | 1.71 | CuPy |
| 512² | 57.6 | 31.5 | 6.50 | CuPy |
| 1024² | 262 | 137 | 4.59 | CuPy |

At 512² the fused CUDA kernels were slower than CuPy's vectorised path (4.32 ms), and a second
run reproduced this. It has not been investigated yet.

**Unstructured (SIMPLEC) solver**, cavity, FP64, ms per iteration
(`benchmarks/simplec_iteration_cost.py`):

| Cells | 1,024 | 4,096 | 16,384 | 65,536 | 262,144 |
|---|---:|---:|---:|---:|---:|
| CPU (NumPy) | 1.6 | 4.7 | 20.4 | 114 | 486 |
| GPU (CuPy) | 15.2 | 16.4 | 18.2 | 18.6 | 30.0 |

A GPU iteration costs at least ~15 ms (kernel launches and host synchronisation), so the GPU
wins only above ~16k cells. The planner times both devices and always ended on the faster one
in the [planner study](docs/validation/planner.md). On 100-iteration runs, though, its timing
overhead exceeded the study's own 15 % budget at 2 of 6 sizes.

Run `flowtrace gpu-report` on your own machine: these numbers do not transfer to other hardware.

## What is inside

| Area | Capabilities |
|---|---|
<<<<<<< HEAD
| Structured solver | 2-D incompressible Navier–Stokes + Boussinesq energy on a MAC grid, exact pressure projection (DCT/DST/FFT or GEMM, multigrid with obstacles), QUICK/TVD, AB2 / Crank–Nicolson, FMG |
| Unstructured solver | Fluent-style pressure-based SIMPLE/SIMPLEC on any cell shape (2-D/3-D), Rhie–Chow, Green–Gauss face-pressure force, AMG-PCG pressure correction |
| Turbulence | k-ω SST (Menter 2003, y⁺-insensitive wall treatment), realizable and standard k-ε with scalable wall functions |
| Zones | Conjugate heat transfer (solid zones, volumetric sources), moving reference frames, rotating walls, translational periodicity |
| Fluent files | `.msh`/`.cas` (ASCII, binary, gzip), CFF `.cas.h5`/`.msh.h5`/`.dat.h5`, `.ip`, `.prof`, journals, PyFluent scripts |
| Mesh quality | Orthogonal quality, equiangle skewness, aspect ratio, non-orthogonality, volume ratio (Fluent-style report) |
| Hardware | NumPy, Numba JIT kernels, CuPy + fused CUDA C++ kernels (NVRTC); C++/OpenMP kernels for the unstructured solver (g++, built on first use) and their CUDA counterparts (`.cu`); measured placement; `flowtrace batch` runs many cases on all cores and the GPU at once; FP32 on consumer GPUs |
| Dashboard | Offline web UI: live residuals and fields, execution plan, saved projects, HTML run reports, English/Persian (RTL), light/dark, keyboard and screen-reader friendly |
| V&V tooling | `flowtrace benchmark` (JSON + HTML/PDF/Markdown), MMS, GCI, warm-start study, GPU log |
=======
| **Hardware layer** (`flowtrace/hardware`) | Probes CPU (model, cores, SIMD), RAM, disk, battery, NVIDIA GPUs (VRAM, compute capability, Tensor/RT cores, FP64:FP32 ratio), libraries. Micro-benchmarks bandwidth, GEMM, DCT, stencils, launch latency, PCIe. **Memory budget** before allocating. **Planner** that times real solver steps on every backend and explains every placement. |
| **Structured solver** (`flowtrace/structured`) | 2-D incompressible Navier–Stokes + Boussinesq energy on a staggered (MAC) grid. QUICK / van Leer TVD / upwind / central convection, Adams–Bashforth 2, Crank–Nicolson diffusion where it pays off, exact pressure projection. Walls (moving), inlets (uniform/parabolic), convective outlets, symmetry, periodic, immersed obstacles. FMG grid sequencing for steady cases (like Fluent's FMG initialisation). |
| **Linear algebra** (`flowtrace/linalg`) | Direct separable Poisson/Helmholtz solver: per axis a fast transform (DCT-II/IV, DST-I/II/IV, FFT) or dense generalised eigen-decomposition applied as GEMM (non-uniform grids, any BC mix). Geometric multigrid-PCG for masked domains. Smoothed-aggregation AMG. Backend-agnostic CG/BiCGSTAB. |
| **Unstructured solver** (`flowtrace/unstructured`) | Fluent's pressure-based segregated algorithm: SIMPLE / SIMPLEC, Rhie–Chow, bounded second-order upwind, non-orthogonal correction, least-squares gradients, AMG-PCG pressure correction, energy equation, Fluent-style scaled residuals. Any cell shape, 2-D and 3-D. **Steady or transient** (Euler/BDF2 with inner iterations and Choi's Rhie–Chow correction). |
| **Physical models** | **k-ω SST** (Menter 2003, Fluent's form) with a y+-insensitive wall treatment (Spalding wall shear, blended ω wall value, k wall-function switch at y+ = 11.25), inlet turbulence from intensity + viscosity ratio, turbulent Prandtl number for heat transfer. **P-1 radiation** with Marshak wall emissivity. **Translational periodic** zones driven by a mean pressure gradient. Boussinesq buoyancy. **Species transport** (passive, non-reacting; laminar + turbulent Schmidt diffusion). |
| **Reports** | Per wall zone: pressure/viscous forces, Cd/Cl, y+ (avg/max), skin friction, convective and radiative heat flow; periodic mass flow; boundary mass balance; Cf/Cp/y+ distributions along a wall. |
| **Fluent interop** (`flowtrace/mesh`, `flowtrace/fluent`) | Reads/writes Fluent `.msh` (ASCII, binary single/double, gzip, legacy ASCII `.cas`), repairs left-handed faces, keeps zone names/types. Writes interpolation (`.ip` v3), profile (`.prof`), TUI journal, PyFluent script; optional live PyFluent bridge. |
| **Runtime & UI** (`flowtrace/runtime`, `flowtrace/ui`) | Threaded jobs, event hub, throttled/downsampled frames, asynchronous checkpoints with back-pressure, dashboard over Server-Sent Events (no external JS, works offline). |
>>>>>>> 3d528151e15014be7c185d9c110063ec4634bd8d

## Documentation

* [Installation](docs/install.md) · [Command line](docs/user-guide/cli.md) · [Dashboard](docs/user-guide/dashboard.md) · [Python API](docs/user-guide/python-api.md)
* [Ansys Fluent workflow](docs/user-guide/fluent.md) · [GPU and the planner](docs/user-guide/gpu.md) · [Native kernels (C++/CUDA)](docs/user-guide/native.md)
* Theory: [MAC solver](docs/theory/mac.md) · [SIMPLEC](docs/theory/simplec.md) · [turbulence](docs/theory/turbulence.md) · [CHT, MRF, periodic](docs/theory/zones.md) · [verification](docs/theory/verification.md)
* [Tutorials (Jupyter)](docs/tutorials/) · [Compressible-flow design proposal](docs/design/compressible-flow.md)

Build the site locally with `pip install -e ".[docs]" && mkdocs serve`.

## Limitations

<<<<<<< HEAD
* Incompressible (Boussinesq) flow only. Compressible flow is a [design proposal](docs/design/compressible-flow.md).
* Steady RANS. No LES/DES, multiphase, combustion, radiation or moving/deforming meshes.
* The structured solver is 2-D. The unstructured solver is 3-D capable, but it is written in
  vectorised Python, so very large 3-D cases run slower than in compiled commercial codes.
* The `.cas.h5`/`.dat.h5` writer follows the public VTK reader's layout and has been round-trip
  tested, **but not yet read back into Fluent**. The verified hand-off is `.msh` + `.ip`.
* **The Fluent column of the warm-start study has not been run** (no licence was available).
  It holds labelled placeholders until someone with a licence runs `flowtrace warmstart-study`.
=======
```bash
python -m flowtrace                  # dashboard at http://127.0.0.1:8765
python -m flowtrace probe            # what FlowTrace sees on this machine
python -m flowtrace bench            # measured throughput of every backend
python -m flowtrace run cavity --re 1000 --n 256 --export out/     # + Fluent package
python -m flowtrace run cylinder --re 100 --n-per-d 24 --t-end 150
python -m flowtrace mesh my_case.msh                                # inspect a Fluent mesh
python -m flowtrace solve my_case.msh --bc inlet=inlet:1.0 --bc outlet=outlet --mu 1.8e-5 --rho 1.2 \
       --turbulence k-omega-sst --intensity 0.05 --export out/
python -m flowtrace solve duct.msh --periodic inflow,outflow --dpdx -1,0 --turbulence sst   # periodic flow
python -m flowtrace solve furnace.msh --energy --radiation p1 --absorption 0.5 --bc wall=wall::T1200,e0.8
python -m flowtrace solve mixer.msh --unsteady --dt 1e-3 --t-end 0.5 --inner 20 \
       --species tracer --bc jet=inlet:2.0:tracer=1
python -m flowtrace case turbulent_channel --param re_tau=395     # built-in Fluent-style cases:
python -m flowtrace case flat_plate                                #   flat_plate, radiation_slab,
python -m flowtrace run step --re 20000 --turbulence k-omega-sst  #   periodic_taylor_green
python -m flowtrace validate         # validation suite
```
>>>>>>> 3d528151e15014be7c185d9c110063ec4634bd8d

## Contributing, citing, licence

<<<<<<< HEAD
Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). To cite FlowTrace, use [CITATION.cff](CITATION.cff); a
JOSS-style paper draft is in [`paper/`](paper/paper.md). FlowTrace is released under the
[MIT licence](LICENSE).
=======
* **Case** – built-in cases (lid-driven cavity, natural convection, channel,
  cylinder wake, backward-facing step, Rayleigh–Bénard, Taylor–Green, and the
  Fluent-style SST channel, SST flat plate, P-1 slab, transient periodic
  Taylor–Green) or **drag-and-drop a Fluent mesh**: zones are listed with their
  Fluent types and mapped to boundary conditions you can edit (`T=…`, `q=…`,
  emissivity `e=…`, periodic pairs).
* **Solver** – transient projection or steady SIMPLEC, schemes, CFL, pressure
  solver, FMG, backend/precision (or *Auto*: measure and choose), checkpoints,
  memory estimate; **Models**: laminar / k-ω SST, inlet turbulence, P-1
  radiation, periodic pressure gradient, transient time stepping.
* **Hardware** – everything the probe found, plus one-click benchmarks.
* **Fluent** – download the hand-off package (`.msh`, `.ip`, `.jou`, PyFluent
  `.py`, `.csv`, README).
* Viewport: contours (Turbo/Viridis/Fluent rainbow/cool-warm), streamlines,
  vectors, mesh edges, hover read-out; fields include k, ω, μt/μ, wall
  distance and incident radiation. Right column: Fluent-style residuals,
  performance (steps/s, Mcell/s, time split per stage), monitors (Nu, probes,
  KE), and the **execution plan with the measured reason for every stage**.
  After a run: validation checks and the wall report (forces, Cd, y+, Cf, heat).

## Fluent workflow (warm start)

1. Solve in FlowTrace (fast, any backend), then **Fluent › Download package**.
2. In Fluent: *File › Read › Mesh* `flowtrace.msh` (or your own mesh of the
   same geometry), set models/BCs, *Initialize*, then
   *File › Interpolate › Read and Interpolate* `flowtrace.ip`.
3. Iterate: Fluent starts from a converged field instead of from rest.

Batch: `fluent 2ddp -g -t4 -i run_fluent.jou`, or `python run_pyfluent.py`.
Going the other way, any Fluent `.msh` can be solved in FlowTrace with the same
pressure-based algorithm Fluent uses.

## How the hardware is used (and why)

* The planner **measures**: it builds the real solver on every backend that fits
  in memory and times real steps. On small grids the CPU often wins (GPU launch
  latency); on large grids the GPU does. The UI shows the numbers.
* GPU (CuPy): every field stays resident in VRAM; only scalars (CFL, residuals)
  and ≤4 fps downsampled frames cross PCIe. Consumer GeForce/RTX cards have
  1/32–1/64 FP64 rate, so FlowTrace defaults to FP32 fields there and FP64 on
  data-centre GPUs.
* CPU: Numba-fused stencils and red-black Gauss–Seidel on all cores
  (OpenMP workers set to passive waiting so they do not starve NumPy/SciPy),
  multi-threaded FFTs.
* Pressure: exact O(N log N) transforms or GEMM-based diagonalisation, chosen by
  timing both; multigrid-PCG when obstacles make the operator non-separable.
* Tensor Cores: used only where they keep accuracy (FP64 MMA on data-centre
  GPUs for the dense transforms). TF32 would degrade the pressure solve, so it is
  not enabled. RT cores/OptiX are not used: the radiation model is P-1 (a
  diffusion equation, solved like the pressure); ray-traced models (Monte-Carlo
  / S2S view factors) are where RT cores would pay off — see roadmap.
* Disk: asynchronous checkpoints with a bounded queue (the solver never waits on
  I/O unless the disk is slower than the checkpoint rate).

## Validation (measured in this repository's test suite)

| Case | Result | Reference |
|---|---|---|
| Taylor–Green vortex, error vs exact solution, n = 16→128 | 1.1e-3, 2.8e-4, 7.2e-5, 1.8e-5 (2nd order) | exact |
| Lid-driven cavity Re=100, 64², MAC | max \|Δu\| = 0.004, max \|Δv\| = 0.0085 | Ghia et al. 1982 |
| Lid-driven cavity Re=100, 48², SIMPLEC on Fluent-format quads | max \|Δu\| = 0.0044 | Ghia et al. 1982 |
| Same, skewed triangles (60° non-orthogonality) | max \|Δu\| = 0.0067 | Ghia et al. 1982 |
| Lid-driven cavity **Re=1000**, 128², SIMPLEC | max \|Δu\| = 0.005, max \|Δv\| = 0.010 (3 546 iterations, 354 s on 4 CPU cores) | Ghia et al. 1982 |
| Natural convection Ra=1e3 via SIMPLEC + Boussinesq | Nu ≈ 1.120 | 1.118 |
| Natural convection Ra=1e4, 64² | Nu = 2.251, u_max = 16.19, v_max = 19.64 | 2.243, 16.18, 19.62 (de Vahl Davis 1983) |
| Natural convection Ra=1e5, 64² (FMG) | Nu = 4.569 | 4.519 |
| Channel Re=50 | outlet profile within 0.006 of Poiseuille, dp/dx = 0.238 (0.24) | analytic |
| Projection | max \|div u\| ≈ 1e-14 (fast solver), 1e-11 (multigrid) | 0 |
| SIMPLEC channel / exported step mesh | net mass imbalance ≈ 1e-12 | 0 |
| **k-ω SST** turbulent channel Re_τ = 395 (periodic, y+ ≈ 0.25) | U_b⁺ = 17.62; u⁺(y⁺=100) = 16.9; wall shear = driving force within 0.03% | DNS U_b⁺ = 17.54 (Moser, Kim & Mansour 1999); log law 16.4 |
| **k-ω SST** flat plate Re_L = 5e6 | Cf(x=0.97) = 0.00264 (fine and coarse grid); Cf(1.5) = 0.00248; C_D = 0.00279 | NASA TMR SST 0.00271; Schlichting 0.00250; NASA C_D ≈ 0.00286 |
| **k-ω SST** in the log-law regime (backward-facing step Re_h = 2e4, y+ 20–80) | stable, reattachment x/h ≈ 5.7 on 8 cells/h | — |
| **P-1** radiating slab, τ = 0.1 / 1 / 5 | wall flux error 2e-10 / 4e-8 / 9e-7 | exact P-1: σ(T₁⁴−T₂⁴)/(1+3τ/4) |
| Periodic Poiseuille (unstructured triangles) | profile error 0.08% | analytic |
| Transient Taylor–Green on a periodic unstructured mesh (BDF2) | error 1.2e-2 → 1.5e-3 (n = 16 → 32) | exact |
| Species: two-stream mixing layer | max \|ΔY\| = 0.0047 at x = 1; species out/in = 1 | erf solution |

Run `python -m pytest` (88 tests) or `python -m flowtrace validate`.

**Steady vs transient:** for steady problems use *Steady SIMPLEC* (works for the
built-in cases too): Re=1000 converges in a few thousand iterations, whereas
time-marching has to integrate tens of convective time units. Use the
transient MAC projection for unsteady physics (vortex shedding, convection
rolls) — it is exact in continuity and GPU-friendly.

**GPU code paths** are tested without a GPU through `tests/fake_gpu.py`, a
NumPy-backed stand-in for CuPy (including CuPy's DCT/DST type limits), which
checks that every GPU-specific branch gives the same results as the CPU path.

**Windows:** double-click `start_flowtrace.bat` (installs on first run, then opens
the dashboard). Sample Fluent meshes are in `examples/`.

## Honest limits / roadmap

* The structured (MAC) solver is 2-D and laminar; turbulence, radiation,
  periodicity and transient pressure-based runs live in the unstructured solver
  (which also runs every built-in case via *Steady SIMPLEC*).
* Heat transfer with SST uses laminar conduction at the wall: resolve the wall
  (y+ ≈ 1) for accurate heat flux; momentum is y+-insensitive.
* Only translational periodicity (no rotational periodic zones yet).
* Not yet: LES/DES, compressible (density-based) flow, multiphase (VOF),
  discrete-ordinates or Monte-Carlo radiation (where OptiX/RT cores apply),
  moving/deforming meshes, reacting flows (species are passive).
* The CuPy GPU path uses the same verified array code as the CPU path, but it
  was developed in a container without a GPU: run `python -m flowtrace bench`
  and the planner on your laptop — it only selects the GPU if it measures
  faster, and falls back automatically if anything fails.
* PyFluent API names change between releases; the bridge falls back to TUI
  commands, and the journal uses long-standing TUI commands.

---

<div dir="rtl">

## فارسی

**FlowTrace** یک حلگر CFD ناهمگن و آگاه از سخت‌افزار است که آماده‌ی اتصال به **Ansys Fluent** است.

### چه چیزی ساخته شده

- **شناخت سیستم هدف:** پردازنده (مدل، هسته‌ها، دستورهای SIMD)، RAM، دیسک، باتری، کارت گرافیک NVIDIA (VRAM، Compute Capability، Tensor/RT Core، نسبت FP64 به FP32) و کتابخانه‌ها را شناسایی می‌کند؛ پهنای باند حافظه، GEMM، DCT، استنسیل و تأخیر PCIe را اندازه می‌گیرد.
- **برنامه‌ریز اجرا:** برای هر بک‌اند موجود، خودِ حلگر را روی خودِ مسئله می‌سازد و چند گام واقعی را **زمان‌گیری** می‌کند؛ سریع‌ترین را انتخاب می‌کند و دلیل جای‌گذاری هر مرحله را نشان می‌دهد (فرض «GPU همیشه سریع‌تر است» وجود ندارد).
- **بودجه‌ی حافظه:** پیش از تخصیص، حافظه‌ی لازم را تخمین می‌زند و اگر جا نشود پیشنهاد می‌دهد (FP32، CPU، شبکه‌ی کوچک‌تر).
- **حلگر سازمان‌یافته (MAC):** ناویه–استوکس تراکم‌ناپذیر دوبعدی + انرژی بوزینسک؛ QUICK/TVD، آدامز–بشفورث ۲، کرانک–نیکلسون، تصویر فشار دقیق، دیواره‌ی متحرک، ورودی/خروجی، تقارن، تناوبی، مانع غوطه‌ور، و FMG برای حالت پایا.
- **ریاضیات سریع:** حلگر مستقیم پواسون/هلمهولتز با تبدیل‌های DCT/DST/FFT یا قطری‌سازی متراکم به‌صورت GEMM، چندشبکه‌ای هندسی + CG، AMG تجمیعی هموارشده.
- **حلگر بی‌سازمان (مانند Fluent):** الگوریتم فشار-مبنا SIMPLE/SIMPLEC با Rhie–Chow، بالادست مرتبه‌دوم محدودشده، تصحیح غیرعمودی، گرادیان کمترین مربعات، AMG برای تصحیح فشار، معادله‌ی انرژی و باقیمانده‌های مقیاس‌شده‌ی Fluent؛ دوبعدی و سه‌بعدی با هر نوع سلول؛ **پایا یا گذرا** (Euler/BDF2 با تکرار داخلی).
- **مدل آشفتگی k-ω SST** (منتر ۲۰۰۳، همان فرم Fluent) با تیمار دیواره‌ی غیرحساس به y⁺ (قانون دیواره‌ی اسپالدینگ)، شدت آشفتگی و نسبت ویسکوزیته در ورودی، و عدد پرانتل آشفته برای انتقال حرارت.
- **مدل تابش P-1** با ضریب صدور دیواره، **مرزهای تناوبی** با گرادیان فشار محرک، **انتقال گونه‌ها** (غیرواکنشی)، و **گزارش‌ها:** نیرو، Cd/Cl، y⁺، ضریب اصطکاک و شار حرارتی هر دیواره.
- **اتصال به Fluent:** خواندن و نوشتن `.msh` (ASCII، باینری، gzip، `.cas` قدیمی)، فایل درون‌یابی `.ip` برای شروع گرم، `.prof`، ژورنال TUI و اسکریپت PyFluent.
- **رابط کاربری حرفه‌ای:** داشبورد وب محلی (بدون اینترنت): کانتور، خطوط جریان، بردار، باقیمانده‌ها، کارایی، پایشگرها، طرح اجرا، اعتبارسنجی و خروجی Fluent.

### نصب و اجرا

</div>

```bash
pip install -e .
pip install numba psutil        # پیشنهادی: هسته‌های JIT چندهسته‌ای
pip install cupy-cuda12x        # برای کارت گرافیک NVIDIA
python -m flowtrace             # باز شدن داشبورد در http://127.0.0.1:8765
```

<div dir="rtl">

### گردش کار با Fluent

۱. در FlowTrace حل کنید و از تب **Fluent** بسته‌ی خروجی را بگیرید.
۲. در Fluent فایل `flowtrace.msh` (یا مش خودتان با همان هندسه) را باز کنید، مدل‌ها و شرایط مرزی را تنظیم و Initialize کنید، سپس با *File › Interpolate › Read and Interpolate* فایل `flowtrace.ip` را بخوانید.
۳. Fluent به‌جای شروع از سکون، از میدان همگراشده‌ی FlowTrace ادامه می‌دهد.

همچنین می‌توانید هر مش `.msh` فلوئنت را مستقیم در داشبورد بکشید و رها کنید و با همان الگوریتم فشار-مبنای Fluent حل کنید.

### محدودیت‌های فعلی (صادقانه)

- حلگر سازمان‌یافته (MAC) دوبعدی و آرام است؛ آشفتگی، تابش، تناوب و حل گذرای فشار-مبنا در حلگر بی‌سازمان هستند.
- اعتبارسنجی SST: کانال آشفته Re_τ=395 با خطای ۰٫۵٪ نسبت به DNS و صفحه‌ی تخت با خطای ۲٫۸٪ نسبت به نتایج SST ناسا.
- هنوز پیاده نشده: LES/DES، جریان تراکم‌پذیر، چندفازی (VOF)، تابش DO یا مونت‌کارلو (جایی که OptiX/RT Core معنا پیدا می‌کند)، مش متحرک، احتراق و واکنش شیمیایی.
- مسیر GPU با CuPy همان کد آرایه‌ای تأییدشده‌ی CPU است اما در محیط توسعه GPU وجود نداشت؛ روی لپ‌تاپ خودتان `python -m flowtrace bench` را اجرا کنید — برنامه‌ریز فقط وقتی GPU را انتخاب می‌کند که واقعاً سریع‌تر اندازه‌گیری شود.

</div>
>>>>>>> 3d528151e15014be7c185d9c110063ec4634bd8d
