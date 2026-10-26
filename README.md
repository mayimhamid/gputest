<div align="center">

# FlowTrace

**A verified, hardware-aware CFD solver and warm-start companion for Ansys Fluent**

![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Backends](https://img.shields.io/badge/backends-NumPy%20%7C%20Numba%20%7C%20CuPy%2FCUDA-blue)
![Tests](https://img.shields.io/badge/tests-150%20passing-brightgreen)
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
| Structured solver | 2-D incompressible Navier–Stokes + Boussinesq energy on a MAC grid, exact pressure projection (DCT/DST/FFT or GEMM, multigrid with obstacles), QUICK/TVD, AB2 / Crank–Nicolson, FMG |
| Unstructured solver | Fluent-style pressure-based SIMPLE/SIMPLEC on any cell shape (2-D/3-D), Rhie–Chow, Green–Gauss face-pressure force, AMG-PCG pressure correction |
| Turbulence | k-ω SST (Menter 2003, y⁺-insensitive wall treatment), realizable and standard k-ε with scalable wall functions |
| Zones | Conjugate heat transfer (solid zones, volumetric sources), moving reference frames, rotating walls, translational periodicity |
| Fluent files | `.msh`/`.cas` (ASCII, binary, gzip), CFF `.cas.h5`/`.msh.h5`/`.dat.h5`, `.ip`, `.prof`, journals, PyFluent scripts |
| Mesh quality | Orthogonal quality, equiangle skewness, aspect ratio, non-orthogonality, volume ratio (Fluent-style report) |
| Hardware | NumPy, Numba JIT kernels, CuPy + fused CUDA C++ kernels (NVRTC); measured placement; FP32 on consumer GPUs |
| Dashboard | Offline web UI: live residuals and fields, execution plan, saved projects, HTML run reports, English/Persian (RTL), light/dark, keyboard and screen-reader friendly |
| V&V tooling | `flowtrace benchmark` (JSON + HTML/PDF/Markdown), MMS, GCI, warm-start study, GPU log |

## Documentation

* [Installation](docs/install.md) · [Command line](docs/user-guide/cli.md) · [Dashboard](docs/user-guide/dashboard.md) · [Python API](docs/user-guide/python-api.md)
* [Ansys Fluent workflow](docs/user-guide/fluent.md) · [GPU and the planner](docs/user-guide/gpu.md)
* Theory: [MAC solver](docs/theory/mac.md) · [SIMPLEC](docs/theory/simplec.md) · [turbulence](docs/theory/turbulence.md) · [CHT, MRF, periodic](docs/theory/zones.md) · [verification](docs/theory/verification.md)
* [Tutorials (Jupyter)](docs/tutorials/) · [Compressible-flow design proposal](docs/design/compressible-flow.md)

Build the site locally with `pip install -e ".[docs]" && mkdocs serve`.

## Limitations

* Incompressible (Boussinesq) flow only. Compressible flow is a [design proposal](docs/design/compressible-flow.md).
* Steady RANS. No LES/DES, multiphase, combustion, radiation or moving/deforming meshes.
* The structured solver is 2-D. The unstructured solver is 3-D capable, but it is written in
  vectorised Python, so very large 3-D cases run slower than in compiled commercial codes.
* The `.cas.h5`/`.dat.h5` writer follows the public VTK reader's layout and has been round-trip
  tested, **but not yet read back into Fluent**. The verified hand-off is `.msh` + `.ip`.
* **The Fluent column of the warm-start study has not been run** (no licence was available).
  It holds labelled placeholders until someone with a licence runs `flowtrace warmstart-study`.

## Contributing, citing, licence

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). To cite FlowTrace, use [CITATION.cff](CITATION.cff); a
JOSS-style paper draft is in [`paper/`](paper/paper.md). FlowTrace is released under the
[MIT licence](LICENSE).
