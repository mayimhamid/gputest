<div align="center">

# FlowTrace

**A verified, hardware-aware CFD solver and warm-start companion for Ansys Fluent**

![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Backends](https://img.shields.io/badge/backends-NumPy%20%7C%20Numba%20%7C%20CuPy%2FCUDA-blue)
![Tests](https://img.shields.io/badge/tests-207%20passing-brightgreen)
![Benchmarks](https://img.shields.io/badge/validation-68%2F70%20checks%20pass-yellowgreen)

[English](README.md) · [فارسی](README.fa.md) · [Documentation](docs/index.md) · [Validation report](docs/validation/index.md) · [Changelog](CHANGELOG.md) · [Release readiness](RELEASE_READINESS.md)

</div>

FlowTrace solves incompressible flow (steady and transient), heat transfer, RANS turbulence and,
with a density-based solver, compressible flow. It measures the CPU and GPU of the machine it
runs on and places each stage of the solve where the measurement says it is fastest. It also
reads and writes the files Ansys Fluent uses.

Its main job is to **compute a good initial field quickly**, which Fluent then interpolates onto
its own mesh and continues from (a *warm start*). Everything it claims is backed by a script in
this repository.

> Ansys and Fluent are registered trademarks of ANSYS, Inc. FlowTrace is an independent
> open-source project. It is compatible with Fluent's file formats, but it is not affiliated with
> or endorsed by ANSYS, Inc.

## Highlights

* **Verified, not only validated.** Manufactured solutions on warped meshes show second-order
  convergence for velocity and temperature on quads and triangles. They also exposed a real
  pressure-force bug that the cavity validation had missed. Every SIMPLE/SIMPLEC run also
  reports how far each linear system and the continuity constraint actually got
  ([solver diagnostics](docs/theory/diagnostics.md)).
* **Checked against the classic references.** These include Ghia's cavity, de Vahl Davis's
  natural convection and the Moser–Kim–Mansour channel DNS. They also cover flat-plate
  skin-friction correlations and the Driver & Seegmiller step. **68 of 70 acceptance checks pass
  in the committed full run.** The two failures are listed below, not hidden. That run predates
  this release's fixes; the re-run on the release code is in progress and will replace the
  numbers below.
* **A companion for Fluent.** FlowTrace writes Fluent's own `.ip` interpolation file, plus a
  journal and a PyFluent script that open *your* case. In the proxy study the FlowTrace field
  cut the iterations to the converged solution by ×1.4 to ×2.4 in four of five cases. In the
  fifth, a cylinder that the two meshes resolve differently, it cost iterations; a
  pre-registered test confirmed that cause. On these small 2-D cases the pre-solve costs more
  time than the iterations save.
* **Measured, not assumed, hardware placement and precision.** The planner times real solver
  steps and records why it chose each device and precision. FlowTrace runs FP64 unless FP32 is
  shown to reach the case's own tolerance; FP32 failed that test in five of six steady cases.
  On a laptop RTX 4060 the structured solver's 1024² step ran about 30× faster in FP32 on the
  GPU than in FP64 on the 16-thread CPU. That is not an equal-accuracy comparison; the GPU's
  FP64 time was not measured. The unstructured solver gains from the GPU only above about 16k
  cells, so small cases stay on the CPU.
* **Fluent file formats, checked against Fluent's own files.** `.msh`/`.cas`, CFF
  `.cas.h5`/`.msh.h5`/`.dat.h5`, `.ip` and `.prof`. The readers pass all seven checks on a
  Fluent-written 3-D polyhedral case (the PyFluent mixing elbow), and the writers reproduce
  Fluent's layout face by face. Fluent itself has not yet read a file FlowTrace wrote (no
  licence).
* **Offline dashboard** in English and Persian, with live residuals, the execution plan and
  exportable run reports. It answers only its own pages: requests from other web pages are
  refused.

## Quick start

**Windows:** double-click `start_flowtrace.bat`. It creates a private `.venv`, installs
FlowTrace, adds the CUDA backend if an NVIDIA GPU is present, and opens the dashboard.

**Any platform** (Python 3.9–3.13; prebuilt CuPy/Numba wheels exist for 3.10–3.13):

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
flowtrace benchmark --tier quick          # a few minutes;  --tier full: everything below
```

An input error (missing or unreadable file, misspelt zone, unavailable backend, not enough
memory) is reported in one line with exit status 2; a run that diverges says why, in one line,
and exits with 1 ([command line](docs/user-guide/cli.md#exit-status-and-errors)).

## Validation at a glance

The numbers below come from the committed `flowtrace benchmark --tier full` run (CPU, FP64). It
was run on 26 October 2026 with FlowTrace 0.2.0, on a laptop (AMD Ryzen 9 7940HS, 8 cores/16
threads, 15 GB RAM, Windows 11, Python 3.13), and took 37 minutes. **It predates this release's
changes**, among them the k-ε wall-function constants. The re-run on the release code is in
progress. The full table, with all 70 acceptance checks, plots and the cases not shown here,
is in the [validation report](docs/validation/index.md). The raw results are in
[`benchmarks/results/`](benchmarks/results/).

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

**The two failures.**

* **Realizable k-ε flat plate (−7.0 % against a 6 % criterion).** The cause was found in this
  release. The wall function paired κ = 0.41 with Fluent's E = 9.793, which matches neither
  Fluent nor the classical log law. With Fluent's κ = 0.4187 the same case gives −4.96 %, and
  the channel improves as well (standard k-ε 3.5 → 1.9 %, realizable 7.6 → 5.8 % against the
  DNS). The result still depends on y⁺; see the [turbulence theory](docs/theory/turbulence.md).
* **Cylinder warm start (×0.65).** The pre-solve and the target mesh resolve the cylinder
  differently (12 and 9 cells per diameter), so their converged wakes differ by about a
  quarter. A pre-registered test confirmed it: with a target that resolves the cylinder like
  the pre-solve, the warm start needs half the iterations (558 → 274, ×2.04).

Criteria were fixed before the runs and were not relaxed afterwards. Both failures are analysed
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

## Warm start: what does it save?

FlowTrace's structured solver pre-solves each case and writes a Fluent `.ip` file. The file is
interpolated onto a *different* mesh, and FlowTrace's SIMPLEC solver then stands in for Fluent's
pressure-based solver. Both runs are counted until the velocity (and temperature) field is
within 10⁻⁴ of the converged solution, and timed on equal terms: solver steps only, with the
pre-solve and the interpolation charged to the warm start.

| Case | Target cells | Cold start | Warm start | Saved | Pre-solve |
|---|---:|---:|---:|---:|---:|
| Lid-driven cavity, Re = 400 | 3,200 | 1,162 | 488 | ×2.4 | 8.2 s |
| Developing channel, Re = 100 | 1,152 | 68 | 47 | ×1.4 | 0.4 s |
| Backward-facing step, Re = 200 | 5,616 | 772 | 418 | ×1.8 | 224 s |
| Cylinder in a channel, Re = 40 | 10,308 | 700 | 1,085 | **×0.65** | 332 s |
| Heated cavity, Ra = 10⁵ | 3,200 | 316 | 158 | ×2.0 | 4.0 s |

How to read this:

* **The cylinder gets slower, and the reason is the mesh pair.** The pre-solve resolves the
  cylinder with 12 cells per diameter, the target mesh with 9. Their converged wakes differ
  (1.87 D vs 2.33 D of recirculation), so the interpolated field must be corrected, and that
  takes longer than starting from rest. With a target that resolves the cylinder at 12 cells
  per diameter too, the warm start saved ×2.04 ([study page](docs/validation/warmstart.md)).
* **The savings are in iterations, not in wall time.** On these small 2-D cases the pre-solve
  takes longer than the SIMPLEC run it shortens (step: 224 s against 8 s for the whole cold
  run). The warm-start study now times both paths on equal terms: solver steps only, with the
  pre-solve and the interpolation charged to the warm start. Its numbers come with the re-run.
  The case for a warm start is a large Fluent mesh, where one iteration costs far more; that
  has not been measured here.
* **The Fluent column has not been run** (no licence). `flowtrace warmstart-study` writes the
  cold/warm journals and runs them through PyFluent when Fluent is available.

## Hardware

Measured with FlowTrace 0.2.0 on a laptop with an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). Its
FP64 rate is 1/64 of FP32. In FP64, GPU results agree with the CPU to within 7×10⁻¹⁴
([GPU log](docs/validation/gpu.md)).

**Precision.** This table predates the current precision rule. Its GPU column is FP32 and its
CPU columns are FP64, so it does not compare equal accuracy. FlowTrace now runs FP64 on every
device and chooses FP32 only for a steady case whose FP32 round-off bound is within its
tolerance. `flowtrace gpu-report` now times every backend in FP64 and the GPU in FP32 as well,
listed separately. No GPU was available to re-measure this table for the current release.

**Structured (MAC) solver**, lid-driven cavity Re = 1000, ms per time step
(`flowtrace gpu-report`):

| Grid | NumPy (FP64) | Numba, 16 threads (FP64) | CuPy fused kernels (FP32) | Planner picked (then) |
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
overhead exceeded the study's own 15 % budget at 2 of 6 sizes. A rule added since (do not try a
GPU that cannot recover its measured set-up time) targets one of them but has not been
measured on a GPU yet.

**Memory.** Before a run, FlowTrace estimates the memory the case needs; on the CPU the
estimate is an upper bound ([measurements](benchmarks/results/memory_estimate.md)). A case
that does not fit on the GPU runs on the CPU; one that fits nowhere is refused, before it
allocates, with advice (`FLOWTRACE_MEMORY_CHECK=off` to try anyway).

Run `flowtrace gpu-report` on your own machine: these numbers do not transfer to other hardware.

## What is inside

| Area | Capabilities |
|---|---|
| Structured solver | 2-D incompressible Navier–Stokes + Boussinesq energy on a MAC grid, exact pressure projection (DCT/DST/FFT or GEMM, multigrid with obstacles), QUICK/TVD, AB2 / Crank–Nicolson, FMG |
| Unstructured solver | Fluent-style pressure-based SIMPLE/SIMPLEC on any cell shape (2-D/3-D), Rhie–Chow, Green–Gauss face-pressure force, AMG-PCG pressure correction; steady or transient (Euler/BDF2, Choi correction); P-1 radiation; passive species transport; per-iteration [algebraic diagnostics](docs/theory/diagnostics.md) |
| Compressible solver | Density-based: HLLC + MUSCL (Venkatakrishnan/Barth), matrix-free implicit steady or SSP-RK3 transient, Euler / Navier–Stokes / k-ω SST, 2-D, 3-D polyhedral and axisymmetric, Fluent-style pressure inlet/outlet and far-field; native C++/CUDA flux kernels ([theory and verification](docs/theory/compressible.md)) |
| Turbulence | k-ω SST (Menter 2003, y⁺-insensitive wall treatment), realizable and standard k-ε with scalable wall functions (Fluent's κ = 0.4187, E = 9.793) |
| Zones | Conjugate heat transfer (solid zones, volumetric sources), moving reference frames, rotating walls, translational periodicity |
| Fluent files | `.msh`/`.cas` (ASCII, binary, gzip), CFF `.cas.h5`/`.msh.h5`/`.dat.h5`, `.ip`, `.prof`, journals, PyFluent scripts; readers checked against Fluent-written files |
| Mesh quality | Orthogonal quality, equiangle skewness (standard element shapes), aspect ratio, non-orthogonality, volume ratio (Fluent-style report) |
| Hardware | NumPy, Numba JIT kernels, CuPy + fused CUDA C++ kernels (NVRTC); C++/OpenMP kernels for the unstructured solver (g++, built on first use) and their CUDA counterparts (`.cu`); measured placement; FP64 unless FP32 is shown to meet the case's tolerance; memory checked before allocating; `flowtrace batch` runs many cases on all cores and the GPU at once |
| Dashboard | Offline web UI: live residuals and fields, execution plan, saved projects, HTML run reports, English/Persian (RTL), light/dark, keyboard and screen-reader friendly; loopback only, refuses requests from other web pages |
| V&V tooling | `flowtrace benchmark` (JSON + HTML/PDF/Markdown), MMS, GCI, warm-start study, GPU log, Fluent-file checks |

## Documentation

* [Installation](docs/install.md) · [Command line](docs/user-guide/cli.md) · [Dashboard](docs/user-guide/dashboard.md) · [Python API](docs/user-guide/python-api.md)
* [Ansys Fluent workflow](docs/user-guide/fluent.md) · [GPU and the planner](docs/user-guide/gpu.md) · [Native kernels (C++/CUDA)](docs/user-guide/native.md)
* Theory: [MAC solver](docs/theory/mac.md) · [SIMPLEC](docs/theory/simplec.md) · [solver diagnostics](docs/theory/diagnostics.md) · [compressible](docs/theory/compressible.md) · [turbulence](docs/theory/turbulence.md) · [CHT, MRF, periodic](docs/theory/zones.md) · [verification](docs/theory/verification.md)
* [Tutorials (Jupyter)](docs/tutorials/) · [Architecture](ARCHITECTURE.md) · [Release readiness](RELEASE_READINESS.md)

Build the site locally with `pip install -e ".[docs]" && mkdocs serve`.

## Limitations

* The compressible solver is verified on canonical cases (Sod, oblique shock, flat plate) but **not yet compared with Fluent** (FFF-7) or run on the full rocket mesh; its CUDA kernel has not run on a GPU yet. See [what is not validated](docs/theory/compressible.md#not-yet-validated).
* RANS only (steady or unsteady). No LES/DES, multiphase, combustion or moving/deforming meshes; radiation is P-1 only.
* The structured solver is 2-D. The unstructured solver is 3-D capable, but it is written in
  vectorised Python, so very large 3-D cases run slower than in compiled commercial codes.
* The Fluent readers are checked against Fluent-written files, but **no file written by
  FlowTrace has been read back into Fluent yet**. 3-D files written by FlowTrace 0.3.0 or
  earlier have every face reversed for Fluent: re-export them.
* **The Fluent column of the warm-start study has not been run** (no licence was available).
  On the small 2-D proxy cases the warm start saved iterations but not wall time.
* The pressure-correction equation has no non-orthogonal correctors. On strongly skewed meshes
  the pressure under-relaxation must stay below about cos²θ; the automatic setting handles it
  ([measured envelope](docs/theory/simplec.md#skewed-meshes-measured-stability-envelope)).
* This release was tested without a GPU, without Fluent and on Linux only; everything that
  could not be checked is listed in [RELEASE_READINESS.md](RELEASE_READINESS.md).

## Contributing, citing, licence

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). To cite FlowTrace, use [CITATION.cff](CITATION.cff); a
JOSS-style paper draft is in [`paper/`](paper/paper.md). FlowTrace is released under the
[MIT licence](LICENSE).
