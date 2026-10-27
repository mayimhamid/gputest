# FlowTrace: System Architecture

This document describes every component in the repository as it is in the release candidate of
October 2026: about 18.4k lines of Python, C++/CUDA and web code in `flowtrace/`. It covers what
each part does, how the parts depend on each other, and how data moves through a run. It is a map
of the code, not a user guide; the user docs are in `docs/`. Line counts in the tables are from
this tree.

Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Context diagram](#2-context-diagram)
3. [Layered architecture](#3-layered-architecture)
4. [Module reference](#4-module-reference)
5. [Key flows](#5-key-flows)
6. [Numerical methods](#6-numerical-methods)
7. [Hardware placement and the planner](#7-hardware-placement-and-the-planner)
8. [Data formats and persistence](#8-data-formats-and-persistence)
9. [Interfaces](#9-interfaces)
10. [Validation and verification](#10-validation-and-verification)
11. [Build, packaging, CI and documentation](#11-build-packaging-ci-and-documentation)
12. [Cross-cutting concerns](#12-cross-cutting-concerns)
13. [Known limitations](#13-known-limitations)
14. [Repository map](#14-repository-map)

---

## 1. Purpose and scope

FlowTrace solves incompressible flow (steady and transient), heat transfer with conjugate solids and P-1
radiation, passive species, RANS turbulence, and compressible flow with a density-based solver. It has
three jobs:

* **Solve** small and medium CFD cases on whatever CPU/GPU the machine has. Each stage of the solve is
  placed on the device that a timing measurement says is fastest.
* **Exchange** data with Ansys Fluent: read Fluent meshes (`.msh`, `.cas`, `.cas.h5`), write
  Fluent interpolation files (`.ip`), profiles, journals and PyFluent scripts.
* **Warm-start** Fluent. Pre-solve the case with FlowTrace, hand the field to Fluent, and measure how
  many iterations the warm start saves (the warm-start study).

Non-goals stated in the code and docs: FlowTrace is not a Fluent replacement and not a parallel
distributed solver (single node only). Its Fluent readers are checked against files written by Fluent
(`validation/fluent_files.py`); Fluent reading files written by FlowTrace has not been checked (that
needs a licence).

---

## 2. Context diagram

```mermaid
flowchart LR
    User([Engineer / researcher])
    Browser[Dashboard in browser<br/>flowtrace/ui/static]
    CLI[flowtrace CLI<br/>flowtrace/cli.py]
    Fluent[(Ansys Fluent<br/>licensed, external)]
    PyFluent[PyFluent<br/>optional]
    Files[(Case files on disk<br/>.msh .cas.h5 .dat.h5 .ip)]
    Disk[(Work directory<br/>projects, reports,<br/>hardware cache)]
    FT[FlowTrace core<br/>solvers, planner, validation]
    GPU[(NVIDIA GPU<br/>CuPy / CUDA)]

    User --> Browser
    User --> CLI
    Browser -- HTTP + SSE<br/>127.0.0.1:8765 --> FT
    CLI --> FT
    FT <--> Files
    FT <--> Disk
    FT -. optional .-> GPU
    FT -- writes .ip, .jou, .py --> Fluent
    Fluent -- reads .ip, .jou --> Fluent
    FT -. optional live link .-> PyFluent --> Fluent
```

FlowTrace never runs inside Fluent. The handoff is file based (the `.ip` file and a journal), and the
optional live link goes through PyFluent, which needs a licence.

---

## 3. Layered architecture

```mermaid
flowchart TB
    subgraph L5[Interfaces]
        CLI[cli.py]
        UI[ui/server.py + static/]
        Main[__main__.py, start_flowtrace.bat]
    end
    subgraph L4[Orchestration]
        Engine[runtime/engine.py<br/>Job, EventHub]
        Ckpt[runtime/checkpoint.py]
        WS[fluent/warmstart.py]
        Study[fluent/warmstart_study.py]
        Suite[validation/suite.py<br/>benchmarks.py]
        Batch[runtime/batch.py<br/>run_batch]
    end
    subgraph L3[Planning and placement]
        Planner[hardware/planner.py]
        Probe[hardware/probe.py]
        Bench[hardware/bench.py]
        Mem[hardware/memory.py]
        GPUR[hardware/gpu_report.py]
    end
    subgraph L2[Solvers]
        MAC[structured/solver.py<br/>MACSolver]
        SIMPLE[unstructured/simple.py<br/>SimpleSolver]
        Turb[unstructured/turbulence.py]
        CHT[unstructured/conjugate.py]
        Diag[unstructured/diagnostics.py]
        Comp[compressible/solver.py<br/>CompressibleSolver]
        Seq[structured/sequencing.py]
        Cases[structured/case.py<br/>unstructured/case.py]
    end
    subgraph L1[Kernels and linear algebra]
        Backend[hardware/backend.py]
        Numba[structured/kernels.py<br/>Numba]
        CUDA[structured/cuda_kernels.py<br/>NVRTC]
        LA[linalg/*<br/>separable, multigrid, krylov, amg]
        Native[native/*<br/>C++/OpenMP and CUDA kernels]
    end
    subgraph L0[Data model and I/O]
        Mesh[mesh/*<br/>UnstructuredMesh, Fluent readers and writers]
        FVM[unstructured/fvmesh.py]
        FluentIO[fluent/export.py, convert.py,<br/>interpolate.py, bridge.py]
        H5[mesh/fluent_h5.py]
    end

    CLI --> Engine
    CLI --> WS
    CLI --> Study
    CLI --> Suite
    CLI --> Batch
    Batch --> SIMPLE
    Batch --> Comp
    UI --> Engine
    Engine --> Planner
    Engine --> Ckpt
    Engine --> MAC
    Engine --> SIMPLE
    Engine --> Seq
    Engine --> Diag
    Diag --> SIMPLE
    WS --> Planner
    WS --> SIMPLE
    Study --> MAC
    Study --> SIMPLE
    Planner --> Probe
    Planner --> Bench
    Planner --> Mem
    Probe --> Backend
    Bench --> Backend
    MAC --> Cases
    SIMPLE --> Cases
    SIMPLE --> Turb
    SIMPLE --> CHT
    SIMPLE --> FVM
    MAC --> LA
    SIMPLE --> LA
    MAC --> Backend
    SIMPLE --> Backend
    MAC --> Numba
    MAC --> CUDA
    SIMPLE --> Native
    Comp --> Native
    Comp --> FVM
    Comp --> Backend
    FVM --> Mesh
    FluentIO --> Mesh
    FluentIO --> H5
    Suite --> MAC
    Suite --> SIMPLE
```

Dependencies point downward. The only upward references are the interface layer calling into
orchestration, and a few lazy imports inside functions (for example `engine.py` imports the solvers
inside methods). Those lazy imports keep `flowtrace --help` and the probe fast, because they avoid
loading NumPy-heavy solver modules and Numba until they are needed.

---

## 4. Module reference

Line counts are from the current tree. "Public" means it is used from outside its own module.

### 4.1 Package root and entry points

| File | Lines | Role |
|---|---:|---|
| `flowtrace/__init__.py` | 19 | Package docstring. Sets `NUMBA_NUM_THREADS`-related env vars before Numba loads: Numba's OpenMP workers busy-wait, which starves NumPy/SciPy calls (the code records a 10× slowdown). |
| `flowtrace/__main__.py` | 3 | `python -m flowtrace` entry, forwards to `cli.main`. |
| `flowtrace/cli.py` | 633 | argparse-based CLI. Subcommands: `ui`, `probe`, `bench`, `run`, `mesh`, `quality`, `h5-inspect`, `solve`, `compressible`, `case`, `batch`, `fluent-warmstart`, `validate`, `benchmark`, `warmstart-study`, `gpu-report`, `parse-transcript`. Installed as the `flowtrace` console script. Default command is `ui`; a non-loopback `ui --host` needs `--allow-remote`. `main()` turns user errors (`OSError`, `ValueError`, `MemoryError`, `BackendUnavailable`) into one line on stderr and exit status 2 (`FLOWTRACE_DEBUG=1` re-raises), Ctrl-C into 130. |
| `start_flowtrace.bat` | - | Windows one-click launcher. Creates `.venv`, picks Python 3.13 to 3.10, installs the extras, adds CuPy when `nvidia-smi` finds a GPU, reinstalls only when the version or GPU situation changes, then starts the dashboard. |

### 4.2 `hardware/`: discover, measure, place

| File | Lines | Role |
|---|---:|---|
| `backend.py` | 140 | Array backend abstraction. `Backend(name, device_index)` for `numpy`, `numba` (CPU JIT) and `cupy` (GPU). Exposes `asarray`, `to_numpy`, `scalar`, `synchronize`, `timer`, `mem_info`, `free_pools`, `supports_transform` (which FFT/DCT kinds exist on the device). `get_backend()` and `available_backends()` are the only factory functions. Raises `BackendUnavailable` when CuPy or CUDA is missing. |
| `probe.py` | 416 | Hardware discovery. Builds a `SystemProfile` (CPU model, physical cores, SIMD flags, RAM, battery state, library versions) and a list of `GPUInfo` (name, memory, compute capability, classification). GPU discovery tries CuPy first, then `nvidia-smi`. `cached_profile()` memoises the result. |
| `bench.py` | 164 | Micro-benchmarks per backend: triad (memory bandwidth), GEMM, DCT, 5-point stencil, launch latency, copy bandwidth. Results are cached on disk with a machine fingerprint, so a different machine or driver triggers a re-run. |
| `planner.py` | 548 | Execution planner. Builds the real solver for the real case, times a few real time steps after warm-up, and picks the fastest configuration. Decides the precision with a recorded reason (FP32 only for a case that passes `fp32_admissible`). Skips GPU timing that cannot pay back its measured set-up time. Produces `ExecutionPlan` (structured) and `UnstructuredPlan`, each with a list of `Placement(stage, device, method, reason)`. The UI shows these reasons. See [section 7](#7-hardware-placement-and-the-planner). |
| `memory.py` | 102 | Memory budgeting. `estimate_structured` and `estimate_unstructured` return a `MemoryEstimate` before allocation. The unstructured estimate carries a measured-overhead term and bounds the measured CPU peak (`benchmarks/memory_estimate.py`). `POST /api/estimate` reports it to the UI, and the engine logs the required and available memory when a run starts. |
| `gpu_report.py` | 215 | GPU validation log (`flowtrace gpu-report`). Checks that the CuPy path matches the CPU answer and records when the GPU is faster. Times every backend in FP64 and the GPU also in FP32, listed separately. Writes Markdown. |

### 4.3 `structured/`: Cartesian MAC solver

| File | Lines | Role |
|---|---:|---|
| `case.py` | 282 | `StructuredCase` description: domain, grid, Reynolds/Rayleigh numbers, boundaries (`Side`), immersed `Obstacle`s, energy switch, turbulence flag. Presets via `make_preset`: lid-driven cavity, natural convection, developing channel, cylinder wake, backward-facing step, Rayleigh-Bénard, Taylor-Green. |
| `solver.py` | 1202 | `MACSolver`: incompressible Navier-Stokes plus Boussinesq energy on a staggered MAC grid. The largest file in the project. Handles convection, diffusion, projection, boundary ghost cells, immersed obstacles (solid masks), and monitors. Field arrays live on the selected backend. Only scalars cross to the host. `resolve_dtype` (`auto` is FP64), `fp32_round_off_bound` and `fp32_admissible` hold the precision rule. |
| `sequencing.py` | 87 | Full-multigrid-style grid sequencing for steady problems. Builds coarse cases (`coarse_case`), prolongs fields (`prolong`), and returns the factors for the FMG start-up (`fmg_factors`). |
| `kernels.py` | 178 | Fused multi-core CPU kernels for the convection and Laplacian hot loops, JIT-compiled with Numba. |
| `cuda_kernels.py` | 215 | Fused CUDA C++ kernels (`conv_u`, `conv_v`, `conv_T`, `lap`) compiled at runtime with NVRTC through CuPy. `available()` and `failure_reason()` tell the planner whether they can be used. E501 is exempted in ruff for the embedded C++ strings. |

### 4.4 `linalg/`: linear solvers (backend-agnostic)

| File | Lines | Role |
|---|---:|---|
| `separable.py` | 252 | `SeparableSolver` solves `(σ − ∇²)u = f` on tensor-product grids. Uniform grids with matching BCs use DCT/DST, which is O(N log N). Everything else, including non-uniform spacing and mixed BCs, uses a dense generalised eigendecomposition, and applying it is a GEMM. `Axis1D` describes one axis. |
| `multigrid.py` | 187 | `MGPoisson2D`: geometric multigrid used as the preconditioner for CG on cell-centred 2-D operators with immersed obstacles. |
| `krylov.py` | 121 | `pcg` and `bicgstab`, written against NumPy or CuPy arrays with callable operators and preconditioners. Returns a `KrylovInfo`; a breakdown (for example r·z underflowing to zero on a diverging outer iteration) is returned as not converged instead of raised. |
| `amg.py` | 161 | Smoothed-aggregation algebraic multigrid. Setup runs on the CPU, cycles run on any backend. `make_amg` is the factory used for the unstructured pressure equation. The `amg` extra (`pyamg`) is listed in `pyproject.toml`, but `amg.py` implements its own aggregation. |

### 4.5 `mesh/`: unstructured mesh and file formats

| File | Lines | Role |
|---|---:|---|
| `unstructured.py` | 814 | Face-based mesh with Fluent's data model: `Zone` (cell, wall, inlet, outlet, interior, periodic), `UnstructuredMesh`, `Geometry` (cell volumes, centroids, face areas and normals; `compute_geometry` repairs left-handed faces and counts them). Also generators: `cartesian_mesh_2d`, `annulus_mesh`, `extrude_to_3d`, `make_periodic`, `split_zone`, `extract_cells`, `sort_faces_by_zone`, `agglomerate_cells` (polyhedra for tests), and `reverse_face_nodes` (Fluent's 3-D face convention). |
| `fluent_msh.py` | 525 | Reader and writer for Fluent `.msh` and legacy `.cas` (ASCII or binary, optionally gzipped). 3-D faces are reversed on read and write: Fluent's right-hand normal points toward c0, FlowTrace's out of it. Any parse failure becomes `FluentMeshError` naming the file and the cause. |
| `fluent_h5.py` | 370 | CFF HDF5: `.cas.h5` and `.msh.h5` meshes, `.dat.h5` cell data. `read_any_mesh` dispatches on file type. `write_data_h5` is used by the warm start. Reading checked against Fluent-written files (`validation/fluent_files.py`); written files reproduce Fluent's node order and c0/c1 face by face; Fluent reading them is not yet checked. |
| `quality.py` | 224 | Mesh quality with Fluent's definitions: orthogonal quality, equiangle skewness (standard element shapes only; polyhedra are n/a and judged by orthogonal quality), aspect ratio. `mesh_quality` returns a `QualityReport`. Exit code 2 from `flowtrace quality` when a critical value is found. |
| `grading.py` | 37 | Node distributions for graded structured blocks: `geometric`, `growth_ratio`, `two_sided`, `first_cell_height`. Used to build wall-resolved RANS meshes. |

### 4.6 `unstructured/`: pressure-based SIMPLE/SIMPLEC solver

| File | Lines | Role |
|---|---:|---|
| `case.py` | 140 | `UnstructuredCase` and `BoundarySpec`. `default_boundaries` gives sensible defaults for zones with no explicit BC. |
| `fvmesh.py` | 307 | `FVMesh`: device-resident finite-volume data (geometry, connectivity, face maps) built from an `UnstructuredMesh`. Keeps arrays on the backend so the iteration loop does not copy them each step. `matrix_like` builds a matrix with the solver's sparsity pattern (native or SciPy/CuPy CSR). |
| `simple.py` | 1048 | `SimpleSolver`: collocated, cell-centred SIMPLE or SIMPLEC, steady or transient (implicit Euler / BDF2, inner iterations, Choi's correction). Momentum uses implicit first-order upwind plus deferred-correction second-order upwind, with explicit non-orthogonal diffusion correction and least-squares gradients. Rhie-Chow interpolation avoids checkerboarding. The pressure-correction equation is solved by AMG-preconditioned CG. Residuals are scaled the way Fluent scales them. Includes the **SIMPLEC stability guard**, which takes snapshots while the continuity residual is healthy and rolls back with a lower pressure under-relaxation when the residual blows up (`guard_exhausted_at` when it cannot go lower). Also passive species, P-1 coupling, `rotation_velocity` for MRF zones and `auto_relax_p`. Each `step()` keeps references to its systems for `diagnostics()`. |
| `diagnostics.py` | 412 | `evaluate(solver)` (behind `SimpleSolver.diagnostics()`): componentwise backward errors of the momentum and energy systems before and after the linear solve, the explicitly computed linear-solver reduction, predictor/corrected local and global continuity, field change and non-finite counts, all on the exact systems of the latest iteration. `summary()` for events and logs; `pressure_matrix_properties()` for the CG conditions. Theory: `docs/theory/diagnostics.md`. Typed with mypy. |
| `turbulence.py` | 551 | RANS models behind a common `TurbulenceModel` base: `KOmegaSST` (Menter 2003 form, as in Fluent's default SST), `KEpsilonRealizable` (Shih et al. 1995, variable C_μ), `KEpsilonStandard`. k-ε wall functions use Fluent's κ = 0.4187, E = 9.793 (`KAPPA_WALL`); SST and `spalding_uplus` use κ = 0.41. `strain_invariants` is shared. `make_turbulence` is the factory. |
| `conjugate.py` | 325 | Conjugate heat transfer. Flow is solved on fluid cells only, with fluid-solid faces as no-slip walls. Energy is solved on all cells, with conductivity at faces from the harmonic mean of the two neighbours. Solid zones have their own conductivity and optional volumetric source. `ConjugateSolver` (with `diagnostics()` for the fluid part) and `make_solver`. |
| `radiation.py` | 87 | `P1Model`: P-1 radiation with Marshak wall emissivity, coupled to energy. |
| `presets.py` | 145 | Built-in Fluent-style cases (`flowtrace case`, dashboard): SST channel, SST flat plate, P-1 slab, periodic transient Taylor-Green, mixing layer, each with its checks. |

### 4.6a `compressible/`: density-based solver

| File | Lines | Role |
|---|---:|---|
| `case.py` | 131 | `CompressibleCase`, `GasModel`, `CompBC` (pressure inlet/outlet, far-field, wall, symmetry, axis); `default_comp_boundaries` maps Fluent zone types. |
| `solver.py` | 661 | `CompressibleSolver`: HLLC flux with MUSCL (Venkatakrishnan or Barth limiter), matrix-free implicit steady solver with CFL ramping, SSP-RK3 transient, 2-D, 3-D polyhedral and axisymmetric. Uses `FVMesh` and the native HLLC kernels in FP64. |
| `sst.py` | 167 | `CompressibleSST`: k-ω SST loosely coupled with the same local time step (hoop strain in axisymmetric cases). |
| `riemann.py` | 132 | Exact Riemann solver, oblique-shock and area-Mach relations used by the verification tests. |
| `run.py` | 54 | `build_case`, `solve`: shared by `flowtrace compressible` and `run_batch`. |

### 4.6b `native/`: compiled kernels for the unstructured solvers

| File | Lines | Role |
|---|---:|---|
| `__init__.py` | 246 | Builds `fvm_kernels.cpp` with g++/clang++ on first use (cached), binds it with ctypes; `NativeCSR`, `csr_from_parts`, `bicgstab_jacobi`, `hllc_flux`. `FLOWTRACE_NO_NATIVE=1` or `disabled()` selects the NumPy/SciPy reference path. |
| `fvm_kernels.cpp`, `.h` | 370 | C++/OpenMP: fused gradients, parallel SpMV, matrix assembly, the Jacobi-BiCGSTAB momentum solve, HLLC flux (ABI 2). |
| `cuda.py`, `fvm_kernels.cu` | 411 | The same kernels in CUDA, compiled with NVRTC through CuPy (no nvcc). |

### 4.7 `fluent/`: exchange with Ansys Fluent

| File | Lines | Role |
|---|---:|---|
| `convert.py` | 147 | Bridges structured cases and Fluent-style meshes: `structured_mesh`, `masked_block_mesh`, `structured_solution_points`, `structured_boundary_specs`. Lets the structured solver's field be exported on a Fluent-shaped mesh. |
| `interpolate.py` | 45 | Maps an `.ip` point cloud onto the cells of another mesh: `interpolate_points`, `interpolate_ip_file`. Used by the warm-start study to simulate what Fluent does on import. |
| `export.py` | 215 | Writers for files Fluent reads: `write_interpolation_file` (`.ip`), `read_interpolation_file`, `write_profile` (`.prof`), `write_journal` (`.jou` for Fluent's text interface), `write_pyfluent_script` (`.py`). Typed with mypy. |
| `warmstart.py` | 167 | `flowtrace fluent-warmstart`. Reads a Fluent case, plans, runs the FlowTrace pre-solve, writes the `.ip` file, the optional `.dat.h5`, the journal and the PyFluent script. `write_warmstart_scripts` and `run_warmstart`. |
| `warmstart_study.py` | 478 | Warm-start study: cold start versus warm start, counted in iterations until the velocity (and temperature) field is within 10⁻⁴ of the converged field, and in wall time on equal terms (solver steps only; interpolation timed). Has a FlowTrace proxy path (`run_proxy_case`) and a Fluent path (`run_fluent_study`, `_pyfluent_run`), and parses Fluent transcripts (`parse_transcript`). Typed with mypy. |
| `bridge.py` | 124 | Optional live link to a licensed Fluent session through PyFluent (`FluentBridge`, `BridgeStatus`, `pyfluent_status`). Excluded from coverage because it needs a licence. |

### 4.8 `runtime/`: jobs, events, checkpoints

| File | Lines | Role |
|---|---:|---|
| `engine.py` | 635 | The job engine. `Job` runs one simulation: plan, optional FMG start-up, iterate, monitor, validate, export. Unstructured runs carry solver diagnostics in residual events (every ≥ 10 iterations), the result and the log; an unknown boundary-zone name is refused; a divergence error carries `divergence_context` (non-orthogonality, guard state). `EventHub` fans events (status, plan, residuals, frames, geometry, log) out to subscribers. It keeps a bounded history (4000 events) so a late subscriber can catch up with `snapshot()`. Helpers `_downsample`, `_display_range`, `_b64` make frames small enough to send. |
| `batch.py` | 222 | `run_batch` / `flowtrace batch`: many independent cases at once, one process per case with an explicit backend; one GPU worker (cases the cost model gives to the GPU, largest first) and one CPU worker per remaining physical core. Pressure- and density-based cases. |
| `checkpoint.py` | 85 | `CheckpointWriter`: asynchronous checkpoint writer (a background thread writes fields while the solver continues). `load_checkpoint` reads them back. The engine creates a writer when `checkpoint_seconds` is set (CLI `--checkpoint`, seconds between checkpoints; off by default). Typed with mypy. |

### 4.9 `validation/`: verification, validation, benchmarks, reports

| File | Lines | Role |
|---|---:|---|
| `suite.py` | 70 | `run_suite`: runs every solver path against published or exact references. Used by `flowtrace validate`. |
| `checks.py` | 87 | Automatic validation for the built-in structured cases (`validate_structured`), plus `strouhal` for the cylinder. The engine calls it at the end of a job. |
| `reference.py` | 66 | Published reference data: Ghia et al. (1982) cavity profiles (`ghia`) and `compare_ghia`. |
| `mms.py` | 196 | Method of manufactured solutions. `MMSResult` with observed order of accuracy for MAC and SIMPLEC on quads and triangles. Found the pressure-force bug recorded in the changelog. |
| `gci.py` | 57 | Grid Convergence Index (Roache 1994; Celik et al. 2008). `GCIResult`, `gci`. Typed with mypy. |
| `turbulent.py` | 202 | RANS validation cases on the unstructured solver: channel (`run_channel`), flat plate with Schultz-Grunow correlation (`run_flat_plate`), backward-facing step with Driver & Seegmiller (`run_bfs`). |
| `benchmarks.py` | 887 | The reproducible benchmark suite. Every number the project publishes comes from here. `Benchmark`, `BenchResult`, `Metric`, `Series`, `Plot`. Tiers `quick` (about 2 min) and `full` (about 40 min). Individual benchmarks: Taylor-Green, MMS (MAC, SIMPLEC quads, SIMPLEC triangles), projection, cavity (MAC, SIMPLEC quads, SIMPLEC triangles, Re 1000), natural convection (Ra 10⁴, 10⁵, SIMPLEC), channel, mass conservation. |
| `report.py` | 221 | Turns `results.json` into `report.html` (self-contained, SVG plots), `report.pdf` (matplotlib, optional extra) and `validation_table.md`. Embedded CSS is exempted from E501. Typed with mypy. |
| `fluent_files.py` | 202 | Checks against files written by Fluent: fetches the PyFluent mixing-elbow example (MIT licence, SHA-256 verified) and tests seven properties (polyhedral read, zone types, orientation, cell closure, `.msh.h5`/`.cas.h5` consistency, Fluent's node order and c0/c1 on write, `.dat.h5` cell mapping). Used by `benchmarks/fluent_files_check.py`, `tests/test_fluent_files.py` and CI. |

### 4.10 `ui/`: dashboard server and front end

| File | Lines | Role |
|---|---:|---|
| `server.py` | 694 | Standard-library HTTP server (`ThreadingHTTPServer`, no framework) bound to `127.0.0.1:8765` by default. `request_allowed` refuses requests with a non-loopback `Host` (when bound to loopback) or an `Origin` other than `Host` (DNS rebinding, cross-site requests). `AppState` holds the current `Job`, the `EventHub`, uploaded meshes and the work directory. Routes are listed in [section 9.2](#92-dashboard-http-api). Also: project save and open, report generation (`run_report_html`, with the solver-diagnostics table), and `start_background`. |
| `static/index.html` | 255 | Single-page dashboard markup. |
| `static/app.js` | 835 | Front-end logic. Subscribes to `/api/events` (Server-Sent Events), draws residual plots and fields, shows the execution plan, and calls the REST endpoints. No build step. |
| `static/i18n.js` | 87 | English and Persian (RTL) strings. |
| `static/style.css` | 187 | Theme and layout, including the RTL layout for Persian. |

### 4.11 `benchmarks/` (scripts and results, outside the package)

| File | Role |
|---|---|
| `planner_unstructured.py` | Experiment: does the measured planner place the unstructured solver on the right device? Writes `results/planner_unstructured.{json,md}`. |
| `simplec_iteration_cost.py` | Cost of one SIMPLEC iteration on CPU and GPU by mesh size. Source of the GPU "floor" (about 15 ms per iteration on the reference laptop GPU). |
| `warmstart_cylinder_wake.py` | Why the cylinder warm start is slower than cold (Re = 40): compares recirculation lengths. |
| `precision_fp32.py` | FP32 against FP64: field difference, validated quantity, and whether FP32 reaches the steady tolerance; source of the precision rule. |
| `skew_envelope.py` | SIMPLE/SIMPLEC stability against skew angle and α_p on sheared quads; source of the cos²θ envelope. |
| `memory_estimate.py` | Measured peak allocation (tracemalloc) and resident-set growth against the memory estimate. |
| `fluent_files_check.py` | Readers and writers against Fluent-written files (see `validation/fluent_files.py`). |
| `native_kernels.py`, `batch_study.py`, `square_cylinder.py` | Native-kernel speed, batch throughput, the square-cylinder study on user meshes. |
| `fluent_warmstart/results.{json,md}` | Warm-start study output. |
| `results/` | Full benchmark output: `results.json`, `report.html`, `report.pdf`, `validation_table.md`, `gpu_latest.md`, `warmstart_cylinder_wake.md`, and the planner and cost data. |

### 4.12 `tests/`

159 test functions in 21 files (about 3k lines); parametrisation makes about 230 test cases.
`conftest.py` sets up shared fixtures. `fake_gpu.py` is a NumPy-backed CuPy stand-in, so the GPU code
paths run in CI without a GPU. `test_gpu_real.py` runs on a real GPU when one is present, and
`test_fluent_files.py` when the Fluent reference files are present (CI job `fluent-files`).

| Test file | Area |
|---|---|
| `test_structured.py` (12) | MAC solver, cases and presets |
| `test_unstructured.py` (4) | SIMPLE/SIMPLEC solver |
| `test_diagnostics.py` (17) | Solver diagnostics: definitions, identities, rolled-back iterations, divergence report, CG conditions |
| `test_turbulence.py` (6), `test_turbulence_and_models.py` (10) | RANS models, transient, P-1, species, Fluent-style presets |
| `test_compressible.py` (7) | Density-based solver: Sod, oblique shock (2-D and polyhedral 3-D), axisymmetric, SST flat plate, native kernels |
| `test_linalg.py` (8) | Separable, multigrid, Krylov (including breakdown), AMG |
| `test_mesh.py` (7) | Mesh data model, geometry, Fluent face orientation, malformed files |
| `test_zones.py` (9) | Zone handling, periodic and sliding boundaries |
| `test_ansys_io.py` (11) | Fluent `.msh`, `.cas.h5`, `.dat.h5` I/O, warm start, CLI input errors, skewness on polyhedra |
| `test_fluent_export.py` (5) | `.ip`, journal, PyFluent script writers |
| `test_fluent_files.py` (1) | Against files written by Fluent (opt-in) |
| `test_planner.py` (10) | Planner decisions, cache, set-up pay-back, stability guard |
| `test_hardware.py` (5) | Probe, benchmarks, memory estimate as an upper bound |
| `test_native.py` (5) | Native C++ kernels against the NumPy reference |
| `test_batch.py` (5) | `run_batch`, wall forces, checkpoint restart from disk |
| `test_gpu_paths.py` (10) | GPU code paths against the fake GPU, precision rule, gpu-report labels |
| `test_gpu_real.py` (10) | GPU code paths on real hardware (skipped without a GPU) |
| `test_verification.py` (10) | MMS and convergence orders |
| `test_server.py` (5) | Dashboard HTTP API, report, Host/Origin rules, `ui --host` refusal |
| `test_square_cylinder.py` (2) | User-mesh square cylinder (skipped without the meshes) |

### 4.13 `docs/`, `paper/`, `examples/`, `.github/`, packaging

| Path | Role |
|---|---|
| `docs/` | MkDocs Material site: install, user guide (CLI, Python API, Fluent, GPU, dashboard), theory (MAC, SIMPLEC, turbulence, zones, verification), API reference (mkdocstrings), validation pages, design note on compressible flow, and three Jupyter tutorials. `hooks.py` publishes the generated report inside the site. `make_notebooks.py` builds the tutorials. |
| `paper/` | JOSS paper draft (`paper.md`, `paper.bib`). |
| `examples/` | Two small Fluent meshes (`cavity_quad.msh`, `heated_duct_tri.msh`) and a README. |
| `.github/workflows/ci.yml` | CI: lint (ruff, mypy), test matrix (Ubuntu and Windows, Python 3.9 to 3.12), quick benchmark run with report artifact, `mkdocs build --strict`, and packaging. Weekly scheduled run to catch dependency drift. |
| `.github/ISSUE_TEMPLATE/`, `pull_request_template.md` | Bug, feature and validation-discrepancy templates. |
| `.pre-commit-config.yaml` | Pre-commit hooks: whitespace and end-of-file fixers, YAML and TOML checks, merge-conflict and large-file checks, ruff, and a local hook. |
| `pyproject.toml` | Build (setuptools), metadata, optional extras (`cpu`, `gpu`, `amg`, `h5`, `report`, `fluent`, `docs`, `dev`, `all`), ruff, pytest, coverage and mypy settings. |
| `CHANGELOG.md`, `CITATION.cff`, `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `README.md`, `README.fa.md` | Project metadata and documentation (English and Persian). |

---

## 5. Key flows

### 5.1 Dashboard run (structured case)

```mermaid
sequenceDiagram
    participant B as Browser (app.js)
    participant S as ui/server.py
    participant J as runtime/engine.py Job
    participant P as hardware/planner.py
    participant M as structured/solver.py MACSolver
    participant H as EventHub

    B->>S: POST /api/estimate
    S-->>B: memory budget (memory.py)
    B->>S: POST /api/run {kind, preset, params}
    S->>J: Job(spec).start()
    S-->>B: 200 OK
    B->>S: GET /api/events (SSE)
    J->>H: status "planning"
    J->>P: plan_structured(case)
    P->>M: build candidates, time a few steps
    P-->>J: ExecutionPlan (+ reasons)
    J->>H: plan event
    opt FMG enabled
        J->>M: solve coarse case, prolong fields
    end
    loop until converged, stopped or max iterations
        J->>M: step()
        J->>H: residuals; frame every N steps
    end
    J->>J: validate_structured(...)
    J->>H: status converged/finished, result
    B->>S: GET /api/export/zip
    S-->>B: package (msh, ip, jou, py, csv)
```

### 5.2 Unstructured solve (`flowtrace solve`, dashboard with mesh)

1. `fluent_msh.read_fluent_mesh` or `fluent_h5.read_any_mesh` gives an `UnstructuredMesh`.
2. `mesh.quality.mesh_quality` reports quality (optional, shown in the UI).
3. `unstructured.case.default_boundaries` plus the user `--bc` overrides give `BoundarySpec`s.
4. `hardware.planner.plan_unstructured` picks CPU or GPU. It skips the GPU below about 16k cells unless
   the measurement says otherwise (see [section 7](#7-hardware-placement-and-the-planner)).
5. `unstructured.fvmesh.FVMesh` is built on the chosen backend.
6. `SimpleSolver` iterates: momentum, pressure correction (AMG-PCG), velocity and flux correction,
   energy (optional, through `conjugate.py` when solid zones exist), turbulence (`turbulence.py`).
   The SIMPLEC guard watches the continuity residual.
7. Output: fields, `result` JSON, export package (`.msh`, `.ip`, `.jou`, `.py`, `.csv`, `.zip`).

### 5.3 Fluent warm start (`flowtrace fluent-warmstart`)

```mermaid
flowchart LR
    A[Fluent case<br/>.cas.h5 or .msh] --> B[read_any_mesh]
    B --> C[quality check]
    B --> D[default_boundaries<br/>+ --bc overrides]
    D --> E[plan_unstructured]
    E --> F[SimpleSolver<br/>pre-solve]
    F --> G[solution at cell centres]
    G --> H[write_interpolation_file<br/>warmstart.ip]
    G --> I[write_data_h5<br/>warmstart.dat.h5]
    H --> J[write_journal<br/>warmstart.jou]
    H --> K[write_pyfluent_script<br/>warmstart_pyfluent.py]
    J --> L[Fluent: read case, interpolate .ip,<br/>continue iterations]
    K --> L
```

### 5.4 Warm-start study (`flowtrace warmstart-study`)

Two cases run on the same target mesh. The cold case starts from uniform fields. The warm case starts
from the FlowTrace pre-solve, interpolated with `interpolate.py`. The study counts iterations until
the velocity (and temperature) field is within 10⁻⁴ of its converged value. The Fluent side parses
the iteration count from the transcript (`parse_transcript`). Output is `results.{json,md}`.

### 5.5 Benchmark run (`flowtrace benchmark --tier quick|full`)

`benchmarks.benchmark(tier)` runs each `Benchmark`, collects `Metric`s with their acceptance criteria,
writes `results.json`, and calls `report.write_report` for the HTML, PDF and Markdown outputs. The
criteria are set in advance and are not relaxed after the run. Failing checks are reported, not hidden.

---

## 6. Numerical methods

| Area | Structured (MAC) | Unstructured (SIMPLE/SIMPLEC) |
|---|---|---|
| Grid | Cartesian, staggered (u on x-faces, v on y-faces, p and T at centres) | Arbitrary polyhedra, collocated cell-centred |
| Convection | Conservative flux form: central, first-order upwind, QUICK or van Leer TVD | Implicit first-order upwind plus deferred-correction second-order upwind |
| Time | Adams-Bashforth 2 (variable step) | Steady (under-relaxation) or transient: implicit Euler / BDF2 with inner iterations |
| Diffusion | Crank-Nicolson in delta form; explicit when obstacles break separability | Central, explicit non-orthogonal correction |
| Pressure coupling | Incremental projection | SIMPLE (α_p = 0.3) or SIMPLEC (α_p = 1), Rhie-Chow |
| Pressure solver | Separable fast solver; multigrid-PCG with obstacles | AMG-preconditioned CG |
| Gradients | Standard | Least squares (momentum), Green-Gauss face pressure (force, verified) |
| Turbulence | Not implemented in `MACSolver` | SST k-ω (2003), realizable k-ε, standard k-ε with wall functions |
| Energy | Boussinesq | Full energy, conjugate heat transfer, P-1 radiation; passive species |
| Diagnostics | Divergence of the projected field | Componentwise backward errors of each assembled system, linear-solver reduction, local and global continuity per iteration (`diagnostics.py`) |
| Multi-grid start | Full-multigrid-style sequencing (`sequencing.py`) | Not in the code path used by the warm-start study |

Verification (manufactured solutions on warped meshes): observed order of about 2 for velocity and
temperature on quads and triangles. Pressure on triangles converges at order 1.20, below the second
order of the other fields, and the acceptance threshold is set to 1.0.

Stability: the SIMPLEC stability guard (`simple.py`) rolls back to the last healthy snapshot and lowers
the pressure under-relaxation (×0.6, floor 0.1) when the residual becomes non-finite or grows 20× above
its running minimum. The guard is only active when `relax_p` is automatic. On uniformly skewed quads the
largest stable α_p tracks cos²θ (θ the non-orthogonality), because the non-orthogonal part of the p′
Laplacian is neglected and no non-orthogonal correctors are run (`benchmarks/results/skew_envelope.md`).

The density-based compressible solver (HLLC, MUSCL, matrix-free implicit or SSP-RK3) is described in
`docs/theory/compressible.md`.

---

## 7. Hardware placement and the planner

FlowTrace does not assume the GPU is faster. The planner measures.

```mermaid
flowchart TD
    Start([plan_structured / plan_unstructured]) --> Prof[probe.cached_profile<br/>+ bench cache]
    Prof --> Cand[_candidates:<br/>backends that exist and fit<br/>the memory budget]
    Cand --> Time[_time_solver: build the real solver<br/>run a few real steps after warm-up]
    Time --> Pick[choose fastest configuration]
    Pick --> Plan[ExecutionPlan / UnstructuredPlan<br/>Placement per stage + reason]
    Plan --> UI[dashboard shows placements]
    Plan --> Run[solver runs with that plan]

    Cache[(hardware cache<br/>machine fingerprint)] -. calibration .-> Pick
    Plan -. unstructured: skip GPU if<br/>CPU beats cached GPU floor .-> Pick
```

Recorded facts that drive the design:

* **Structured solver:** on a laptop RTX 4060, a 1024² step was about 30× faster on the GPU in FP32 than
  on the 16-thread CPU in FP64. That is not an equal-accuracy comparison, and the GPU's FP64 time was not
  measured on that laptop.
* **Precision rule:** `auto` is FP64. On a consumer GPU the planner times an FP32 candidate only for a
  steady case whose FP32 round-off bound u₃₂·L/(U·Δt) is within `steady_tol` (`fp32_admissible`). In
  `benchmarks/precision_fp32.py`, FP32 stalled above the tolerance in five of six steady cases, at
  0.07-0.33 of the bound. The decision and its reason are the plan's first placement.
* **Unstructured solver:** a GPU SIMPLEC iteration has a floor of about 15 ms in FP64 on that GPU. Per
  iteration the GPU is 3.5 to 9.5× slower than the CPU below 5k cells and breaks even near 16k cells.
  Hence `plan_unstructured` times the CPU first and skips the GPU when the CPU already beats the cached
  floor. It also skips the GPU when, even at its floor, it cannot recover its measured set-up time
  (`gpu_setup_ms`) over the iterations left (not yet re-measured on a GPU).
* **Known cost:** on 100-iteration runs, the planner's timing overhead exceeded its 15 % budget at two of
  six mesh sizes (×1.48 and ×1.43 versus the faster fixed choice). See [section 13](#13-known-limitations).

Memory: `memory.py` estimates the footprint before a run. The dashboard shows the estimate through
`/api/estimate`, and the engine logs the required versus available memory at the start of a run. On the
CPU the estimate bounds the measured peak (estimate/peak 1.23-2.10 over the measured configurations,
traced allocations and resident-set growth); the GPU pool
overhead has not been measured. The planner skips a device the case does not fit on (GPU → CPU
fallback), and raises an actionable `MemoryError` when it fits on none. `FLOWTRACE_MEMORY_CHECK=off`
overrides the refusal.

---

## 8. Data formats and persistence

| Artefact | Format | Written by | Read by |
|---|---|---|---|
| Fluent mesh | `.msh`, `.cas` (ASCII, optional gzip) | `fluent_msh.write_fluent_mesh` | `fluent_msh.read_fluent_mesh` |
| Fluent CFF HDF5 | `.cas.h5`, `.msh.h5` | `fluent_h5.write_case_h5` | `fluent_h5.read_case_h5`, `read_any_mesh` |
| Fluent cell data | `.dat.h5` | `fluent_h5.write_data_h5` | `fluent_h5.read_data_h5` |
| Interpolation file | `.ip` | `export.write_interpolation_file` | Fluent; `export.read_interpolation_file`; `interpolate.py` |
| Profile | `.prof` | `export.write_profile` | Fluent |
| Journal | `.jou` | `export.write_journal` | Fluent (text interface) |
| PyFluent script | `.py` | `export.write_pyfluent_script` | PyFluent |
| Hardware cache | JSON, keyed by machine fingerprint | `bench.py`, `planner.py` | `planner.cached_calibration`, `probe.cached_profile` |
| Benchmark results | `results.json`, `report.html`, `report.pdf`, `validation_table.md` | `benchmarks.py`, `report.py` | Humans; `docs/hooks.py` |
| Checkpoints | Written by `CheckpointWriter` | `checkpoint.py` (async thread) | `checkpoint.load_checkpoint` |
| Projects | `<workdir>/projects/<name>` (spec, UI state, uploaded mesh, last result) | `server.save_project` | `server.open_project` |
| Run reports | Self-contained HTML with a PNG snapshot | `server.run_report_html` | Humans |
| Export package | `.msh`, `.ip`, `.jou`, `.py`, `.csv`, `.zip` | `Job.export_package` | Fluent, users |

The work directory is set with `--workdir` (default `flowtrace_runs`, relative to the current directory).
It holds runs, checkpoints, exports, `projects/` and reports. The hardware cache is kept separately and
is not in this directory.

---

## 9. Interfaces

### 9.1 Command line

```
flowtrace                       # = flowtrace ui
flowtrace ui                    # dashboard at http://127.0.0.1:8765 (non-loopback --host needs --allow-remote)
flowtrace probe                 # what FlowTrace sees on this machine
flowtrace bench                 # micro-benchmark every backend
flowtrace run cavity --re 1000 --n 512 [--backend cupy] [--export out/]
flowtrace mesh FILE             # inspect a Fluent mesh and its quality
flowtrace quality FILE          # quality report; exit 2 if critical
flowtrace h5-inspect FILE       # print the HDF5 group/dataset tree
flowtrace solve FILE --bc NAME=ZONE[:VALUE...] [--energy] [--turbulence ...] [--solid ...] --export out/
flowtrace compressible FILE --case case.json --export out/
flowtrace case NAME             # built-in Fluent-style cases (SST channel, flat plate, P-1 slab, ...)
flowtrace batch FILE... [--set key=v1,v2] --out runs/
flowtrace fluent-warmstart CASE --turbulence ... --bc ... --out ws/
flowtrace validate              # validation suite
flowtrace benchmark [--tier quick|full] [--out DIR]
flowtrace warmstart-study ...
flowtrace gpu-report
flowtrace parse-transcript cold.trn warm.trn
```

Exit status: 0 success; 1 the run failed or a check failed (`benchmark`, `validate`, `batch`); 2 usage or
input error (one line on stderr; `FLOWTRACE_DEBUG=1` for the traceback) and `quality` below Fluent's
limits; 130 interrupted.

### 9.2 Dashboard HTTP API

All routes are served by `ui/server.py` on `127.0.0.1:8765`. The server is local only, and it has no
authentication. Every request passes `request_allowed` first: when bound to loopback the `Host` header
must name a loopback host, and an `Origin` header, when sent, must equal `Host` (`null` is refused).
Anything else gets 403, which blocks DNS rebinding and cross-site requests.

| Method and path | Purpose |
|---|---|
| `GET /` | Dashboard page (`index.html`) |
| `GET /api/system` | Hardware profile, backends, cached benchmarks, PyFluent status |
| `POST /api/benchmark` | Run micro-benchmarks on every backend |
| `GET /api/presets` | Built-in cases and their parameters |
| `POST /api/estimate` | Memory budget for a case before running it |
| `POST /api/run` | Start a job: `{kind, preset | case | mesh_id, params, overrides}` |
| `POST /api/stop` | Stop the running job |
| `POST /api/view` | Change the displayed field |
| `GET /api/state` | Current job, plan, residual history, log |
| `GET /api/events` | Server-Sent Events stream: status, plan, residuals, frame, geometry, log |
| `POST /api/mesh` | Upload a Fluent mesh (raw body, `X-Filename` header) |
| `GET /api/mesh/...` | Mesh geometry for the viewer |
| `GET /api/export/<kind>` | Download `msh`, `ip`, `jou`, `py`, `csv` or `zip` |
| `GET /api/projects` | List saved projects |
| `POST /api/projects` | Save `{name, spec, ui}` plus mesh and last result |
| `GET /api/projects/<name>` | Open a project (re-registers its mesh) |
| `POST /api/report` | Self-contained HTML report of the current run (`image`: PNG data URL) |

### 9.3 Python API

The public Python API is documented in `docs/user-guide/python-api.md` and `docs/api/`
(mkdocstrings, NumPy docstrings). Entry points a user will import: `flowtrace.structured.case`
(presets), `flowtrace.structured.solver.MACSolver`, `flowtrace.unstructured.case.UnstructuredCase`,
`flowtrace.unstructured.simple.SimpleSolver`, `flowtrace.mesh.fluent_msh`, `flowtrace.mesh.fluent_h5`,
`flowtrace.hardware.planner`, `flowtrace.validation.*`.

### 9.4 Fluent

FlowTrace talks to Fluent in three ways:

1. File handoff: `.ip` interpolation file plus a journal (`.jou`), which Fluent runs in its text interface.
2. A PyFluent script (`.py`) that opens the user's case, reads the `.ip`, and runs.
3. Live link through `fluent/bridge.py`, if PyFluent and a licence are present.

---

## 10. Validation and verification

Three layers, each with its own code:

| Layer | Question | Code |
|---|---|---|
| Code verification | Does the discretisation converge at the design order? | `mms.py`, `gci.py`, `benchmarks.py` (MMS and projection tests), `test_verification.py` |
| Validation | Does the solver match published or exact results? | `reference.py` (Ghia), `turbulent.py` (channel, flat plate, BFS), `benchmarks.py` (de Vahl Davis, Taylor-Green, CHT energy balance, MRF) |
| Engineering checks | Is this case's result plausible, and was the run stable? | `checks.py`, solver residuals, SIMPLEC guard logs |
| Algebraic checks | Did each linear system and the continuity constraint reach what the residuals claim? | `unstructured/diagnostics.py` (every run: events, result, report) |
| Interoperability | Do the Fluent readers and writers match Fluent's own files? | `validation/fluent_files.py` (Fluent-written mixing elbow) |

Published results (`docs/validation/index.md`, from the full tier): 68 of 70 acceptance checks pass in
the committed run (FlowTrace 0.2.0; the re-run on the release code is in progress). The two failures are
recorded openly: the realizable k-ε flat plate at y⁺ ≈ 56 (−7.0 % against a 6 % criterion; since traced to
the wall-function constants, −4.96 % with Fluent's κ) and the cylinder warm start (×0.65, slower than cold,
a property of the mesh pair confirmed by a pre-registered test). Both have analysis in the report.

---

## 11. Build, packaging, CI and documentation

```mermaid
flowchart LR
    Dev[Developer push / PR] --> CI
    subgraph CI[.github/workflows/ci.yml]
        Lint[lint: ruff + mypy]
        Test[test: pytest on<br/>Ubuntu + Windows,<br/>Python 3.9-3.13]
        Ref[test-reference-path:<br/>FLOWTRACE_NO_NATIVE=1]
        FF[fluent-files:<br/>Fluent-written mixing elbow]
        Bench[benchmarks-quick:<br/>flowtrace benchmark --tier quick]
        Docs[docs: mkdocs build --strict]
        Pkg[package: build wheel + sdist]
        Lint --> Test --> Bench
        Lint --> Ref
        Lint --> FF
        Test --> Pkg
    end
    Weekly[Weekly cron] --> CI
    Bench --> Art[(validation-report artifact)]
```

* **Packaging:** setuptools via `pyproject.toml`. Version comes from `flowtrace.__version__`. The
  console script `flowtrace` points to `cli:main`. Package data is `flowtrace/ui/static/*`.
* **Extras:** `cpu` (Numba, psutil), `gpu` (CuPy CUDA 12, nvidia-ml-py), `amg` (pyamg), `h5` (h5py),
  `report` (matplotlib), `fluent` (ansys-fluent-core), `docs`, `dev`, `all`.
* **Lint and types:** ruff (line length 130, rules E, F, W, B). mypy checks the typed core:
  `gci`, `report`, `mesh/quality`, `mesh/grading`, `hardware/memory`, `fluent/export`,
  `fluent/warmstart_study`, `runtime/checkpoint`, `unstructured/diagnostics`. Numerical modules rely on
  tests instead.
* **Tests:** pytest, `testpaths = ["tests"]`, a `slow` marker for long validation tests. Coverage excludes
  `fluent/bridge.py`.
* **Docs:** MkDocs Material with mkdocstrings (NumPy style), mkdocs-jupyter for the tutorials (executed
  outside CI, `execute: false`), MathJax for equations. `docs/hooks.py` copies the generated
  `benchmarks/results` into the site. The build runs with `--strict`.
* **Launcher:** `start_flowtrace.bat` for Windows users, with a dry-run switch
  (`FLOWTRACE_LAUNCHER_DRYRUN=1`) that installs and checks without starting the dashboard. Neither the
  tests nor CI run it.

---

## 12. Cross-cutting concerns

**Backends and one code path.** Solver code is written once against an `xp` array module (NumPy or
CuPy) and a `Backend` object. Hot loops have optional fused implementations: Numba on the CPU
(`structured/kernels.py`), CUDA through NVRTC on the GPU (`structured/cuda_kernels.py`). The GPU path
is tested on a NumPy-backed fake (`tests/fake_gpu.py`) in CI and on real hardware where available.

**Precision.** `resolve_dtype` gives FP64 unless the case asks for `fp32`. The planner adds FP32 only for a
steady structured case on a consumer GPU that passes `fp32_admissible`, and records why
(`ExecutionPlan.precision_reason`). The unstructured and compressible solvers are FP64 unless set
otherwise. The published validation results are FP64 on the CPU.

**Concurrency.** The dashboard server runs the job in a background thread (`Job.start`). Events go
through `EventHub`, a thread-safe queue fan-out. Checkpoints are written by a separate thread
(`CheckpointWriter`). Numba's OpenMP busy-wait is disabled at import time so it does not starve the
NumPy and SciPy calls around it.

**Failure handling.** Missing optional dependencies raise `BackendUnavailable` or are detected at import
time, and the planner routes around them. Mesh files that cannot be parsed raise `FluentMeshError` with
the file and the cause. The CLI reports input errors in one line with exit status 2. A case that does not
fit in memory on any device is refused before it allocates (see [section 7](#7-hardware-placement-and-the-planner)). A diverging SIMPLEC run is rolled back by the
stability guard, and every rollback is logged and written to the run report. A Krylov breakdown is
returned as "not converged", so the guard and the divergence report see it. A run that still diverges
stops with the mesh's non-orthogonality and the guard state in the message. Job errors set status
`error` and are sent to the UI.

**Security and exposure.** The dashboard binds to `127.0.0.1` by default. It has no authentication,
because it is meant for a single user on their own machine. Requests from other web pages (non-loopback
`Host`, foreign or `null` `Origin`) are refused with 403. A non-loopback `--host` needs `--allow-remote`
and prints a warning. Uploaded meshes are stored in the work directory with sanitised names
(`_safe_name`).

**Privacy.** No telemetry or network calls are part of the core. The optional PyFluent bridge talks only
to the user's local Fluent session. `validation/fluent_files.download` fetches the public Fluent example
files only when called explicitly (benchmark script, CI).

**Internationalisation.** The dashboard is in English and Persian (`i18n.js`, RTL CSS). The docs are in
English, with `README.fa.md` for Persian.

**Reproducibility.** Every published number comes from `flowtrace benchmark` and is written to
`benchmarks/results/`. Acceptance criteria are fixed before a run. The hardware fingerprint is stored with
each cached calibration so results from different machines are not mixed.

---

## 13. Known limitations

These are stated in the code, docs, changelog and `RELEASE_READINESS.md`. They are not hidden defects.

| Area | Limitation |
|---|---|
| Fluent files | Readers checked against Fluent-written files; Fluent reading files written by FlowTrace has not been checked (needs a licence). 3-D files written by FlowTrace ≤ 0.3.0 have every face reversed for Fluent: re-export them. |
| Fluent warm start | The Fluent column of the warm-start study has not been run (no licence); the proxy is FlowTrace's own SIMPLEC. On the small 2-D proxy cases the pre-solve costs more wall time than it saves. |
| SIMPLEC | No non-orthogonal correctors: the largest stable α_p falls with cos²θ on skewed meshes. The guard handles the measured cases (up to 60° on sheared quads). |
| Planner | On 100-iteration runs, timing overhead exceeded the 15 % budget at 2 of 6 mesh sizes. The set-up pay-back rule that targets one of them is not yet measured on a GPU; the other (65k cells) is not addressed. |
| Precision | FP32 is admitted by a round-off bound, not by a measured error of the case at hand; mixed precision (FP32 storage, FP64 accumulation) is not implemented. |
| Turbulence | Standard k-ε diverges on the flat-plate set-up (overflow); realizable k-ε and SST are the validated choices there. Wall-function results depend on y⁺ (table in `docs/theory/turbulence.md`). |
| Residuals | The Fluent-style momentum residual of the first iteration from rest divides by a 1e-300 floor (~1e302); the convergence test starts at iteration 6, and the diagnostics are defined in that case. |
| Pressure order | Pressure on triangles converges at order 1.20, not 2. |
| Compressible | Verified on canonical cases; not compared with Fluent (FFF-7) or run on the full rocket mesh; the CUDA kernel has not run on a GPU. |
| Memory | The estimate bounds the measured CPU peak, so a case near the limit can be refused although it would fit (`FLOWTRACE_MEMORY_CHECK=off` overrides); the GPU pool overhead is not measured. |
| Checkpoints | Restart is available from the Python API (`state_dict` / `load_state`, also from the files `CheckpointWriter` writes); the CLI and dashboard do not resume runs, and the stability guard's lowered `relax_p` is not saved. |
| Scale | Single node. No domain decomposition or MPI. |
| Dashboard | Local single-user server, no authentication (loopback, Host/Origin checks). |
| Platforms | CI runs Ubuntu and Windows (Python 3.9-3.13); the Windows launcher `start_flowtrace.bat` is not run by tests or CI; macOS is not tested. |
| Fluent | Live Fluent link needs a licence and is not covered by tests (excluded from coverage). |

---

## 14. Repository map

```
gputest/
├── flowtrace/                  Python package (core)
│   ├── __init__.py, __main__.py, cli.py
│   ├── hardware/               probe, bench, backend, planner, memory, gpu_report
│   ├── linalg/                 separable, multigrid, krylov, amg
│   ├── mesh/                   unstructured, fluent_msh, fluent_h5, quality, grading
│   ├── structured/             case, solver (MAC), sequencing, kernels, cuda_kernels
│   ├── unstructured/           case, fvmesh, simple, diagnostics, turbulence, conjugate, radiation, presets
│   ├── compressible/           case, solver (HLLC, density-based), sst, riemann, run
│   ├── native/                 C++/OpenMP kernels (built on first use) and their CUDA versions
│   ├── fluent/                 convert, interpolate, export, warmstart, warmstart_study, bridge
│   ├── runtime/                engine, batch, checkpoint
│   ├── validation/             suite, checks, reference, mms, gci, turbulent, benchmarks, report, fluent_files
│   └── ui/                     server.py, static/ (index.html, app.js, i18n.js, style.css)
├── benchmarks/                 experiment scripts and published results
├── tests/                      pytest suite (21 files) and fake GPU
├── docs/                       MkDocs site, theory, API, tutorials (ipynb)
├── examples/                   two Fluent meshes
├── paper/                      JOSS paper draft
├── .github/                    CI workflow, issue and PR templates
├── pyproject.toml              packaging, extras, tool configuration
├── mkdocs.yml                  site configuration
├── start_flowtrace.bat         Windows launcher
├── README.md, README.fa.md     project overview (English, Persian)
├── CHANGELOG.md, RELEASE_READINESS.md, CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md, LICENSE
└── .gitignore, .pre-commit-config.yaml
```
