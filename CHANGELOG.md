# Changelog

All notable changes to FlowTrace. The project follows [Semantic Versioning](https://semver.org/):
MAJOR for incompatible API/CLI changes, MINOR for new features, PATCH for fixes.
Until 1.0.0, minor versions may still change the Python API; the CLI stays backward compatible.

## [Unreleased]

Release-readiness work. Every claim below has a script, a test or a results file behind it.
[`RELEASE_READINESS.md`](RELEASE_READINESS.md) lists the commands that were run, their outcomes,
and what is still open.

### Changed (may change results, scripts or habits)
- **`precision="auto"` is now FP64 unless FP32 is shown to meet the case's own accuracy
  target.** Before, `auto` chose FP32 on every consumer GPU without a check.
  - Measured by `benchmarks/precision_fp32.py` on six cavity and natural-convection cases. FP32
    fields stay within 4.1e-5 of FP64, and the validated quantities agree. But FP32 reached
    the steady tolerance of 1e-6 in only one case. In the other five its residual stalled at
    0.07-0.33 of the round-off bound u32·L/(U·dt), and the run went on to 3× the FP64 step
    count without converging.
  - The structured planner now times an FP32 candidate on a consumer GPU only when that bound
    is within `steady_tol` (`fp32_admissible`). Transient cases stay FP64 unless
    `precision="fp32"` is given.
  - The decision and its reason are the plan's first placement ("Numerical precision",
    `ExecutionPlan.precision_reason`, `precision_by_backend`).
  - An explicit `precision="fp32"` is still honoured, with a note when the bound exceeds
    `steady_tol`.
- `flowtrace gpu-report` times every backend in FP64 and lists the GPU's FP32 time separately.
  The committed laptop table (README, `docs/validation/gpu.md`) predates this change: its GPU
  column is FP32 against FP64 on the CPU, so it is not an equal-accuracy comparison.
- **The k-ε wall functions use Fluent's constants, κ = 0.4187 and E = 9.793.** They used κ = 0.41
  with E = 9.793 (B = 5.57), which matches neither Fluent nor the classical log law.
  - Realizable k-ε flat plate against Schultz-Grunow: −6.97 % → −4.96 % (criterion 6 %).
  - Channel at Re_τ = 392 against the DNS: standard k-ε 3.5 % → 1.9 %, realizable
    7.6 % → 5.8 %.
  - SST and Spalding's law keep κ = 0.41.
  - The flat-plate result depends on y⁺ (−7.3 % at y⁺ ≈ 30 to −2.1 % at y⁺ ≈ 105; table in
    `docs/theory/turbulence.md`).
- **Dashboard security (breaking for `ui --host`).**
  - The server now refuses, with HTTP 403, any request whose `Host` is not a loopback name
    (when it is bound to loopback) or whose `Origin`, if sent, differs from `Host`. This closes
    DNS-rebinding and cross-site requests: any web page the user opened could otherwise start
    runs and write files.
  - `flowtrace ui --host <non-loopback address>` no longer starts without `--allow-remote`
    (exit 2). With it, a warning is printed, because the dashboard has no authentication.
- **CLI errors.** A user error (missing or unreadable file, malformed mesh, unknown zone name,
  unavailable backend, out of memory) now prints one line, `flowtrace <command>: error: ...`, and
  exits with status 2. It used to print a Python traceback with status 1.
  - This holds whether the command finds the error or the solver job does (`run`, `solve`,
    `case`).
  - A diverged run prints its one-line diagnosis and exits with 1. The dashboard log gets the
    same lines; unexpected faults keep their traceback.
  - `FLOWTRACE_DEBUG=1` restores the traceback; Ctrl-C exits with 130.
- A boundary condition for a zone the mesh does not have is now refused with the list of
  zones (dashboard, `flowtrace solve`). Before, it was silently ignored.
- **Memory refusal for unstructured cases.** The planner already skipped a device a case does
  not fit on, and refused a case that fits on none. With the old estimate (1.9-3.3× too low)
  that check almost never triggered, so a case too big for the RAM started anyway. With the
  corrected estimate it does trigger. The refusal comes before anything large is allocated and
  says how much memory may be needed, how much is free, and what would help. Because the
  estimate is an upper bound, a case near the limit can be refused although it would have fitted.
  `FLOWTRACE_MEMORY_CHECK=off` starts it anyway.

### Fixed
- **CI was red on every run, including the base of this release**, for a reason outside the code.
  `ubuntu-latest` is now Ubuntu 26.04, and `actions/setup-python` has no Python 3.9 for it (3.9
  has been end of life since October 2025). So the `test (ubuntu-latest, 3.9)` cell failed in
  its set-up step. The `package` and `benchmarks-quick` jobs need the whole matrix, so they never
  ran. Every other job passed. The Linux 3.9 cell now runs on `ubuntu-24.04`, which still has 3.9.
- **The pressure-correction CG raised `ZeroDivisionError` on a diverging run** when r·z underflowed
  to exactly zero. The stability guard and the divergence report then never ran. It depends on
  round-off: it showed up in three tests on Python 3.9 with NumPy 2.0 and SciPy 1.13, not with
  NumPy 2.5. CG now reports the breakdown (not converged) and the outer iteration handles it.
  Iterations that did not break down are unchanged, because the new branch is taken only where
  the old code raised. Regression test: `test_pcg_breakdown_is_reported_not_raised`.
- **Fluent 3-D face orientation, in the readers and the writers (`.msh`, `.cas`, `.cas.h5`).**
  - Fluent stores a 3-D face so that the right-hand normal of its node order points toward c0.
    This was measured on Fluent-written files and is the convention OpenFOAM's
    `fluentMeshToFoam` reads.
  - FlowTrace read every 3-D face reversed. Its geometry repair flipped them back, so solutions
    were right, but every Fluent 3-D mesh was reported as "re-oriented on import".
  - FlowTrace also wrote 3-D files with every face reversed relative to Fluent; nobody has
    checked what Fluent does with such a file.
  - 2-D was already right.
  - 3-D files written by FlowTrace ≤ 0.3.0 still read: the repair flips them, and the quality
    report notes it.
- **Equiangle skewness is no longer computed for polyhedral cells**, for which Fluent does not
  define it. It flagged false CRITICAL values: `flowtrace quality` exited with 2 on Fluent's own
  mixing-elbow mesh. Skewness is now computed for standard elements only (triangle and quad;
  tetrahedron, pyramid, prism and hexahedron). Other cells are reported as n/a and judged by
  orthogonal quality.
- **Malformed `.msh`/`.cas` files** raised raw `IndexError`, `KeyError` or `zlib.error`. They now
  raise `FluentMeshError`, naming the file and the cause; eight malformed-file tests. HDF5 files
  that are not complete CFF meshes or data (`.cas.h5`, `.msh.h5`, `.dat.h5`) do the same, where
  they used to raise a `KeyError` from inside h5py.
- **Memory estimate for unstructured cases.** Measured peaks were 1.9-3.3× the estimate. A
  measured-overhead term makes it an upper bound over six SIMPLEC configurations
  (`benchmarks/memory_estimate.md`). Estimate/peak is 1.31-1.56 against traced allocations
  (tracemalloc), and 1.23-2.10 against the resident-set growth of a fresh process, which also
  counts memory the native kernels allocate and varies by a few per cent between runs. The MAC
  estimate was already an upper bound (1.60-1.77). The CuPy pool overhead on a GPU has not been
  measured.
- The dashboard's memory estimate used 8-byte values for FP32 cases.
- When the structured planner rebuilds the winning solver (`keep_solver=False`), it now uses the
  precision that was timed.
- The run report's local-continuity label now says what is computed: max cell imbalance / (sum
  of |face mass fluxes| per cell).
- Hardening: the dashboard serves static files only from inside `static/`. The check is now
  path containment instead of a string prefix, which a sibling directory such as `static2`
  would have passed; no such directory exists.

### Added
- **Checks against files written by Ansys Fluent** (`flowtrace.validation.fluent_files`,
  `benchmarks/fluent_files_check.py`, opt-in `tests/test_fluent_files.py`, CI job
  `fluent-files`).
  - Data: the PyFluent mixing-elbow example (MIT licence), about 9 MB, fetched on demand and
    SHA-256 verified.
  - All seven properties pass, including:
    - polyhedral CFF read (17,822 cells, 3-9 nodes per face);
    - no face re-oriented and every cell closed to 2e-16;
    - `.msh.h5` and `.cas.h5` with the same topology;
    - Fluent's node order and c0/c1 reproduced on write for all 91,581 faces;
    - `.dat.h5` data mapped to the right cells (outlet/inlets mass flow 1.0021).
  - Not covered: Fluent reading files that FlowTrace wrote (that needs a licence).
- **Planner: no GPU timing that cannot pay for itself.** An unstructured GPU candidate is not
  timed when, even at its floor cost, it cannot recover its measured set-up time
  (`gpu_setup_ms`, new in the cached calibration) over the iterations left. This targets the
  9,216-cell overhead in the planner study. **It has not been re-measured on a GPU** (none was
  available).
- `pressure_matrix_properties()` checks symmetry, sign pattern and diagonal dominance of the
  assembled pressure-correction matrix: the conditions for using CG. The execution plan states
  them, and warns when they fail.
- A diverged pressure-based run now reports the mesh's maximum non-orthogonality, whether the
  stability guard was off (explicit `relax_p`) or exhausted (`stability_guard_exhausted_at` in
  `describe()`), and the measured skew limit.
- Warm-start study: wall-clock time is measured on equal terms. Only solver steps are counted,
  and the interpolation is timed (`cold_solver_seconds`, `warm_solver_seconds`,
  `interpolation_seconds`, `wallclock_ratio`). The full-tier benchmark reports the ratio as
  information, with no criterion.
- Measurement scripts with committed results: `benchmarks/skew_envelope.py` (stability against
  skew angle and α_p), `precision_fp32.py`, `memory_estimate.py` and `fluent_files_check.py`.
- Tests: dashboard Host/Origin rules and a live refusal check; `ui --host` without
  `--allow-remote`; FP64-by-default and FP32 admissibility on the fake GPU; gpu-report precision
  labels; the planner pay-back rule; memory upper bound, GPU → CPU fallback, refusal and
  override; Fluent face orientation (`.msh`, `.cas.h5`, a hand-written Fluent hexahedron);
  malformed meshes; skewness on polyhedra; checkpoint restart from disk; one-line CLI and job
  errors with their exit statuses; CG breakdown.
- CI: Python 3.13 added to the test matrix (it was already claimed); the `fluent-files` job.

### Documented
- **Skew stability envelope** (`docs/theory/simplec.md`, `benchmarks/results/skew_envelope.md`).
  On uniformly sheared quads the largest stable α_p tracks cos²θ, because the non-orthogonal part
  of the p′ Laplacian is neglected. Non-orthogonal correctors are not implemented. The automatic
  guard converged at every angle up to 60°, while SIMPLE at α_p = 0.3 diverged from 30°.
- **Cylinder warm start.** A pre-registered test of the explanation in the validation report
  (`docs/validation/warmstart.md`). A target mesh that resolves the cylinder like the pre-solve
  keeps the saving: ×1.58 at 8 cells/D and ×2.04 at 12 cells/D (558 → 274 iterations). A coarser
  target lowers it (×1.18 at 8 → 6 cells/D) or reverses it (×0.65 at 12 → 9 cells/D, the
  committed case). The failing check is a property of that mesh pair; it stays a failing check.
- **Warm-start wall time.** In the committed run's data, the pre-solve plus the warm run took
  longer than the cold run alone in every proxy case (1.5× to 28×), and the extra cylinder runs
  agree. Iterations are saved; on these small 2-D cases wall time is not. The study now measures
  this on equal terms (see Added).
- `docs/theory/diagnostics.md`, and the evaluation points of every residual in
  `docs/theory/simplec.md`.

### Added: SIMPLE/SIMPLEC solver diagnostics
- `SimpleSolver.diagnostics()` (`flowtrace/unstructured/diagnostics.py`, `docs/theory/diagnostics.md`)
  reports, for the latest outer iteration and each on the exact system it names: the momentum
  (and energy) residual of the assembled system at the old field and of the under-relaxed system
  at the linear solution, both as componentwise backward errors (Oettli-Prager) with absolute
  norms; the explicitly computed linear-solver residual reduction next to the solver's
  recurrence value; predictor and corrected local continuity (cell imbalance relative to the cell
  throughflow); the signed global boundary imbalance; the field change per iteration; and
  non-finite counts. Rows below the rounding error of the largest rows (gamma_K u) are excluded
  and counted; 0/0 = 0, x/0 = inf.
- The systems are referenced, not copied, and released at the start of the next iteration (peak
  memory unchanged); a bit-for-bit check of 64 arrays over six configurations on the native and
  the NumPy path shows the iteration itself is unchanged. One evaluation costs 0.2-0.3 of an
  iteration; the engine evaluates every >= 10 iterations (<= 3 % of run time).
- Exposed in residual events, the run result, the run log, the HTML run report and `run_batch`
  results; `ConjugateSolver.diagnostics()` for the fluid sub-mesh. Tests in
  `tests/test_diagnostics.py`, GPU-path agreement in `tests/test_gpu_paths.py`.

### Documented
- The Fluent-style momentum residual divides by a 1e-300 floor when every velocity is exactly zero
  (first iteration from rest: ~1e302). Unchanged, because it feeds the convergence test (which
  starts at iteration 6); the diagnostics are defined in that case.

### Added: density-based compressible solver (`flowtrace.compressible`)
- Euler / Navier–Stokes / k-ω SST for an ideal gas on 2-D, 3-D polyhedral and axisymmetric
  meshes: HLLC flux, MUSCL with Venkatakrishnan or Barth limiting, matrix-free implicit steady
  solver with CFL ramping, SSP-RK3 transient, Fluent-style pressure inlet/outlet, far-field,
  wall, symmetry/axis boundaries. Theory and verification: `docs/theory/compressible.md`.
- Compressible k-ω SST (Menter 2003, low-Re wall treatment, hoop strain in axisymmetric cases),
  loosely coupled with the same local time step; `turb_intensity` / `turb_visc_ratio` per zone.
- Native HLLC kernels: `ft_hllc_flux` in C++/OpenMP (ABI 2) and CUDA; picked automatically in
  FP64. 89x faster than the NumPy expression on 1M 3-D faces (4 threads).
- `flowtrace compressible MESH --case case.json --export DIR` (writes `.ip` and `.npz`), and
  `BatchJob(solver="density-based")` so `run_batch` places compressible cases on CPU workers and
  the GPU alongside pressure-based ones.
- `agglomerate_cells()`: merge cells into polyhedra (used to test polyhedral meshes end to end
  through the Fluent `.msh` writer and reader).
- Tests (`tests/test_compressible.py`): Sod tube (order 0.9 in L1), M = 2 / 10° oblique shock
  (p2/p1 1.7067 vs 1.7066 exact), the same on 24-face polyhedra in 3-D via `.msh`, axisymmetric
  rest state (laminar and SST), SST flat plate, native vs array kernels; CuPy-path and real-GPU
  tests.
- Not yet done: FFF-7 comparison with Fluent and the full rocket-mesh run (the Fluent data and
  the rocket mesh are not in the repository), and a CUDA run on real hardware.

### Fixed
- Compressible scaled residuals: momentum components now share one scale, so a component that
  starts at round-off level (w in a quasi-2-D 3-D case) no longer blocks convergence.

### Added (merged from the GitHub branch: commits 57b2937, 3d52815)
- Transient pressure-based solver: implicit Euler / BDF2 with inner iterations per time step
  and Choi's Rhie-Chow correction; transient Taylor-Green on a doubly periodic mesh as a check.
  Time terms also apply to k and omega/epsilon.
- P-1 radiation with Marshak wall emissivity, coupled to energy; exact slab solution as a check.
- Passive species transport (laminar + turbulent Schmidt diffusion), analytic mixing layer.
- Built-in Fluent-style cases (`flowtrace case ...`, dashboard "Fluent-style" group): SST
  channel, SST flat plate, P-1 slab, transient Taylor-Green, mixing layer, each with checks.
- `wall_report()` / `wall_profile()` (Cd, Cl, y+, Cf, convective and radiative heat flow),
  `periodic_mass_flow()`; `case.periodic` zone pairs and `periodic_pressure_gradient`.
- Fluent journals and PyFluent scripts set SST, P-1 and transient settings.
- Duplicated features (SST, periodic zones) keep the local, verified implementations; the
  GitHub-side tests run unchanged against them.

### Fixed
- The run report showed the shared server log and residual history, so a report could carry
  another job's validation lines (an external audit found Ghia errors from a CPU MAC run next to
  a GPU SIMPLEC run). Log and residual events now carry the job id and the report filters by it;
  it also shows the local continuity (max cell imbalance), not only the net mass balance.
- The CFF reader failed on Fluent Meshing 2025 files (one-element array attributes; numbered
  node-coordinate sections). It now reads the 3.19M-cell rocket mesh in 1.5 s.

## [0.3.0] - 2026-10-26

### Added
- Native kernels for the unstructured solver (`flowtrace/native/`). `fvm_kernels.cpp`
  (C++17 + OpenMP, C API in `fvm_kernels.h`) is compiled with g++/clang++ on first use and
  cached, statically linked on Windows. It provides parallel CSR products, fused least-squares
  and Green-Gauss gradients, face interpolation, matrix assembly and a complete
  Jacobi-BiCGSTAB solve. `fvm_kernels.cu` holds the same face/cell kernels for the GPU plus
  fused BiCGSTAB updates and reductions (4 host synchronisations per iteration instead of 6);
  it is compiled by NVRTC, so it needs no `nvcc` or host compiler. Results equal the reference
  path (bit-identical products, gradients to 1e-13, identical Krylov iteration counts). Speed:
  `benchmarks/native_kernels.py`, `docs/user-guide/native.md`. The reference path remains,
  via `FLOWTRACE_NO_NATIVE=1` or when no compiler is found, and CI runs it.
- `flowtrace probe` reports whether the C++ and CUDA kernels are active.
- Square-cylinder validation on the author's GAMBIT four-mesh set
  (`benchmarks/square_cylinder.py`, `docs/validation/square-cylinder.md`): mesh study and
  GCI, a Reynolds sweep checked against published scaling laws, and a domain-size study. The
  coarse mesh is a regression test (`tests/test_square_cylinder.py`, `tests/data/`).
- `flowtrace batch` and `flowtrace.runtime.batch.run_batch`: solve many unstructured cases at
  once, such as mesh-independence sets or `--set key=v1,v2` parameter sweeps. One GPU worker
  takes the cases the calibrated cost model assigns to the GPU, and every other physical core
  runs one CPU case with single-threaded BLAS. Each case writes a Fluent `.ip` file, an `.npz`
  of all fields and a wall-force summary, and one failing case does not stop the batch.
- `wall_forces()` on the unstructured solvers: pressure and viscous force on every wall zone,
  from the momentum equation's own wall terms (wall-function or SST wall viscosity with
  turbulence). It balances a body-force-driven periodic channel to 1e-8.

### Performance
- Measured backend planner for the unstructured solver (`plan_unstructured`), used by the
  dashboard, `flowtrace solve` and `flowtrace fluent-warmstart`. Before, `backend="auto"` always
  chose the GPU. A GPU SIMPLEC iteration has a floor of about 15 ms on an RTX 4060 Laptop GPU
  in FP64. Per iteration, the GPU is therefore 3.5–9.5× slower than the CPU below 5k cells and
  breaks even near 16k cells (`benchmarks/simplec_iteration_cost.py`). The planner times the
  CPU, skips the GPU when the CPU beats the GPU's cached floor, and otherwise times both. In
  the study (`benchmarks/planner_unstructured.py`, `docs/validation/planner.md`) it always ends
  on the right device. On 100-iteration runs, though, its timing overhead exceeds the study's
  15 % budget at 2 of 6 mesh sizes (×1.48 and ×1.43 the faster fixed choice).

### Fixed
- Runs were not reproducible. pyamg draws random vectors from NumPy's global RNG for its
  spectral-radius estimates, so two identical runs differed by ~1e-8 after 20 iterations. The
  setup is now seeded, the caller's RNG state is restored, and runs in separate processes are
  bit-identical.
- SIMPLEC diverged on meshes with coherent ~45° skew, such as GAMBIT block meshes around a
  square cylinder: within 12 iterations at Re = 400, and even at Re = 4. A stability guard now
  takes snapshots while the continuity residual is healthy. If the residual becomes non-finite
  or grows 20× above its running minimum, the guard restores the fields and lowers the pressure
  under-relaxation (×0.6, down to 0.1). The running minimum is floored at 1e-6 of the face mass
  flux, so round-off noise in periodic flows is not mistaken for growth. The iteration counter
  is not rewound, so discarded iterations count as work. The guard is only active when
  `relax_p` is automatic, and every rollback is logged and written to the run report. Warped
  triangles (43°) are unaffected and keep α_p = 0.8: the worst face angle alone does not
  predict the instability. The principled fix, non-orthogonal correctors in the p′ equation,
  is not implemented yet.
- The warm-start report's `solve_seconds` now includes the planner's timing iterations.
- The stability guard also rescues the Driver & Seegmiller step with SST. On that orthogonal
  mesh SIMPLEC runs at α_p = 1.0, and the step diverged after 31 iterations (measured). With
  the guard it converges, and x_r/H is within the 10 % acceptance band.
- The k and ε/ω residuals no longer count cells held at the lower bound. In free-stream cells
  where k sits at its floor, the unscaled imbalance never decreases, and the k-ε flat plate
  could not reach its tolerance.

### Changed
- The warm-start study now counts iterations to the **solution**: cold and warm runs are
  counted until |U| (and T) is within 10⁻⁴ (relative, max norm) of the tightly converged cold
  field. The old criterion (the cold run's final absolute residual) stalled when the warm run
  converged to a slightly different round-off level. Fluent-style "iterations to scaled
  residual 10⁻⁵" are still listed.
- `warmstart_proxy` writes its case folders next to the benchmark output and checks that the
  warm run reaches the cold-start solution.
- The GPU guide's SIMPLEC cost-per-iteration table is now generated by
  `benchmarks/simplec_iteration_cost.py`.

### Documentation
- README (English and Persian) and the JOSS paper draft now quote the numbers of the full
  benchmark run committed in `benchmarks/results/`: 24 of 26 benchmarks and 68 of 70
  acceptance checks pass. The two failures, realizable k-ε flat-plate skin friction and the
  cylinder warm start, are analysed in the validation report.
- `benchmarks/warmstart_cylinder_wake.py` shows why the cylinder warm start is slower: the
  pre-solve and target meshes resolve the cylinder differently (12 vs 9 cells per diameter),
  and their converged recirculation lengths differ (1.87 D vs 2.33 D).
- The planner study has been run at the committed code. A1 and A3 pass at all sizes; A2
  fails at 9,216 and 65,536 cells, and the overhead is explained in the study page.

## [0.2.0] - 2026-10-25

### Verification and validation
- `flowtrace benchmark`: a reproducible benchmark suite (`quick` / `full` tiers). It writes
  `results.json`, a self-contained HTML report with plots, a PDF and a Markdown table. Every
  number in the README is produced by it.
- Method of Manufactured Solutions for both solvers (`flowtrace.validation.mms`). Observed
  orders: MAC 2.0 for u, v, p, T; SIMPLEC 1.9–2.0 for velocity/temperature on warped quads and
  triangles; pressure 2.0 on quads and 1.2–1.5 on triangles.
- Grid Convergence Index (Celik et al. 2008) for the natural-convection Nusselt number and the
  cavity centreline minimum.
- Warm-start study (`flowtrace warmstart-study`). It measures the iteration savings of a FlowTrace
  `.ip` initial field with FlowTrace's own SIMPLEC solver on a different mesh, and generates the
  cold/warm Fluent journals plus a PyFluent runner. Fluent results stay placeholders until run
  with a licence. `flowtrace parse-transcript` counts iterations in Fluent transcripts.
- `flowtrace gpu-report`: GPU vs CPU correctness and speed log (measured on an RTX 4060 Laptop
  GPU).

### Physics
- RANS turbulence in the unstructured solver: k-ω SST (Menter 2003, y⁺-insensitive wall
  treatment with Spalding's law), realizable and standard k-ε with scalable wall functions.
  Validated on the Re_τ = 392 channel (DNS), the zero-pressure-gradient flat plate and the
  Driver & Seegmiller backward-facing step.
- Conjugate heat transfer: solid cell zones with conductivity and volumetric sources, a single
  energy equation with harmonic face conductivities (`ConjugateSolver`).
- Moving reference frames (absolute-velocity formulation) and rotating walls.
- Translational periodic zones (Fluent `.msh` section 18 read/write), with a driving body force.
- Compressible-flow design document (`docs/design/compressible-flow.md`); not implemented yet.

### Ansys Fluent interoperability
- CFF/HDF5 `.cas.h5`/`.msh.h5` mesh reader/writer and `.dat.h5` cell-data reader/writer,
  following the public VTK reader's layout. Round-trip tested; not yet verified against
  Fluent-written files.
- Mesh quality report with Fluent-style orthogonal quality, equiangle skewness and aspect
  ratio (`flowtrace quality`, also in `flowtrace mesh`).
- `flowtrace fluent-warmstart`: pre-solve any Fluent mesh/case and write `.ip` (+ `.dat.h5`),
  a journal and a PyFluent script that continue the user's own case.
- Nominative use of the Ansys/Fluent names with a trademark notice in docs and generated scripts.

### Fixed (found by the new verification suite)
- SIMPLEC was inconsistent on triangular and skewed meshes. The momentum equation used a
  least-squares cell pressure gradient that is not the adjoint of the face-flux divergence, and
  MMS errors stalled at O(1). It now uses the Green–Gauss face-pressure force (Fluent's choice),
  which restores second order. The old behaviour is still available as
  `pressure_gradient="least-squares"`.
- Rhie–Chow fluxes now project both pressure gradients on the orthogonal part of the face
  vector.
- Zero-gradient boundary values (pressure at walls, outlet velocity, adiabatic temperature) are
  reconstructed with the cell gradient along the face, which keeps skewed boundary cells
  consistent.
- MAC time step: AB2 with upwind-biased convection plus explicit diffusion needs a combined
  stability limit. Without it the channel case oscillated at the outlet and never converged; it
  now converges in 760 steps.
- Continuity residual normalisation floor, so periodic and warm-started runs report
  convergence correctly.
- SIMPLEC pressure relaxation is now automatic: 1.0 on orthogonal meshes, 0.8 above 5° of
  non-orthogonality. At 1.0, finely resolved warped meshes diverged because the pressure
  correction neglects the non-orthogonal face terms.
- BiCGSTAB stops when the initial residual is already at round-off level. Before, a converged
  outer iteration could make the recurrence coefficients overflow.

### Performance
- Profiled before porting: on SIMPLEC, 74 % of an iteration was the pressure-correction AMG-PCG.
  Its default tolerance is now 0.1 (Fluent's AMG default). Measured outer iterations are
  unchanged and wall time drops 15–50 %.
- Fused CUDA C++ kernels (NVRTC via `cupy.RawModule`) for the MAC convection and Laplacian
  stencils on the GPU: 1.2–1.9× faster per step from 128² up. Equality tests against the
  vectorised CuPy path are included, and that path remains the fallback.
- GPU warm-up in the planner and the GPU report, so first-use kernel compilation is not
  measured as solver time.

### Tooling
- GitHub Actions CI (Windows + Linux, Python 3.9–3.12, ruff, mypy, coverage), pre-commit, MkDocs
  documentation with theory manual, user guide, validation report, API reference and Jupyter
  tutorials; CITATION.cff, CONTRIBUTING, Code of Conduct, issue templates, JOSS paper draft.
- Dashboard: saved projects, report export, light/dark theme, Persian/English interface,
  accessibility improvements.

## [0.1.0]

- First release: hardware probe and planner, MAC structured solver, SIMPLE/SIMPLEC
  unstructured solver, Fluent `.msh`/`.ip`/`.prof`/journal/PyFluent export, web dashboard.
