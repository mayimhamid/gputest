# Changelog

All notable changes to FlowTrace. The project follows [Semantic Versioning](https://semver.org/):
MAJOR for incompatible API/CLI changes, MINOR for new features, PATCH for fixes.
Until 1.0.0, minor versions may still change the Python API; the CLI stays backward compatible.

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
