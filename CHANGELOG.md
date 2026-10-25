# Changelog

All notable changes to FlowTrace. The project follows [Semantic Versioning](https://semver.org/):
MAJOR for incompatible API/CLI changes, MINOR for new features, PATCH for fixes.
Until 1.0.0, minor versions may still change the Python API; the CLI stays backward compatible.

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
