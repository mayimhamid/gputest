# FlowTrace release readiness, October 2026

**Final status: `READY WITH DOCUMENTED LIMITATIONS`** (provisional: confirmed or revised when the full validation tier, now running on the release code, is pushed to this PR)

This report records what was checked before tagging the next FlowTrace release, how it was
checked, what was found and fixed, and what is still open. Every number in it comes from a
command listed in [section 5](#5-commands-run-and-their-outcomes). Results that this
environment could not produce (GPU, Fluent, Windows) are marked as not run; none is estimated.

## 1. Decision

The core supported workflows are reliable on the evidence gathered here: CPU solves through the
CLI, the Python API and the dashboard, Fluent mesh and data exchange, and the verification and
validation suite. None of the open issues is a known wrong result in a supported workflow. The
release is **not** `READY FOR RELEASE`, because the following remain open:

* external Fluent verification: Fluent has not read a FlowTrace-written file, and the Fluent
  column of the warm-start study has not run;
* the GPU paths and the planner's GPU-side rules, which could not be run without a GPU;
* the documented planner overhead, warm-start wall-time and model deviations (section 7);
* CI, which has never been green and whose fix is unverified until it runs.

It is not `NOT READY FOR RELEASE` either. No critical numerical-correctness, data-integrity,
installation or runtime issue is known to remain. The ones found during this work are fixed and
tested: the reversed 3-D Fluent faces, the CG exception on diverging runs, the false skewness
alarms, the k-ε wall-function constants, and the unchecked FP32 default.

The version in `flowtrace/__init__.py` is still 0.3.0. The `[Unreleased]` section of the
changelog includes a breaking CLI change (`ui --host` needs `--allow-remote`), which matters for
choosing the next version number. That choice is the maintainer's.

## 2. Starting point and environment

* **Code.** Branch `claude/busy-noether-ug0mat` at `b120c45`, which is the merge of PR #1 into the
  default branch `claude/beautiful-feynman-i198h1`. The working tree was clean at the start.
* **Machine.** Linux container (kernel 6.18), Intel Xeon @ 2.80 GHz, 4 cores, 15 GB RAM.
  Python 3.13.17 with NumPy 2.5.3, SciPy 1.18.1, Numba 0.68.0, pyamg 5.3.0, h5py 3.16.0,
  matplotlib 3.11.2; g++ 13.3 for the native kernels. Python 3.9.25 (NumPy 2.0.2, SciPy 1.13.1)
  through `uv` for the oldest supported version.
* **Not available here.** No GPU or CUDA: the GPU paths ran only against the NumPy-backed
  CuPy stand-in (`tests/fake_gpu.py`). No Ansys Fluent licence. No Windows or macOS. CI was not
  run: nothing was pushed.
* **Baseline** (before any change):
  * `pytest` 163 passed, 30 skipped (native kernels);
  * `FLOWTRACE_NO_NATIVE=1 pytest` 154 passed, 39 skipped;
  * `ruff` and `mypy` clean;
  * `flowtrace benchmark --tier quick` 21/21 benchmarks passed;
  * committed full-tier table: 68 of 70 checks pass (FlowTrace 0.2.0, Windows laptop with an
    RTX 4060). Failing: `flat_plate_kepsilon` (−6.97 % against a 6 % criterion) and
    `warmstart_proxy`, cylinder (×0.645 against > 1);
  * committed planner study: criterion A2 fails at 9,216 and 65,536 cells.

## 3. Acceptance gates

### Gate A: numerical integrity — met

| Criterion | Evidence |
|---|---|
| Diagnostic definitions match the assembled systems | The diagnostics evaluate the exact matrices and right-hand sides of each iteration, which `step()` keeps by reference: the assembled and the under-relaxed momentum systems, and the pressure-correction system. Identities checked by tests: b_r − A_r u = b − A u at the old field, and corrected cell imbalance = m* + A_p p′ (minus the pressure-correction residual). A bit-for-bit comparison of 64 arrays over six configurations (native and NumPy paths) shows the iteration itself is unchanged. Definitions in `docs/theory/diagnostics.md`. |
| New diagnostics have tests | 17 tests in `tests/test_diagnostics.py` (definitions, zero and non-finite policy, row exclusion, rolled-back iterations, divergence report, CG conditions); CPU/fake-GPU agreement on the same captured iteration (`tests/test_gpu_paths.py`); end to end through events, result, log and report (`tests/test_server.py`). |
| Conservation and manufactured-solution checks are reproducible | On the final code, `tests/test_verification.py` (manufactured solutions and observed orders), the conservation tests (mass balance, CHT energy balance, periodic and rotating-zone exact solutions) and the diagnostics' continuity identities pass in both suites. At the baseline the quick tier passed 21/21 on this machine; the committed full tier passes every verification check. The full tier on the release code is running. |
| Known failures and unsupported cases remain visible | The committed run's two failing checks (`flat_plate_kepsilon`, `warmstart_proxy` cylinder) stay in the benchmark with their criteria unchanged and are analysed on the validation pages. Unsupported and unverified cases are listed in the README limitations and in section 7. |

### Gate B: runtime integrity — met for the CPU; the GPU side is unverified (no GPU)

| Criterion | Evidence |
|---|---|
| CPU-only operation works | Whole test suite on a machine without a GPU, on both the native-kernel and the NumPy reference path (section 5). Clean Python 3.9 installation of the wheel without extras: `flowtrace quality`, `flowtrace solve --export` and the Python API all work. |
| Optional GPU paths fail safely when unavailable | No CUDA here: `probe` reports `Backends numpy`, the planner records "GPU unavailable", `gpu-report` writes "GPU path not exercised", and `test_gpu_real.py` skips. An explicit `--backend cupy` raises `BackendUnavailable`, reported as a one-line error with exit status 2. A GPU the case does not fit on falls back to the CPU (`test_memory_budget_falls_back_to_the_cpu_then_refuses_with_advice`). |
| Planner decisions are explainable and measured | Every plan lists its placements with a measured reason: precision, solver, device, pressure method (with the CG conditions checked on the assembled matrix) and the GPU skip reason. FP32 is chosen only when a case passes the documented accuracy test (`benchmarks/precision_fp32.md`). The CPU-only planner study is queued after the full tier. The GPU side of the planner study could not be re-measured here (no GPU). |
| No material performance regression without explanation | Diagnostics cost 0.2–0.3 of an iteration per evaluation, evaluated every ≥ 10 iterations (≤ 3 %). Changes made on purpose: FP64 instead of an untested FP32 on consumer GPUs (accuracy before speed; FP32 is still chosen where it is admissible); refusal of cases whose upper-bound memory estimate exceeds free memory (override `FLOWTRACE_MEMORY_CHECK=off`). Both are in the changelog. The baseline timings on this machine were taken while other jobs ran, so an A/B timing of five benchmarks on the starting and the final tree is queued for a quiet machine. |

### Gate C: release integrity — met with limitations (CI not yet run on the changes; full tier pending)

| Criterion | Evidence |
|---|---|
| Tests, packaging and documentation checks run where supported | `ruff`, `mypy`, both test suites on the final code (207 passed / 30 skipped with native kernels; 198 / 39 on the reference path), the Python 3.9 suite, `python -m build` and `twine check` (PASSED), a clean-install smoke test, `mkdocs build --strict` (passes on the final docs), and a browser check of the dashboard. CI could not be run from here. Its history shows a single environmental failure, now fixed but unverified. |
| The validation report records failures and environmental limits | `docs/validation/index.md` and the generated report show every failing check with its criterion. Environmental limits (no GPU, no Fluent, single machine) are stated on the validation, planner, GPU and warm-start pages and in this report. |
| Installation and a basic example work from a clean environment | Fresh Python 3.9 virtual environment, `pip install` of the built wheel (dependencies NumPy and SciPy only), then `flowtrace --version`, `--help`, `probe`, `quality`, `solve --energy --export` (converged, `.msh`/`.ip`/`.jou`/`.py`/`.csv` written) and a Python-API cavity run. Repeated on the final code: every step passed. |
| Public claims match the available evidence | README (English and Persian), docs home page, architecture document and user guides were audited against the code and the results. Corrected stale or unsupported statements, among them "compressible flow is a design proposal", "no radiation", "steady RANS only", "HDF5 not checked against Fluent", the FP32 GPU speed-up presented without its precision caveat, "memory estimate is not enforced", and "launcher tested in CI". |

## 4. Files changed and why

**Solver and numerics**

| File | Change | Why |
|---|---|---|
| `flowtrace/unstructured/diagnostics.py` (new) | `evaluate`, `summary`, `pressure_matrix_properties` | Phase 1: componentwise backward errors of the exact systems, linear-solver reduction, local and global continuity, field change, health; CG conditions |
| `flowtrace/unstructured/simple.py` | `step()` keeps references to its systems; `diagnostics()`; `guard_exhausted_at`; `describe()` reports it | Diagnostics on the real systems without copying; divergence reporting |
| `flowtrace/unstructured/fvmesh.py` | `matrix_like()` | Build \|A\| with the solver's own sparsity pattern (native or SciPy/CuPy) |
| `flowtrace/unstructured/conjugate.py` | `diagnostics()` for the fluid part | Same diagnostics for conjugate cases |
| `flowtrace/unstructured/turbulence.py` | `KAPPA_WALL = 0.4187` in the k-ε wall functions | Fluent's κ/E pair; measured improvement on flat plate and channel (Phase 2.2) |
| `flowtrace/linalg/krylov.py` | CG breakdown is reported, not raised | `ZeroDivisionError` on diverging runs hid the divergence handling (found by the Python 3.9 run) |
| `flowtrace/structured/solver.py` | `auto` precision is FP64; `fp32_round_off_bound`, `fp32_admissible` | No FP32 without an accuracy test (Phase 3.1) |

**Planner, hardware, runtime**

| File | Change | Why |
|---|---|---|
| `flowtrace/hardware/planner.py` | precision decision with reasons; FP32 candidate only if admissible; GPU set-up pay-back rule; `gpu_setup_ms` calibration; rebuilt solver keeps the timed precision | Phase 3.1 and 3.2 |
| `flowtrace/hardware/memory.py` | measured-overhead term; actionable refusal; `FLOWTRACE_MEMORY_CHECK=off` | Estimate was 1.9-3.3× below the measured peak (Phase 3.3) |
| `flowtrace/hardware/gpu_report.py` | every backend timed in FP64, GPU also in FP32 (labelled) | Equal-precision comparison; keeps the report meaningful after the precision change |
| `flowtrace/runtime/engine.py` | diagnostics in events, result, log; CG conditions in the plan; unknown zones refused; divergence context; one-line logging of input errors and divergence | Phases 1.5, 2.3, 5 |
| `flowtrace/runtime/batch.py` | diagnostics summary per case | Phase 1.5 |
| `flowtrace/cli.py` | one-line errors and exit statuses 2/1/130; `ui --allow-remote`; `solve` checks zone names up front | Phase 5 (error messages, input validation, dashboard exposure) |
| `flowtrace/ui/server.py` | Host/Origin checks (403); diagnostics table in the report; FP32 item size in the memory estimate; static-file containment | Phase 5 (local-only security), Phase 1.5 |

**Fluent interoperability and mesh quality**

| File | Change | Why |
|---|---|---|
| `flowtrace/mesh/unstructured.py` | `reverse_face_nodes` | Fluent's 3-D face convention |
| `flowtrace/mesh/fluent_msh.py`, `fluent_h5.py` | reverse 3-D faces on read and write; parse failures (`.msh`, `.cas`, `.cas.h5`, `.dat.h5`) become `FluentMeshError` | Measured on Fluent-written files; malformed files (Phase 4) |
| `flowtrace/mesh/quality.py` | skewness only for standard element shapes | False CRITICAL on Fluent's own polyhedral mesh |
| `flowtrace/validation/fluent_files.py` (new) | checks against the PyFluent mixing elbow | Phase 4: an independently produced compatibility fixture |
| `flowtrace/fluent/warmstart_study.py`, `validation/benchmarks.py` | wall-clock timing on equal terms; informational metric | Phase 2.2: separate iteration and wall-clock savings |

**Tests** (all new tests are regression tests for the changes above)

| File | Added |
|---|---|
| `tests/test_diagnostics.py` (new) | 19 tests: definitions, identities, zero/non-finite policy, row exclusion, rolled-back iteration, divergence report, CG conditions, one-line job errors, CLI backend error |
| `tests/test_fluent_files.py` (new) | Fluent-written files (opt-in, CI job) |
| `tests/test_gpu_paths.py` | diagnostics on the fake GPU; precision rule; gpu-report labels |
| `tests/test_planner.py` | pay-back rule; set-up calibration; memory fallback, refusal and override |
| `tests/test_mesh.py` | 3-D orientation (`.msh`, `.cas.h5`, hand-written Fluent hexahedron); 8 malformed `.msh` files; malformed CFF files |
| `tests/test_ansys_io.py` | skewness on polyhedra; unknown zone in `fluent-warmstart` and `solve` (exit 2, one line) |
| `tests/test_server.py` | diagnostics end to end; Host/Origin rules; live 403 checks; static containment; `ui --host` refusal |
| `tests/test_hardware.py` | memory estimate is an upper bound |
| `tests/test_batch.py` | checkpoint restart from disk |
| `tests/test_linalg.py` | CG breakdown |

**Benchmarks, CI, packaging, docs**

| File | Change |
|---|---|
| `benchmarks/precision_fp32.py`, `skew_envelope.py`, `memory_estimate.py`, `fluent_files_check.py` (new) and `benchmarks/results/*` | Measurements behind the precision rule, the skew envelope, the memory bound and the Fluent checks |
| `benchmarks/planner_unstructured.py` | `--modes`/`--name` for machines without a GPU; time breakdown per run |
| `benchmarks/results/{results.json, validation_table.md, report.html, report.pdf}`, `benchmarks/fluent_warmstart/results.*` | To be regenerated by `flowtrace benchmark --tier full` on the final code (running; pushed to the PR when done) |
| `.github/workflows/ci.yml` | Linux Python 3.9 cell on `ubuntu-24.04` (the only cause of the red CI); Python 3.13 in the matrix; `fluent-files` job |
| `pyproject.toml`, `mkdocs.yml` | `diagnostics.py` type-checked; diagnostics page in the nav |
| `README.md`, `README.fa.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `docs/**`, `paper/paper.md` | Synchronised with the code and the measurements (stale statements corrected, a stray control character in `docs/install.md` removed); this report |

## 5. Commands run and their outcomes

All commands were run in `/home/user/gputest` on the machine in section 2. Times are wall-clock
times on that machine, often with other jobs running (noted where it matters).

| Step | Command | Outcome |
|---|---|---|
| Baseline lint and types | `ruff check flowtrace tests`; `mypy` | clean; clean (8 files) |
| Baseline tests | `python -m pytest -q` | 163 passed, 30 skipped (439 s) |
| Baseline reference path | `FLOWTRACE_NO_NATIVE=1 python -m pytest -q` | 154 passed, 39 skipped (554 s) |
| Baseline quick tier | `flowtrace benchmark --tier quick --out <scratch>` | 21 of 21 benchmarks pass |
| k-ε diagnosis | flat plate with κ = 0.41 / 0.4187 at four first-cell heights; channel Re_τ = 392 with both κ (scratch scripts around `validation.turbulent`) | reproduced −6.97 %; −4.96 % with κ = 0.4187; y⁺ sweep and channel numbers in `docs/theory/turbulence.md`; standard k-ε diverged on the plate |
| Skew envelope | `python benchmarks/skew_envelope.py` | `benchmarks/results/skew_envelope.md` (θ = 0-60°, seven relaxation settings) |
| FP32 study | `python benchmarks/precision_fp32.py` | `benchmarks/results/precision_fp32.md`: FP32 reached tolerance in 1 of 6 cases |
| Memory study | `python benchmarks/memory_estimate.py` (twice) | `benchmarks/results/memory_estimate.md`: upper bound in all 11 cases |
| Fluent-written files | `python benchmarks/fluent_files_check.py` | 7 of 7 properties pass (`benchmarks/results/fluent_files_check.md`) |
| Polyhedral quality | `flowtrace quality ~/.flowtrace/fluent_examples/mixing_elbow.cas.h5` | exit 2 (false CRITICAL) before the fix, exit 0 after |
| Diagnostics do not change the iteration | bit-for-bit comparison of 64 arrays, six configurations, native and NumPy paths | identical |
| Cylinder warm-start test | pre-registered matched/mismatched runs (8 and 12 cells/D) through `fluent.warmstart_study.run_proxy_case` | 8 cells/D: ×1.58 (matched) and ×1.18 (coarser target); 12 cells/D matched: ×2.04 (558 → 274) |
| Python 3.9 test run | `uv venv --python 3.9`; `uv pip install -e ".[cpu,amg,h5,report]" pytest`; `python -m pytest -q` | 193 passed, 4 failed, 30 skipped: three `ZeroDivisionError` (fixed, F7) and one outdated test expectation (updated); the three re-run and pass |
| Malformed CFF files | `flowtrace quality` on an HDF5 file without CFF groups, a partial CFF file and a non-HDF5 file | before: one-line error, traceback (exit 1), one-line error; after: one-line errors, exit 2 for all three |
| Interim suites | `python -m pytest -q`; `FLOWTRACE_NO_NATIVE=1 python -m pytest -q` (before the last changes) | 204 passed, 30 skipped; 195 passed, 39 skipped |
| Final lint and types | `ruff check flowtrace tests benchmarks`; `mypy` | clean; clean (9 files) |
| Final tests, native kernels | `python -m pytest -q -rfEs` | **207 passed, 30 skipped** (all skips: no CUDA GPU), 0 failed; 1,931 s on a busy machine |
| Final tests, reference path | `FLOWTRACE_NO_NATIVE=1 python -m pytest -q` | **198 passed, 39 skipped**, 0 failed; 1,934 s |
| Package | `python -m build`; `twine check dist/*` | sdist and wheel built; both PASSED |
| Clean install | fresh Python 3.9 venv; `uv pip install dist/flowtrace-0.3.0-py3-none-any.whl`; `flowtrace --version`, `probe`, `quality`, `solve --energy --export`, `solve --bc nosuchzone=wall`, Python API cavity | all steps pass on the final wheel: `quality` exit 0; `solve` converged and exported `.msh`/`.ip`/`.jou`/`.py`/`.csv`; misspelt zone: exit 2, one line; API cavity converged in 286 iterations, diagnostics finite |
| Dashboard in a browser | Playwright/Chromium against `start_background()`, on `127.0.0.1` and `localhost`: run a case, read the legend, collect console errors and 403s | both hosts: run accepted, job finished, no console errors, no 403 |
| Full validation tier | `flowtrace benchmark --tier full --out benchmarks/results` | **running** on the final code; results, report and README numbers will be pushed to the PR |
| Planner on a CPU-only machine | `python benchmarks/planner_unstructured.py --sizes 32,64,96,128,192,256 --iters 100 --repeats 3 --modes numpy,auto --name planner_unstructured_cpu_only` | queued after the full tier (needs a quiet machine) |
| Documentation | `mkdocs build --strict` (Python 3.12 venv with `.[docs]`) | exit 0 on the final docs (126 s) |
| GitHub CI history | `actions_list` on the repository (read-only) | all 5 runs red, each for one reason: `test (ubuntu-latest, 3.9)` fails in `actions/setup-python` (no 3.9 for Ubuntu 26.04), so `package` and `benchmarks-quick` never ran; every other job green |
| Not run | GPU tests on hardware, Fluent runtime, Windows/macOS, CI on the changes | no GPU, licence or other OS here; CI runs once the branch is pushed |

## 6. Verified fixes and their evidence

| # | Fix | Evidence |
|---|---|---|
| F1 | **k-ε wall-function constants** (κ 0.41 → 0.4187 with E = 9.793). | Pre-registered: the change was predicted to remove part of the flat-plate deficit. Realizable k-ε flat plate −6.97 % → −4.96 % against Schultz-Grunow (criterion 6 %); channel Re_τ = 392: standard 3.52 → 1.87 %, realizable 7.57 → 5.84 %. y⁺ sweep in `docs/theory/turbulence.md`. Full tier: the official re-run is in progress. |
| F2 | **FP32 only when admissible**; `auto` = FP64. | `benchmarks/results/precision_fp32.md`: FP32 missed the steady tolerance in 5 of 6 cases. `test_auto_precision_is_fp64_unless_fp32_is_shown_admissible`. |
| F3 | **Fluent 3-D face orientation** (read and write). | `benchmarks/results/fluent_files_check.md`: 0 of 91,581 faces re-oriented on read; written file reproduces Fluent's node order and c0/c1 for 91,581 / 91,581 faces. Tests in `tests/test_mesh.py`. |
| F4 | **Skewness on polyhedra.** | `flowtrace quality mixing_elbow.cas.h5` exited 2 (false CRITICAL) and now exits 0; `test_skewness_is_not_applied_to_polyhedral_cells`. |
| F5 | **Malformed meshes → `FluentMeshError`.** | Eight parametrised malformed-file tests (`tests/test_mesh.py`). |
| F6 | **Memory estimate is an upper bound; refusal is actionable.** | `benchmarks/results/memory_estimate.md`: estimate/peak 1.31-1.56 (tracemalloc) and 1.23-2.10 (resident set, two runs: 1.30-2.09 and 1.23-2.10) for SIMPLEC; `test_unstructured_memory_estimate_is_an_upper_bound`, `test_memory_budget_falls_back_to_the_cpu_then_refuses_with_advice`. |
| F7 | **CG breakdown reported instead of raised.** | Python 3.9 / NumPy 2.0: 3 tests failed with `ZeroDivisionError`, and pass after the fix; `test_pcg_breakdown_is_reported_not_raised` reproduces the old exception. |
| F8 | **Dashboard: DNS rebinding and cross-site requests refused**; non-loopback host needs `--allow-remote`. | `test_request_allowed_rules`, `test_server_refuses_dns_rebinding_and_cross_site_requests` (forged `POST /api/run` → 403, no job started), `test_ui_refuses_a_network_host_without_allow_remote`. A Playwright/Chromium session on `127.0.0.1` and on `localhost` ran a case with no 403 and no console errors. |
| F9 | **CLI and job errors in one line with exit statuses.** | `test_fluent_warmstart_rejects_unknown_zone`, `test_solve_rejects_unknown_zone_before_running`, `test_job_reports_divergence_and_input_errors_in_one_line`, `test_cli_reports_an_unavailable_backend_in_one_line`. |
| F10 | **Unknown boundary zones refused** (were silently ignored). | Same tests as F9; engine check for the dashboard. |
| F11 | **gpu-report compares equal precision.** | `test_gpu_report_compares_equal_precision` (fake GPU). Not run on a real GPU. |
| F12 | **Warm-start wall time measured on equal terms.** | `ProxyResult.wallclock_ratio`; full-tier metric (no criterion): values come with the full-tier re-run. |
| F13 | **Divergence report names the cause.** | `test_divergence_report_names_the_guard_state_and_the_skew`. |
| F15 | **Malformed CFF files → `FluentMeshError`** (were a `KeyError` traceback). | `test_malformed_cff_files_raise_a_clear_error`; `flowtrace quality partial.cas.h5` exits 2 with one line. |
| F16 | **CI matrix: Linux Python 3.9 on `ubuntu-24.04`.** | The `actions/python-versions` manifest has 3.9.25 for Ubuntu 22.04 and 24.04 only; unverified until CI runs. |
| F14 | **Rebuilt structured solver keeps the timed precision; FP32 memory estimate uses 4-byte values.** | Code review; covered by `test_auto_precision_is_fp64_unless_fp32_is_shown_admissible` and the server end-to-end test. |

## 7. Unresolved issues

Severity: **high** means it can give a wrong answer or block a supported workflow; **medium** means a
claim is unverified or performance is below its own target; **low** means a limitation with a
workaround or a cosmetic issue. None of the open issues is a known wrong result in a supported workflow.

| # | Issue | Severity | Impact | Reproduce / evidence |
|---|---|---|---|---|
| U1 | Fluent has not read any file FlowTrace wrote (`.msh`, `.cas.h5`, `.dat.h5`, `.ip`), and the Fluent column of the warm-start study has not run. | medium | Fluent-side compatibility of the writers rests on the face-by-face comparison with Fluent's own files (F3), not on Fluent itself. | Needs a licence: `flowtrace warmstart-study` and the journals in `ws/`. |
| U2 | GPU paths not run on real hardware for this release: SIMPLEC/MAC kernels, the precision rule's timing, the planner pay-back rule, gpu-report, the CUDA HLLC kernel. | medium | GPU behaviour is covered only by the fake-GPU tests and by the earlier laptop logs. | `pytest tests/test_gpu_real.py`, `flowtrace gpu-report`, `python benchmarks/planner_unstructured.py` on a CUDA machine. |
| U3 | Planner overhead: A2 failed at 9,216 and 65,536 cells in the committed GPU study. The pay-back rule addresses 9,216 cells but is unmeasured; 65,536 cells is not addressed. | medium | Up to about 1.5× on 100-iteration runs at those sizes; about 3 % on a 2,000-iteration run. | `docs/validation/planner.md`. |
| U4 | Warm start: no wall-time saving on the five small 2-D proxy cases, and the cylinder case loses iterations (×0.65 on the study's mesh pair; ×2.04 when the meshes resolve the cylinder alike). | medium | The benefit claim is limited to iterations, and to meshes that resolve the geometry alike. | `flowtrace benchmark --only warmstart_proxy`; `docs/validation/warmstart.md`. |
| U5 | No non-orthogonal correctors in the p′ equation. The largest stable α_p falls with cos²θ. | low (the automatic guard handles the measured envelope) | A user-set α_p above cos²θ diverges on skewed meshes; the error message now says so. | `python benchmarks/skew_envelope.py`. |
| U6 | Standard k-ε diverged (overflow) on the flat-plate set-up, with κ = 0.41 and with 0.4187. | low | Use realizable k-ε or SST for flat plates (both validated). | `flowtrace.validation.turbulent.run_flat_plate("k-epsilon-standard")`; a re-check on the final code is queued. |
| U7 | Pressure converges at order 1.2 on triangles (velocity and temperature 2.0). | low | Documented, threshold 1.0. | `flowtrace benchmark --only mms_simplec_tri`. |
| U8 | The first-iteration momentum residual from rest divides by a 1e-300 floor (~1e302). | low (cosmetic; the convergence test starts at iteration 6) | Large first value in residual plots. | Cavity from rest, `SimpleSolver.residuals()` at iteration 1. |
| U9 | Memory: the GPU pool overhead is unmeasured, and the conservative refusal can block a case that would fit. | low | `FLOWTRACE_MEMORY_CHECK=off` overrides. | `benchmarks/memory_estimate.py`. |
| U10 | Checkpoint restart only through the Python API; the guard's lowered `relax_p` is not saved. | low | A restarted run re-tunes α_p. | `tests/test_batch.py::test_checkpoint_on_disk_restart_reaches_the_same_answer`. |
| U11 | Mixed precision (FP32 storage, FP64 accumulation) is not implemented; the FP32 rule uses a round-off bound, not a measured error of the case at hand. | low | Consumer GPUs run FP64 for most steady cases. | `benchmarks/precision_fp32.md`. |
| U12 | CI has never been green; the fix (Linux 3.9 on `ubuntu-24.04`) and the new cells (`fluent-files`, Python 3.13) have not run yet. Windows launcher and macOS not tested. | medium | `package` and `benchmarks-quick` have never run in CI; platform-specific failures would show only there or on user machines. | The PR's CI run; `start_flowtrace.bat` with `FLOWTRACE_LAUNCHER_DRYRUN=1`. |
| U13 | Compressible solver: no FFF-7 comparison with Fluent, no full rocket-mesh run (data not in the repository). | medium | Verified on canonical cases only. | `docs/theory/compressible.md#not-yet-validated`. |
| U14 | 3-D meshes written by FlowTrace ≤ 0.3.0 have every face reversed for Fluent. | low (re-export fixes it) | Users with such files should re-export them. | Read the old file: the quality report notes re-oriented faces. |
| U15 | The full validation tier has not yet been re-run on the release code; `benchmarks/results/` and the README's validation numbers are still those of the committed FlowTrace 0.2.0 run. | medium (process) | The k-ε numbers in the committed table predate the fix. | `flowtrace benchmark --tier full --out benchmarks/results` (running; results will be pushed to the PR). |

## 8. Commit message

The changes were committed with this message (the full-tier results follow in a separate commit):

```text
Release readiness: solver diagnostics, precision rule, Fluent interop fixes, CI fix

Diagnostics and numerics
- SimpleSolver.diagnostics(): componentwise backward errors of the exact momentum and energy
  systems before and after the linear solve, the true linear-solver reduction, predictor and
  corrected local and global continuity, field change and health; in events, results, the log
  and the run report
- CG reports a breakdown instead of raising ZeroDivisionError on diverging runs
- k-epsilon wall functions use Fluent's kappa = 0.4187 with E = 9.793 (flat plate -6.97 % ->
  -4.96 % vs Schultz-Grunow; channel standard 3.5 -> 1.9 %, realizable 7.6 -> 5.8 % vs DNS)
- a diverged run reports the mesh non-orthogonality and the stability guard's state

Planner and hardware
- precision "auto" is FP64; FP32 only for steady cases whose round-off bound is within
  steady_tol (FP32 missed the tolerance in 5 of 6 measured cases); reasons in the plan
- gpu-report times every backend in FP64 and the GPU also in FP32
- a GPU candidate is not timed when it cannot repay its measured set-up time
- the memory estimate is an upper bound (tracemalloc and resident set); actionable refusal
  with a FLOWTRACE_MEMORY_CHECK=off override

Fluent interoperability
- 3-D faces follow Fluent's orientation on read and write (.msh, .cas.h5); checked against
  the Fluent-written PyFluent mixing elbow (7 of 7 properties)
- skewness only for standard element shapes; malformed .msh/.cas/.cas.h5/.dat.h5 files raise
  FluentMeshError

Reliability and security
- the dashboard refuses DNS-rebinding and cross-site requests; ui --host needs --allow-remote
- one-line CLI and job errors with exit status 2 (input) or 1 (run failed); unknown boundary
  zones are refused

CI, benchmarks, docs
- CI: the Linux Python 3.9 cell runs on ubuntu-24.04 (the only cause of the red CI); Python
  3.13 in the matrix; fluent-files job
- benchmarks: precision_fp32, skew_envelope, memory_estimate, fluent_files_check, with results;
  planner study gains a CPU-only mode
- README (English and Persian), ARCHITECTURE, CHANGELOG, user guides, validation pages,
  RELEASE_READINESS.md

Tests: 207 passed / 30 skipped (native kernels), 198 / 39 (reference path), Python 3.9 suite
passes. Not run here: GPU hardware, Fluent, Windows/macOS. The full validation tier is being
re-run on this code and will follow in a separate commit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013Qb2f5onmDVNaPa8DLr8e6
```
