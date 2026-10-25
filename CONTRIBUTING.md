# Contributing to FlowTrace

Thank you for helping. FlowTrace is used for engineering decisions, so the bar is
**measured correctness**: every change to the numerics comes with a test, and every number
we publish comes from a script in this repository.

## Set up

```bash
git clone <your fork> flowtrace && cd flowtrace
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[cpu,amg,h5,report,dev]"
pre-commit install
```

A GPU is optional. GPU code paths are also tested with a NumPy-backed CuPy stand-in
(`tests/fake_gpu.py`). With an NVIDIA GPU, `pip install -e ".[gpu]"` enables
`tests/test_gpu_real.py`.

## Before you open a pull request

```bash
ruff check flowtrace tests        # lint
mypy                              # typed core modules (see pyproject.toml)
pytest -q                         # ~2 minutes on a laptop
flowtrace benchmark --tier quick  # if you touched a solver: the verification suite
```

* **Numerics changes.** Run the manufactured-solution benchmarks (`mms_*`). The observed order
  must not drop. Explain any change in a validated number in the pull request, and rerun
  `flowtrace benchmark --tier full` if it moves a number quoted in the README.
* **New physics.** Include a verification case with a known answer (exact, manufactured or
  analytic), a validation case against published data with its citation, and documentation in
  `docs/theory/`.
* **Fluent interoperability.** Never claim compatibility you have not checked. Write "round-trip
  tested" unless a file was read back into Ansys Fluent, and say which release.
* **Performance claims.** Measure first (`cProfile`, `flowtrace gpu-report`), and keep the
  Python/NumPy fallback plus an equality test whenever you add compiled or CUDA code.

## Style

* Code reads like the surrounding code: NumPy-style vectorised kernels, docstrings that state
  the equations and their sources, comments only where the *why* is not obvious.
* `ruff` enforces formatting basics (line length 120).
* Public behaviour stays backward compatible within a minor version. In particular, CLI flags
  are never renamed without a deprecation period.

## Reporting problems

Use the issue templates: bug report, validation discrepancy, feature request. For a wrong
result, please attach the case (mesh + `flowtrace solve` command or case JSON), the expected
value with its source, and the output of `flowtrace probe`.

## Trademarks

Ansys and Fluent are trademarks of ANSYS, Inc. Use the names only to state compatibility
("reads Ansys Fluent mesh files"), never to imply endorsement, and do not add Ansys logos to
the project.

## Licence

By contributing you agree that your contribution is released under the MIT licence of this
project.
