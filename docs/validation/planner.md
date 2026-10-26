# Backend planner study (unstructured solver)

**Question.** Does the measured planner place the unstructured SIMPLEC solver on the faster
device, and what does the measurement itself cost?

**Hypothesis.** A SIMPLEC iteration on the GPU has a size-independent floor (kernel launches,
host synchronisation for every Krylov dot product). The GPU therefore loses on small meshes
and wins on large ones. Before this study, `backend="auto"` always chose the GPU.

**Method.** Lid-driven cavity at Re = 100 on uniform quads, FP64, a fixed number of SIMPLEC
iterations, so every configuration does the same work. Each run is a fresh process. The timer
covers device initialisation (the CUDA context), mesh generation, solver set-up, the planner's
timing iterations and all remaining iterations. Python imports are excluded. Configurations:

* **CPU**: fixed NumPy backend.
* **GPU**: fixed CuPy backend, which is what `auto` did before.
* **planner**: `auto` with the measured planner.
* **oracle**: the faster of the two fixed runs. It is an experimental reference, not something
  the planner can know at run time.

**Acceptance criteria**, fixed before the first run:

* **A1.** The planner chooses the oracle's device whenever the oracle is more than 15 % faster.
* **A2.** Planner time ≤ 1.15 × oracle time at every size (median of the repeats).
* **A3.** The planner's final field agrees with the fixed run on the same device to within
  10 × the fixed run's own repeat-to-repeat difference (floor 1e-10). In other words, the
  timing iterations change nothing beyond run-to-run noise.

A3 was first written as max |Δu| ≤ 1e-10. A smoke run, made before any study numbers were
taken, showed that two fixed NumPy runs in separate processes already differ by about 6e-8
after 20 iterations. Within one process the solver is bit-exact (unit test
`test_kept_solver_continues_exactly`). The baseline could not meet the original threshold, so
it was replaced before the study was run.

## Results

--8<-- "benchmarks/results/planner_unstructured.md"

## Reproduce

```bash
python benchmarks/planner_unstructured.py --sizes 32,64,96,128,192,256 --iters 100 --repeats 3
```

The machine constants (GPU floor and CPU cost per cell) are measured once per machine and
cached in `~/.flowtrace/planner.json`. Delete that file after a driver or library upgrade.

## Limitations

* Measured on one laptop (RTX 4060 Laptop GPU, FP64). The crossover size depends on the
  CPU/GPU pair, the precision and the physics. Do not transfer these numbers to other
  hardware: run the study there.
* Only 2-D quads were measured. Turbulence and energy add work on both devices. The floor
  argument still holds, because extra physics cannot make a GPU iteration cheaper than a
  laminar one on a tiny mesh, but the crossover moves.
* The planner times about 10 iterations per device. For a run of fewer than ~50 iterations,
  that overhead is a noticeable fraction of the total.
