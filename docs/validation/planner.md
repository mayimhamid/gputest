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

**Reading the result.** The study rejects the planner as specified: A2 fails at two of six
sizes. A1 and A3 hold everywhere. The planner always ends on the right device, and its timing
iterations do not change the solution beyond run-to-run noise. What fails is the cost of
deciding. In `planner_unstructured.json` (`plan_seconds` of the planner runs):

* **9,216 cells:** the CPU is not fast enough to rule out the GPU from the cached floor, so the
  planner builds and times a solver on both devices. That phase took 2.2–2.4 s, as long as the
  whole 100-iteration CPU run (2.3 s). It then correctly kept the CPU (Numba, 13–16 ms per
  iteration against 21–25 ms on the GPU).
* **65,536 cells:** the planner spends ten CPU timing iterations (about 0.13 s each, 1.3 s in
  total) before it moves to the GPU. Planning took 3.7 s in all. A fixed GPU run of the same 100
  iterations takes 4.0 s, the planner 5.8 s (×1.43).

Compared with the old `auto`, which always chose the GPU, the planner is 2.4–2.9× faster below
5,000 cells and 1.13–1.43× slower above 30,000 cells. Both overheads are fixed costs. At
65,536 cells, a 2,000-iteration run at the measured 25 ms per iteration takes about 50 s, so
the extra 1.7 s would be about 3 %. This study measures 100 iterations, though, and by its own
criteria the planner is not yet good enough.

## Changes after the study (not yet re-measured)

* **9,216 cells: implemented, unmeasured.** The planner now skips the GPU when it cannot pay
  for itself. Even at its floor cost, the GPU saves at most (CPU time − floor) per iteration.
  Over the iterations left, that must exceed the GPU's measured set-up time (`gpu_setup_ms`,
  added to the cached calibration). At 9,216 cells the CPU iteration (13–16 ms) is already below
  the 16.6 ms floor, so the GPU would no longer be built and timed. That should remove the
  2.2–2.4 s planning phase, but the study has **not been re-run on a GPU**: none was available
  for this release. Until it is, A2 stands as failed. Unit tests check the rule's arithmetic
  (`tests/test_planner.py`).
* **65,536 cells: not addressed.** The ten CPU timing iterations before the move to the GPU
  remain. Possible fixes, not implemented: predict the CPU cost from the cached cost per cell
  before timing the CPU, and stop timing once one device leads by a wide margin.

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
