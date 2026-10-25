# GPU log

Produced by `flowtrace gpu-report --out benchmarks/results` on the machine named below. The log
records:

* **Correctness:** MAC and SIMPLEC solvers on CuPy against NumPy in FP64, after a fixed number of
  steps or iterations, plus the divergence of an FP32 run.
* **Speed:** milliseconds per MAC time step for each backend and grid size, after a warm-up
  (best of three blocks). The planner's choice is listed for each grid size. The planner times
  real steps itself.

--8<-- "benchmarks/results/gpu_latest.md"

Your hardware will differ. Run the command on the target machine before relying on the planner.
