# GPU log

Produced by `flowtrace gpu-report --out benchmarks/results` on the machine named below. The log
records:

* **Correctness:** MAC and SIMPLEC solvers on CuPy against NumPy in FP64, after a fixed number of
  steps or iterations, plus the divergence of an FP32 run.
* **Speed:** milliseconds per MAC time step for each backend and grid size, after a warm-up
  (best of three blocks). The planner's choice is listed for each grid size. The planner times
  real steps itself.

!!! warning "Precision of the committed log"
    The log below was measured with FlowTrace 0.2.0 on a laptop RTX 4060. Back then the
    GPU ran FP32 by default, so its speed column is **FP32 on the GPU against FP64 on the CPU**.
    That is not an equal-accuracy comparison, and the GPU's FP64 time was not measured. The
    current `flowtrace gpu-report` times every backend in FP64 and lists the GPU's FP32 time
    separately. Precision is now FP64 unless FP32 is shown to meet the case's tolerance (see
    [GPU and the planner](../user-guide/gpu.md#precision)). No GPU was available to re-measure
    the log for this release.

--8<-- "benchmarks/results/gpu_latest.md"

Your hardware will differ. Run the command on the target machine before relying on the planner.
