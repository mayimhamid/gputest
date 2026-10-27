# GPU and the hardware planner

## Backends

| Backend | Device | What runs there |
|---|---|---|
| `numpy` | CPU | vectorised NumPy/SciPy, multi-threaded FFTs |
| `numba` | CPU, all cores | fused JIT stencil kernels (MAC solver), JIT AMG aggregation |
| `cupy` | NVIDIA GPU | every field resident in VRAM, cuFFT/cuBLAS/cuSPARSE, fused CUDA C++ stencils |

With `backend="auto"` the **planner measures** instead of assuming the GPU is faster. It
builds the real solver on each backend that fits in memory, times real steps after a warm-up,
and keeps the fastest. This covers both the structured MAC solver and the unstructured
SIMPLE/SIMPLEC solver (dashboard, `flowtrace solve`, `flowtrace fluent-warmstart`). Every
decision and its measured reason appear in the dashboard log and the run report. (The bare
`get_backend("auto")` used by library code without a case still picks GPU, then Numba, then
NumPy.)

## Precision

GeForce and RTX cards run FP64 at 1/32 to 1/64 of their FP32 speed, so FP32 is tempting on
them. But FP32 is a different accuracy, and a speed-up bought with it is not like-for-like.
FlowTrace therefore **runs FP64 unless FP32 is shown to meet the case's own accuracy target**:

* `precision="auto"` (the default of structured cases) and `"fp64"` build FP64 fields on every
  device.
* For a steady structured case, on a consumer GPU, the planner adds an FP32 candidate only if
  FP32 can reach the case's steady tolerance. The test is that the round-off bound
  u₃₂·L/(U·Δt), the normalised residual one step's velocity round-off produces, is within
  `steady_tol` (`fp32_admissible`). If FP32 is admissible, both precisions are timed and the
  faster one is kept.
* Transient cases have no accuracy target to test against, so they stay FP64 unless you ask
  for `precision="fp32"`.
* `precision="fp32"` is always honoured. The plan notes when the bound exceeds `steady_tol`,
  in which case the residual will probably stall above it and the run end at `max_steps`.

The decision and its reason are the first line of the execution plan ("Numerical precision").
The reason for every candidate device is in `ExecutionPlan.precision_by_backend`.

Why the test exists. `benchmarks/precision_fp32.py` ran six steady cases in both precisions:

--8<-- "benchmarks/results/precision_fp32.md"

The FP32 fields were within 4.1e-5 of FP64, and the validated quantities agreed to the digits
shown. But FP32 reached the steady tolerance in only one case. In the other five, its residual
stalled at 0.07–0.33 of the bound, and the run took three times the FP64 step count without
reporting convergence. The pressure projection stays divergence-free to about 1e-5 in FP32
and 1e-14 in FP64 (see the [GPU log](../validation/gpu.md)). The unstructured (SIMPLEC) and
compressible solvers run FP64 unless `precision="fp32"` is set; the planner does not choose
FP32 for them.

## Fused CUDA kernels

On the GPU, the convection and Laplacian stencils of the MAC solver are CUDA C++ kernels,
compiled at run time by NVRTC through `cupy.RawModule`. They replace dozens of small elementwise
launches per term. Before porting, profiling showed the CuPy time step to be launch-bound;
after porting it was measured 1.2–1.9× faster from 128² cells up. The vectorised CuPy code
remains the fallback (set `FLOWTRACE_NO_CUDA_KERNELS=1` to force it), and the test suite checks
that both give the same answer in FP64.

## The unstructured solver on the GPU

SIMPLEC runs unchanged on CuPy. The AMG hierarchy is built on the CPU and applied on the GPU.
For the unstructured solver the profile is dominated by the AMG pressure solve. Its default
inner tolerance of 0.1, the same as Fluent's AMG default, was chosen from measurements: outer
iterations stayed the same and wall time dropped 15–50 %.

A SIMPLEC iteration on the GPU launches thousands of small kernels and synchronises with the
host for every Krylov dot product. Its cost therefore has a **floor that does not shrink with
the mesh**. Cavity, FP64, ms per iteration, measured by `benchmarks/simplec_iteration_cost.py`:

--8<-- "benchmarks/results/simplec_iteration_cost.md"

So the planner for unstructured cases:

1. times the real case on the CPU for one AMG rebuild period (10 iterations, at most ~5 s);
2. skips the GPU if the CPU is already below 0.7 × the GPU floor. The floor is one iteration on
   a 16 × 16 mesh, measured once per machine and cached in `~/.flowtrace/planner.json`;
3. also skips the GPU when it cannot pay for itself: even at its floor, it saves at most
   (CPU time − floor) per iteration, and over the iterations left that must exceed the GPU's
   measured set-up time (`gpu_setup_ms`, building a second small solver on a warm device, also
   cached). This rule targets the planning overhead the [planner study](../validation/planner.md)
   found at 9,216 cells. It has not yet been re-measured on a GPU;
4. otherwise times the GPU as well and keeps the faster solver.

The iterations used for timing are real iterations, and the kept solver continues from them.
The [planner study](../validation/planner.md) compares this against fixed CPU, fixed GPU and
the measured oracle.

## Measure your own machine

```bash
flowtrace gpu-report --out results/
```

This writes `gpu_<host>.md/json`: device, CUDA versions, micro-benchmarks, GPU-vs-CPU equality
checks, and ms per step per backend for 64²–1024² grids with the planner's choice. Every
backend is timed in FP64. The GPU is also timed in FP32, and that time is listed separately.

## Memory

Before a run, `hardware/memory.py` estimates the memory the case needs. The dashboard shows
the estimate, and the engine logs it next to the memory available. For the unstructured solver
the estimate includes a measured-overhead term. On the CPU it is an upper bound.
`benchmarks/memory_estimate.py` measured the peak traced allocation (tracemalloc) of eleven
configurations: 2-D and 3-D, laminar, energy and SST. For the SIMPLEC cases it also measured
the growth of the resident set of a fresh process, which includes memory that compiled code
allocates itself:

--8<-- "benchmarks/results/memory_estimate.md"

The planner uses the estimate before anything large is allocated. It does not time or use a
device the case does not fit on, so a GPU that is too small falls back to the CPU. If the case
fits on no device, the planner refuses it with a `MemoryError` that says how much memory the
case may need, how much is free, and what would help. The CLI prints that as a one-line error,
with exit status 2. Because the estimate is an upper bound, a case near the limit may be
refused although it would have fitted. Set `FLOWTRACE_MEMORY_CHECK=off` to start it anyway. The
overhead of CuPy's memory pool on a GPU has not been measured.
