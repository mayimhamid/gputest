# GPU and the hardware planner

## Backends

| Backend | Device | What runs there |
|---|---|---|
| `numpy` | CPU | vectorised NumPy/SciPy, multi-threaded FFTs |
| `numba` | CPU, all cores | fused JIT stencil kernels (MAC solver), JIT AMG aggregation |
| `cupy` | NVIDIA GPU | every field resident in VRAM, cuFFT/cuBLAS/cuSPARSE, fused CUDA C++ stencils |

`backend="auto"` picks the GPU, then Numba, then NumPy. For structured cases the **planner
measures** instead: it builds the real solver on each backend that fits in memory, times real
steps after a warm-up, and keeps the fastest. Every decision and its measured reason appear in
the dashboard and in `flowtrace run` output.

## Precision

GeForce/RTX cards run FP64 at 1/32–1/64 of FP32 speed. FlowTrace therefore defaults to FP32
fields on consumer GPUs and FP64 on data-centre GPUs, and `precision="fp64"` forces double.
The pressure projection stays divergence-free to ~1e-5 in FP32 and ~1e-14 in FP64; see the
[GPU log](../validation/gpu.md).

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

## Measure your own machine

```bash
flowtrace gpu-report --out results/
```

This writes `gpu_<host>.md/json`: device, CUDA versions, micro-benchmarks, GPU-vs-CPU equality
checks, and ms per step per backend for 64²–1024² grids with the planner's choice.
