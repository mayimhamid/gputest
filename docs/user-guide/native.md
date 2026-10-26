# Native kernels (C++/OpenMP and CUDA)

The unstructured SIMPLE/SIMPLEC solver is written against NumPy/SciPy (CPU) and CuPy (GPU),
and that code path remains the reference. Profiling showed that a 23k-cell iteration was not
spent in any one heavy operation. It went into hundreds of small NumPy calls on a single
core: face interpolation, gradients, assembly, sparse products and the Python loop of the
Krylov solver. FlowTrace therefore ships compiled kernels for those parts:

| File | Language | Compiled by | What it contains |
|---|---|---|---|
| `flowtrace/native/fvm_kernels.h` | C API | — | the functions below, callable from C, C++ or Python (ctypes) |
| `flowtrace/native/fvm_kernels.cpp` | C++17 + OpenMP | g++ / clang++ (first use) | parallel CSR products, fused least-squares and Green–Gauss gradients, face interpolation, matrix assembly, a complete Jacobi-BiCGSTAB solve |
| `flowtrace/native/fvm_kernels.cu` | CUDA C++ | NVRTC through CuPy (first use); `nvcc -cubin` also works where nvcc has a host compiler | the same face/cell kernels plus fused BiCGSTAB vector updates and reductions (4 host synchronisations per iteration instead of 6) |

Both are compiled the first time a solver needs them and cached:

* C++: `~/.flowtrace/native/<hash>/flowtrace_fvm.{dll,so}`, built with
  `g++ -O3 -march=native -fopenmp -shared`. On Windows the library is also linked
  `-static`, so it needs no MinGW runtime DLLs; MSYS2's UCRT64 g++ works.
  `FLOWTRACE_CXX` selects another compiler, and `FLOWTRACE_NATIVE_PORTABLE=1` builds for a
  generic x86-64-v2 CPU instead of `-march=native`.
* CUDA: compiled in memory by NVRTC (CuPy keeps its own cache). No `nvcc` and no host
  compiler are needed. On Windows, nvcc cannot even build device-only code without MSVC's
  `cl.exe`, which is why NVRTC is used.

```bash
flowtrace probe        # "Native  C++/OpenMP kernels (16 threads)" and "CUDA FV kernels (NVRTC)"
```

## When they are used

* CPU backends (`numpy`, `numba`) in FP64: the C++ kernels, if a compiler was found.
* GPU backend (`cupy`) in FP64: the CUDA kernels.
* FP32 runs, machines without a compiler, and `FLOWTRACE_NO_NATIVE=1` use the reference path.
  `FLOWTRACE_NO_CUDA_KERNELS=1` disables only the CUDA kernels.
* Every OpenMP loop is parallel over cells or faces without atomics, so results do not depend
  on the thread count. `flowtrace batch` sets `OMP_NUM_THREADS=1` in its workers: there, the
  parallelism comes from running cases side by side.

## Correctness

`tests/test_native.py` and `tests/test_gpu_real.py` compare every kernel with the reference.

* Face interpolation, the sparse products and the assembled matrices are bit-identical.
* Gradients agree to 1e-13.
* The C++ and CUDA BiCGSTAB solvers take exactly the same iterations as
  `flowtrace.linalg.krylov.bicgstab`.
* 40 SIMPLEC iterations agree to 1e-10.
* A converged run with energy agrees to 1e-6.

The energy case is compared at convergence for a reason. On that convection-dominated
system, a change of 1e-15 in the right-hand side changes the inner Jacobi-BiCGSTAB result by
about 1e-2, so rounding alone makes iterates differ between any two implementations.

Making the comparison exact required one more fix. pyamg drew random vectors from NumPy's
global RNG, so two runs of the *same* code differed by about 1e-8. FlowTrace now seeds that
setup, and two runs in separate processes are bit-identical.

## Speed

Measured with `benchmarks/native_kernels.py`: three interleaved rounds per configuration in
fresh processes, 50 SIMPLEC iterations each, FP64.

--8<-- "benchmarks/results/native_kernels.md"

What remains on the CPU is mostly pyamg's single-threaded Gauss–Seidel smoothing and AMG
setup, plus the parts of the momentum right-hand side that are still NumPy. Those are the
next candidates.
