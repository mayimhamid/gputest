## GPU log - NVIDIA GeForce RTX 4060 Laptop GPU

Measured 2026-10-26T06:38:08 on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics (8C/16T), Windows-11-10.0.26300-SP0.

CuPy 14.2.0, CUDA runtime 12090, driver API 13040.

### Micro-benchmarks

| Backend | Bandwidth GB/s | GEMM FP64 GF/s | GEMM FP32 GF/s | DCT 512² ms | Stencil Mcell/s | Launch µs | H2D GB/s |
|---|---|---|---|---|---|---|---|
| numpy | 39.92 | 291.05 | 420.55 | 1.327 | 122.0 | 0.0 | - |
| numba | 40.62 | 229.96 | 363.68 | 1.321 | 5277.2 | 0.0 | - |
| cupy | 227.92 | 206.39 | 9193.0 | 0.368 | 5322.7 | 10.53 | 8.73 |

### GPU vs CPU correctness

| Check | Value |
|---|---|
| MAC cavity fp64, 50 steps: max|u_gpu - u_cpu| | 3.331e-16 |
| MAC cavity fp64: max|div u| on GPU | 7.105e-15 |
| MAC natural_convection fp64, 50 steps: max|u_gpu - u_cpu| | 1.776e-14 |
| MAC natural_convection fp64: max|div u| on GPU | 4.547e-13 |
| MAC cylinder fp64, 50 steps: max|u_gpu - u_cpu| | 1.110e-15 |
| MAC cylinder fp64: max|div u| on GPU | 1.907e-12 |
| MAC rayleigh_benard fp64, 50 steps: max|u_gpu - u_cpu| | 3.331e-14 |
| MAC rayleigh_benard fp64: max|div u| on GPU | 5.151e-14 |
| MAC cavity fp32, 200 steps: max|div u| | 7.510e-06 |
| SIMPLEC cavity tri fp64, 100 its: max|u_gpu - u_cpu| | 6.280e-14 |

### MAC solver, lid-driven cavity Re=1000: ms per time step

| Grid | numpy | numba | cupy | Planner choice |
|---|---|---|---|---|
| 64² | 0.379 (fp64) | 0.484 (fp64) | 1.787 (fp32); vectorised 3.18 | numpy |
| 128² | 0.913 (fp64) | 0.835 (fp64) | 1.693 (fp32); vectorised 3.226 | numba |
| 256² | 3.705 (fp64) | 2.43 (fp64) | 1.707 (fp32); vectorised 3.119 | cupy |
| 512² | 57.559 (fp64) | 31.481 (fp64) | 6.498 (fp32); vectorised 4.319 | cupy |
| 1024² | 262.003 (fp64) | 136.585 (fp64) | 4.587 (fp32); vectorised 5.696 | cupy |
