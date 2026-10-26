| cells | 1 024 | 4 096 | 16 384 | 65 536 | 262 144 |
|---|---:|---:|---:|---:|---:|
| CPU (NumPy) | 1.6 | 4.7 | 20.4 | 113.6 | 485.8 |
| GPU (CuPy) | 15.2 | 16.4 | 18.2 | 18.6 | 30.0 |

_ms per iteration, FP64, median of up to 20 iterations; measured 2026-10-26T06:38:07 on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics and NVIDIA GeForce RTX 4060 Laptop GPU. Reproduce: `python benchmarks/simplec_iteration_cost.py --sizes 32,64,128,256,512`._
