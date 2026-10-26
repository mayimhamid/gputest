Measured 2026-10-26T19:53:29 on AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD (16 threads, OpenMP 16); FlowTrace 0.3.0 @ 9bca4f832d (uncommitted changes), Python 3.13.17. Median of 3 interleaved rounds of 50 SIMPLEC iterations, FP64.

| case | cells | device | kernels | reference ms/it | native ms/it | speed-up | max rel. du | N1 | N2 |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| cavity64 | 4,096 | CPU | C++/OpenMP | 4.63 | 4.77 | 0.97x | 1.3e-15 | pass | pass |
| cavity64 | 4,096 | GPU | CUDA | 17.25 | 11.52 | 1.50x | 1.8e-15 | pass | pass |
| cavity128 | 16,384 | CPU | C++/OpenMP | 22.13 | 16.67 | 1.33x | 2.1e-15 | pass | pass |
| cavity128 | 16,384 | GPU | CUDA | 16.98 | 11.39 | 1.49x | 2.7e-15 | pass | pass |
| cavity256 | 65,536 | CPU | C++/OpenMP | 163.71 | 86.56 | 1.89x | 2.0e-15 | pass | pass |
| cavity256 | 65,536 | GPU | CUDA | 18.73 | 12.00 | 1.56x | 2.3e-15 | pass | pass |
| cavity512 | 262,144 | CPU | C++/OpenMP | 730.75 | 342.84 | 2.13x | 2.0e-09 | pass | pass |
| cavity512 | 262,144 | GPU | CUDA | 43.64 | 28.16 | 1.55x | 2.7e-15 | pass | pass |
| SquareCylinder-Finest | 22,984 | CPU | C++/OpenMP | 33.73 | 24.87 | 1.36x | 1.7e-14 | pass | pass |
| SquareCylinder-Finest | 22,984 | GPU | CUDA | 19.75 | 13.26 | 1.49x | 3.0e-14 | pass | pass |

Overall: all criteria met.

Reproduce: `python benchmarks/native_kernels.py --sizes 64,128,256,512 --mesh SquareCylinder-Finest.msh`
