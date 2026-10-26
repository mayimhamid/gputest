Measured 2026-10-26T06:37:43 on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics (16 threads) + NVIDIA GeForce RTX 4060 Laptop GPU (driver 13040); FlowTrace 0.2.0 @ 32d202d7ae, Python 3.13.17, Windows-11-10.0.26300-SP0.
Case: lid-driven cavity Re=100, uniform quads, SIMPLEC, FP64, second-order; fixed 100 iterations (tol = 0); median of 3 fresh processes (spread = max - min).
Calibration: GPU floor 16.6 ms/iteration, CPU 1.56 us/cell/iteration.

| cells | CPU (s) | GPU = old auto (s) | planner (s) | planner chose | oracle | planner / oracle | old auto / oracle | A1 | A2 | A3 |
|---:|---:|---:|---:|---|---|---:|---:|---|---|---|
| 1,024 | 1.13 ± 0.08 | 3.58 ± 0.15 | 1.22 ± 0.04 | CPU | CPU | 1.08 | 3.17 | pass | pass | pass |
| 4,096 | 1.50 ± 0.04 | 3.84 ± 0.34 | 1.58 ± 0.06 | CPU | CPU | 1.05 | 2.55 | pass | pass | pass |
| 9,216 | 2.31 ± 0.29 | 3.94 ± 0.25 | 3.42 ± 0.17 | CPU | CPU | 1.48 | 1.71 | pass | **FAIL** | pass |
| 16,384 | 3.89 ± 0.04 | 3.91 ± 0.32 | 4.47 ± 0.14 | GPU | CPU | 1.15 | 1.00 | pass | pass | pass |
| 36,864 | 8.56 ± 0.40 | 3.97 ± 0.16 | 4.50 ± 0.31 | GPU | GPU | 1.13 | 1.00 | pass | pass | pass |
| 65,536 | 14.20 ± 0.11 | 4.02 ± 0.08 | 5.76 ± 0.06 | GPU | GPU | 1.43 | 1.00 | pass | **FAIL** | pass |

Overall: at least one criterion FAILED.

Reproduce: `python benchmarks/planner_unstructured.py --sizes 32,64,96,128,192,256 --iters 100 --repeats 3`
