Measured 2026-10-26T14:25:28 on AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD (16 threads); FlowTrace 0.2.0 @ 57fab437ea (uncommitted changes). Viscosity 0.1, tol 1e-07.

| case | cells | sequential: device, its, s | batch: device, its, s | force (seq) | force (batch) | B2 |
|---|---:|---|---|---:|---:|---|
| SquareCylinder-Coarse | 5,824 | numba, 311, 3.8 | cpu, 311, 4.2 | 4.245518 | 4.245518 | pass |
| SquareCylinder-Medium | 10,904 | numba, 2148, 75.0 | cpu, 2146, 61.4 | 4.295224 | 4.295224 | pass |
| SquareCylinder-Fine | 16,464 | numba, 1111, 49.9 | cpu, 1111, 49.1 | 4.310324 | 4.310324 | pass |
| SquareCylinder-Finest | 22,984 | cupy, 1664, 138.2 | gpu, 1664, 101.4 | 4.322339 | 4.322339 | pass |

Whole study: sequential 269.2 s, batch 103.8 s, speed-up 2.59x (B1 pass).

Reproduce: `python benchmarks/batch_study.py C:/Users/Hamid/Desktop/Square Cylinder Mesh Independency/SquareCylinder-Coarse.msh C:/Users/Hamid/Desktop/Square Cylinder Mesh Independency/SquareCylinder-Medium.msh C:/Users/Hamid/Desktop/Square Cylinder Mesh Independency/SquareCylinder-Fine.msh C:/Users/Hamid/Desktop/Square Cylinder Mesh Independency/SquareCylinder-Finest.msh --zone BluffWalls --mu 0.1`
