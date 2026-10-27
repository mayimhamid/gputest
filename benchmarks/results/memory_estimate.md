Measured 2026-10-27T09:59:54 with FlowTrace 0.3.0 on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics (CPU). Peak allocated: tracemalloc. Peak RSS growth: VmHWM - VmRSS of a fresh process whose peak was reset after the imports and a warm-up solve (Linux).

| Solver | Case | Peak allocated | Peak RSS growth | Estimate | Estimate / peak | Estimate / RSS growth | Upper bound |
|---|---|---|---|---|---|---|---|
| SIMPLEC | 2-D, 4,096 cells, laminar | 12.3 MB | - | 18.2 MB | 1.48 | - | yes |
| SIMPLEC | 2-D, 16,384 cells, laminar | 28.5 MB | - | 42.6 MB | 1.49 | - | yes |
| SIMPLEC | 2-D, 65,536 cells, laminar | 113.0 MB | - | 140.1 MB | 1.24 | - | yes |
| SIMPLEC | 2-D, 16,384 cells, laminar, energy | 28.9 MB | - | 43.6 MB | 1.51 | - | yes |
| SIMPLEC | 2-D, 16,384 cells, k-omega-sst | 31.7 MB | - | 42.6 MB | 1.34 | - | yes |
| SIMPLEC | 3-D, 32,768 cells, laminar | 85.5 MB | - | 110.2 MB | 1.29 | - | yes |
| MAC | cavity 128x128 | 4.0 MB | - | 6.8 MB | 1.68 | - | yes |
| MAC | cavity 256x256 | 16.0 MB | - | 26.9 MB | 1.69 | - | yes |
| MAC | cavity 512x512 | 63.8 MB | - | 107.6 MB | 1.69 | - | yes |
| MAC | natural_convection 256x256 | 20.2 MB | - | 32.2 MB | 1.60 | - | yes |
| MAC | cylinder 256x128 | 11.2 MB | - | 19.8 MB | 1.77 | - | yes |
