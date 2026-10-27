Measured 2026-10-27T10:17:48 with FlowTrace 0.3.0 on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics (CPU). Peak allocated: tracemalloc. Peak RSS growth: VmHWM - VmRSS of a fresh process whose peak was reset after the imports and a warm-up solve (Linux).

| Solver | Case | Peak allocated | Peak RSS growth | Estimate | Estimate / peak | Estimate / RSS growth | Upper bound |
|---|---|---|---|---|---|---|---|
| SIMPLEC | 2-D, 4,096 cells, laminar | 12.0 MB | - | 17.0 MB | 1.42 | - | yes |
| SIMPLEC | 2-D, 16,384 cells, laminar | 27.2 MB | - | 38.0 MB | 1.40 | - | yes |
| SIMPLEC | 2-D, 65,536 cells, laminar | 107.7 MB | - | 121.7 MB | 1.13 | - | yes |
| SIMPLEC | 2-D, 16,384 cells, laminar, energy | 27.6 MB | - | 39.0 MB | 1.41 | - | yes |
| SIMPLEC | 2-D, 16,384 cells, k-omega-sst | 30.4 MB | - | 38.0 MB | 1.25 | - | yes |
| SIMPLEC | 3-D, 32,768 cells, laminar | 83.6 MB | - | 96.1 MB | 1.15 | - | yes |
| MAC | cavity 128x128 | 4.0 MB | - | 6.8 MB | 1.68 | - | yes |
| MAC | cavity 256x256 | 16.0 MB | - | 26.9 MB | 1.69 | - | yes |
| MAC | cavity 512x512 | 63.8 MB | - | 107.6 MB | 1.69 | - | yes |
| MAC | natural_convection 256x256 | 20.2 MB | - | 32.2 MB | 1.60 | - | yes |
| MAC | cylinder 256x128 | 11.2 MB | - | 19.8 MB | 1.77 | - | yes |
