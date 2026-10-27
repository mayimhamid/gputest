Measured 2026-10-26T21:35:33 with FlowTrace 0.3.0 on Intel(R) Xeon(R) Processor @ 2.80GHz (CPU). Peak allocated: tracemalloc. Peak RSS growth: VmHWM - VmRSS of a fresh process whose peak was reset after the imports and a warm-up solve (Linux).

| Solver | Case | Peak allocated | Peak RSS growth | Estimate | Estimate / peak | Estimate / RSS growth | Upper bound |
|---|---|---|---|---|---|---|---|
| SIMPLEC | 2-D, 4,096 cells, laminar | 13.2 MB | 9.3 MB | 19.4 MB | 1.47 | 2.10 | yes |
| SIMPLEC | 2-D, 16,384 cells, laminar | 30.7 MB | 30.8 MB | 47.5 MB | 1.55 | 1.54 | yes |
| SIMPLEC | 2-D, 65,536 cells, laminar | 121.8 MB | 130.1 MB | 159.6 MB | 1.31 | 1.23 | yes |
| SIMPLEC | 2-D, 16,384 cells, laminar, energy | 31.1 MB | 32.8 MB | 48.6 MB | 1.56 | 1.48 | yes |
| SIMPLEC | 2-D, 16,384 cells, k-omega-sst | 33.9 MB | 34.2 MB | 47.5 MB | 1.40 | 1.39 | yes |
| SIMPLEC | 3-D, 32,768 cells, laminar | 94.2 MB | 79.8 MB | 125.3 MB | 1.33 | 1.57 | yes |
| MAC | cavity 128x128 | 4.0 MB | - | 6.8 MB | 1.68 | - | yes |
| MAC | cavity 256x256 | 16.0 MB | - | 26.9 MB | 1.69 | - | yes |
| MAC | cavity 512x512 | 63.8 MB | - | 107.6 MB | 1.69 | - | yes |
| MAC | natural_convection 256x256 | 20.2 MB | - | 32.2 MB | 1.60 | - | yes |
| MAC | cylinder 256x128 | 11.2 MB | - | 19.8 MB | 1.77 | - | yes |
