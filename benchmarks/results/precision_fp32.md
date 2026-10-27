Measured 2026-10-26T20:55:00 with FlowTrace 0.3.0 on Intel(R) Xeon(R) Processor @ 2.80GHz (Python 3.13.17), CPU NumPy backend.

| Case | FP32 vs FP64 field (max rel.) | Validated quantity FP64 / FP32 | Steady tol. | FP32 reached it? | FP32 residual floor | u32 L/(U dt) | floor / bound |
|---|---|---|---|---|---|---|---|
| cavity re=100, n=32 | 4.8e-06 | Centreline u vs Ghia et al. (Re=100): 0.0033247 / 0.0033266 | 1e-06 | yes | 9.58e-07 | 1.14e-05 | 0.08 |
| cavity re=100, n=64 | 7.7e-06 | Centreline u vs Ghia et al. (Re=100): 0.004195 / 0.0041983 | 1e-06 | no (12,930 steps vs 4,310 for FP64) | 1.12e-06 | 1.26e-05 | 0.09 |
| cavity re=100, n=128 | 7.8e-06 | Centreline u vs Ghia et al. (Re=100): 0.0050219 / 0.0050253 | 1e-06 | no (26,340 steps vs 8,780 for FP64) | 2.87e-06 | 2.56e-05 | 0.11 |
| cavity re=1000, n=128 | 4.1e-05 | Centreline u vs Ghia et al. (Re=1000): 0.0039822 / 0.0039971 | 1e-06 | no (176,580 steps vs 58,860 for FP64) | 2.55e-06 | 3.61e-05 | 0.07 |
| natural_convection ra=10000, n=64 | 1.2e-06 | Mean Nusselt vs de Vahl Davis (Ra=10000): 2.2511 / 2.2511 | 1e-06 | no (5,730 steps vs 1,910 for FP64) | 5.96e-06 | 1.81e-05 | 0.33 |
| natural_convection ra=100000, n=64 | 6.8e-06 | Mean Nusselt vs de Vahl Davis (Ra=100000): 4.5686 / 4.5686 | 1e-06 | no (14,100 steps vs 4,700 for FP64) | 3.70e-06 | 1.79e-05 | 0.21 |
