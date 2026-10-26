Measured 2026-10-26T20:00:48, FlowTrace 0.3.0 @ 9bca4f832d (uncommitted changes); steady SIMPLEC, residual tolerance 1e-7.

| Re | Cd | Lr/D | iterations |
|---:|---:|---:|---:|
| 10 | 3.6859 | 0.6575 | 2317 |
| 20 | 2.6074 | 1.3573 | 1659 |
| 30 | 2.1777 | 2.0966 | 1382 |
| 40 | 1.9373 | 2.8425 | 1836 |

P1: Lr/D = 0.07294 Re -0.0852, R^2 = 0.99978, separation onset Re = 1.17 (Sen et al. 2011: 1.15) -> pass.
P2: Cd ~ Re^-0.466 (criterion -0.66 +- 0.10) -> FAIL.

Reproduce: `python benchmarks/square_cylinder.py sweep`
