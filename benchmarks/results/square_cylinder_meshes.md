Measured 2026-10-26T19:53:30, FlowTrace 0.3.0 @ 9bca4f832d (uncommitted changes); steady SIMPLEC, residual tolerance 1e-7.

| mesh | cells | side walls | Cd | Cd pressure | Cd viscous | Lr/D | iterations | converged |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Coarse | 5,824 | wall | 2.1228 | 1.7823 | 0.3405 | 2.3485 | 311 | True |
| Medium | 10,904 | wall | 2.1476 | 1.8428 | 0.3048 | 2.3377 | 990 | True |
| Fine | 16,464 | wall | 2.1552 | 1.8543 | 0.3009 | 2.3311 | 1111 | True |
| Finest | 22,984 | wall | 2.1612 | 1.8628 | 0.2984 | 2.3239 | 1665 | True |
| Coarse | 5,824 | symmetry | 1.9065 | 1.6055 | 0.3011 | 2.8977 | 311 | True |
| Medium | 10,904 | symmetry | 1.9302 | 1.6609 | 0.2694 | 2.8567 | 941 | True |
| Fine | 16,464 | symmetry | 1.9373 | 1.6715 | 0.2658 | 2.8425 | 1680 | True |
| Finest | 22,984 | symmetry | 1.9429 | 1.6794 | 0.2636 | 2.8311 | 3964 | True |

Grid convergence (Celik et al. 2008; Medium/Fine/Finest, h ~ 1 / faces on the body):

| side walls | quantity | observed order | extrapolated | GCI fine | asymptotic ratio |
|---|---|---:|---:|---:|---:|
| wall | Cd | 0.50 | 2.1996 | 2.22 % | 0.75 |
| wall | Lr_over_D | 1.88 | 2.3139 | 0.54 % | 0.34 |
| symmetry | Cd | 0.44 | 1.9842 | 2.65 % | 0.78 |
| symmetry | Lr_over_D | 0.55 | 2.7644 | 2.94 % | 0.73 |

Reproduce: `python benchmarks/square_cylinder.py meshes`
