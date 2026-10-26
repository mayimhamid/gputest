Measured 2026-10-26T20:19:17, FlowTrace 0.3.0 @ 9bca4f832d (uncommitted changes); steady SIMPLEC, residual tolerance 1e-7.

| domain | blockage | cells | Re | Cd | Lr/D | iterations | converged |
|---|---:|---:|---:|---:|---:|---:|---|
| user-like | 8.3 % | 38,046 | 10 | 3.6938 | 0.6458 | 6748 | True |
| user-like | 8.3 % | 38,046 | 40 | 1.9357 | 2.7600 | 3377 | True |
| large | 2.5 % | 64,028 | 10 | 3.1117 | 0.6219 | 12000 | False |
| large | 2.5 % | 64,028 | 40 | 1.6708 | 2.8336 | 12000 | False |

user-like: Cd exponent between Re 10 and 40 = -0.466

large: Cd exponent between Re 10 and 40 = -0.449

Reproduce: `python benchmarks/square_cylinder.py domain`
