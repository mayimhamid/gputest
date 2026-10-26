| Grid | Cells per D | Recirculation length, pre-solve field | Recirculation length, SIMPLEC converged | Max-norm distance of the interpolated field |
|---|---:|---:|---:|---:|
| source grid (pre-solve resolution) | 12 | 1.79 D | 1.87 D | 5 % |
| target grid of the study | 9 | 1.78 D | 2.33 D | 21 % |

_Lengths from the rear of the cylinder to the last reversed-flow cell centre on the wake axis; resolution one cell (0.08 D and 0.11 D). Reproduce: `python benchmarks/warmstart_cylinder_wake.py`._
