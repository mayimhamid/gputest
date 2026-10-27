Measured 2026-10-26T20:50:56 with FlowTrace 0.3.0 on Intel(R) Xeon(R) Processor @ 2.80GHz (Python 3.13.17).
Cavity Re = 10, 24 x 24 uniformly sheared quads; columns = pressure under-relaxation alpha_p.

| theta | min orth. quality | cos^2 theta | 1 | 0.8 | 0.5 | 0.33 | 0.2 | auto + guard | SIMPLE (0.3) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.00 | ok 201 | ok 207 | ok 216 | ok 226 | ok 239 | ok 201 | ok 200 |
| 10 | 0.985 | 0.97 | ok 193 | ok 197 | ok 208 | ok 217 | ok 230 | ok 197 | ok 191 |
| 20 | 0.940 | 0.88 | ok 221 | ok 191 | ok 199 | ok 211 | ok 224 | ok 191 | ok 191 |
| 30 | 0.866 | 0.75 | diverged @68 | diverged @1778 | ok 194 | ok 205 | ok 219 | ok 241, 1 rollback(s) -> 0.48 | diverged @130 |
| 35 | 0.819 | 0.67 | diverged @86 | diverged @224 | ok 190 | ok 200 | ok 216 | ok 216, 1 rollback(s) -> 0.48 | diverged @73 |
| 40 | 0.766 | 0.59 | diverged @35 | diverged @96 | ok 3476 | ok 196 | ok 211 | ok 451, 1 rollback(s) -> 0.48 | diverged @56 |
| 45 | 0.707 | 0.50 | diverged @45 | diverged @71 | diverged @2496 | ok 189 | ok 204 | ok 247, 2 rollback(s) -> 0.29 | diverged @48 |
| 50 | 0.643 | 0.41 | diverged @27 | diverged @45 | diverged @137 | ok 1540 | ok 200 | ok 233, 2 rollback(s) -> 0.29 | diverged @43 |
| 55 | 0.574 | 0.33 | diverged @31 | diverged @38 | diverged @107 | not converged (4000) | ok 200 | ok 457, 3 rollback(s) -> 0.17 | diverged @35 |
| 60 | 0.500 | 0.25 | diverged @35 | diverged @41 | diverged @78 | diverged @195 | ok 327 | ok 292, 3 rollback(s) -> 0.17 | diverged @31 |
