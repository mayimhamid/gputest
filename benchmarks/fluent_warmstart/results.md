### Proxy: FlowTrace SIMPLEC, cold start vs warm start from the FlowTrace `.ip` file

Measured. Both runs are counted until the velocity (and temperature) field is within 0.0001 (relative, max norm) of the converged solution. The last two columns count iterations until the scaled residuals reach the tolerance, which is what a solver's convergence monitor shows; the warm count there is distorted by the continuity normalisation (see the text).

| Case | Target cells | Cold | Warm | Saved | FlowTrace pre-solve | Cold to tol. | Warm to tol. |
|---|---|---|---|---|---|---|---|
| Lid-driven cavity, Re = 400 | 3,200 | 1,162 | 488 | x2.4 | 19.9 s | 960 | 417 |
| Developing channel flow, Re = 100 | 1,152 | 68 | 47 | x1.4 | 0.9 s | 77 | - |
| Backward-facing step, Re = 200 (laminar) | 5,616 | 772 | 419 | x1.8 | 245.9 s | 608 | 372 |
| Cylinder in a channel, Re = 40 (steady) | 10,308 | 700 | 1,085 | x0.6 | 561.1 s | 886 | - |
| Natural convection, Ra = 1e5 (heated cavity) | 3,200 | 316 | 160 | x2.0 | 12.0 s | 275 | - |

Wall time on equal terms (solver steps only, until the same field tolerance; the warm path also pays the FlowTrace pre-solve and the interpolation). A ratio above 1 would mean the whole warm-start path is faster. The proxy's target solver is cheap, so this says nothing about a large Fluent case; it keeps iteration savings and elapsed-time savings apart.

| Case | Cold solver time | Pre-solve + interpolation + warm solver | Wall-time ratio |
|---|---|---|---|
| Lid-driven cavity, Re = 400 | 13.13 s | 25.46 s (19.87 + 0.07 + 5.52) | x0.52 |
| Developing channel flow, Re = 100 | 0.69 s | 1.37 s (0.90 + 0.05 + 0.42) | x0.50 |
| Backward-facing step, Re = 200 (laminar) | 28.90 s | 261.93 s (245.88 + 0.26 + 15.79) | x0.11 |
| Cylinder in a channel, Re = 40 (steady) | 30.95 s | 607.49 s (561.14 + 0.57 + 45.78) | x0.05 |
| Natural convection, Ra = 1e5 (heated cavity) | 7.37 s | 15.35 s (12.00 + 0.08 + 3.27) | x0.48 |

### Ansys Fluent: cold start vs `.ip` warm start

**Not measured** - no licensed Fluent available when this table was generated (ansys-fluent-core is not installed (pip install ansys-fluent-core)). Run `python -m flowtrace warmstart-study` on a machine with Fluent + PyFluent, or run `cold.jou`/`warm.jou` in each case folder and `python -m flowtrace parse-transcript cold.trn warm.trn`.

| Case | Status | Cold iterations | Warm iterations | Saved |
|---|---|---|---|---|
| cavity | not run | _placeholder_ | _placeholder_ | - |
| channel | not run | _placeholder_ | _placeholder_ | - |
| step | not run | _placeholder_ | _placeholder_ | - |
| cylinder | not run | _placeholder_ | _placeholder_ | - |
| heated | not run | _placeholder_ | _placeholder_ | - |

_Generated 2026-10-27T06:45:34 on Intel(R) Xeon(R) Processor @ 2.80GHz, residual tolerance 1e-05._
