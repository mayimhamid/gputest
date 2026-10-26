### Proxy: FlowTrace SIMPLEC, cold start vs warm start from the FlowTrace `.ip` file

Measured. Both runs are counted until the velocity (and temperature) field is within 0.0001 (relative, max norm) of the converged solution. The last two columns count iterations until the scaled residuals reach the tolerance, which is what a solver's convergence monitor shows; the warm count there is distorted by the continuity normalisation (see the text).

| Case | Target cells | Cold | Warm | Saved | FlowTrace pre-solve | Cold to tol. | Warm to tol. |
|---|---|---|---|---|---|---|---|
| Lid-driven cavity, Re = 400 | 3,200 | 1,162 | 488 | x2.4 | 8.2 s | 960 | 417 |
| Developing channel flow, Re = 100 | 1,152 | 68 | 47 | x1.4 | 0.4 s | 77 | - |
| Backward-facing step, Re = 200 (laminar) | 5,616 | 772 | 418 | x1.8 | 224.2 s | 617 | 366 |
| Cylinder in a channel, Re = 40 (steady) | 10,308 | 700 | 1,085 | x0.6 | 331.9 s | 886 | - |
| Natural convection, Ra = 1e5 (heated cavity) | 3,200 | 316 | 158 | x2.0 | 4.0 s | 276 | - |

### Ansys Fluent: cold start vs `.ip` warm start

**Not measured** - no licensed Fluent available when this table was generated (ansys-fluent-core is not installed (pip install ansys-fluent-core)). Run `python -m flowtrace warmstart-study` on a machine with Fluent + PyFluent, or run `cold.jou`/`warm.jou` in each case folder and `python -m flowtrace parse-transcript cold.trn warm.trn`.

| Case | Status | Cold iterations | Warm iterations | Saved |
|---|---|---|---|---|
| cavity | not run | _placeholder_ | _placeholder_ | - |
| channel | not run | _placeholder_ | _placeholder_ | - |
| step | not run | _placeholder_ | _placeholder_ | - |
| cylinder | not run | _placeholder_ | _placeholder_ | - |
| heated | not run | _placeholder_ | _placeholder_ | - |

_Generated 2026-10-26T06:23:39 on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics, residual tolerance 1e-05._
