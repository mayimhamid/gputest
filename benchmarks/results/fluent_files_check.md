Measured 2026-10-26T20:20:30 with FlowTrace 0.3.0, Python 3.13.17, Linux-6.18.44-fc-v153-x86_64-with-glibc2.39.
Reference files: PyFluent mixing-elbow example, ansys/example-data (MIT licence, Copyright Synopsys, Inc. and ANSYS, Inc.); SHA-256 in `flowtrace/validation/fluent_files.py`.

| Property | Measured | Result |
|---|---|---|
| 3-D polyhedral mesh read (17,822 cells, 91,581 faces) | 17,822 cells, 91,581 faces with 3-9 nodes | pass |
| zone types | wall-elbow (wall), wall-inlet (wall), outlet (pressure-outlet), cold-inlet (velocity-inlet), hot-inlet (velocity-inlet), symmetry-xyplane (symmetry), interior--elbow-fluid (interior), elbow-fluid (fluid) | pass |
| no face re-oriented on read | 0 / 0 | pass |
| every cell closes (|sum S| / sum |S| < 1e-12) | 2.00e-16 | pass |
| mesh and case: same topology, volumes in the ratio 0.0254^3 | same topology True, ratio 1.6387064e-05 | pass |
| written back: Fluent's node order and c0/c1 for every face | 91,581 / 91,581 of 91,581 | pass |
| data mapped to cells: outlet / inlets mass flow within 10 % | 1.0021 | pass |

Not covered: Fluent reading files that FlowTrace wrote (needs a licensed Fluent).
