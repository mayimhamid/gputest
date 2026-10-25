---
title: 'FlowTrace: a verified, hardware-aware CFD solver that warm-starts Ansys Fluent'
tags:
  - Python
  - computational fluid dynamics
  - SIMPLEC
  - RANS turbulence
  - GPU
  - verification and validation
authors:
  - name: mayimhamid
    affiliation: 1
affiliations:
  - name: Independent researcher (affiliation to be completed by the author)
    index: 1
date: 26 October 2026
bibliography: paper.bib
---

# Summary

FlowTrace is an open-source Python package for incompressible computational fluid dynamics
(CFD). It provides two solvers:

* a structured, staggered-grid projection solver for two-dimensional Navier–Stokes flow with
  Boussinesq heat transfer;
* an unstructured, cell-centred, pressure-based SIMPLE/SIMPLEC solver [@vandoormaal1984] with
  Rhie–Chow interpolation [@rhie1983] for two- and three-dimensional meshes of arbitrary cells.

The unstructured solver includes RANS turbulence (k-ω SST [@menter2003], realizable
[@shih1995] and standard k-ε), conjugate heat transfer between fluid and solid zones, moving
reference frames and translational periodicity. All numerics are written once against the NumPy
array API [@harris2020; @virtanen2020] and run unchanged on the CPU, on Numba JIT kernels
[@lam2015] or on NVIDIA GPUs through CuPy [@okuta2017]. For each case a planner times real
solver steps on every available backend and chooses where each stage runs.

FlowTrace reads and writes the file formats of the commercial solver Ansys Fluent: mesh and case
files (`.msh`, CFF/HDF5 `.cas.h5`), cell data (`.dat.h5`), interpolation files (`.ip`), boundary
profiles, journals and PyFluent scripts. Its intended use is as a *pre-solver*. It computes an
initial field quickly on the hardware at hand, and Fluent interpolates that field onto its own
mesh and continues from it (a warm start) instead of starting from rest.

# Statement of need

Mature open-source CFD codes such as OpenFOAM [@weller1998] and SU2 [@economon2016] are full
replacements for commercial solvers. Python frameworks such as Dedalus [@burns2020] target
spectral methods on simple domains. Many engineering groups, however, keep a commercial solver
as the solver of record for contractual and qualification reasons. They need tools that make
that solver cheaper to use, are easy to install on a workstation or laptop, use the GPU that is
already there, and are trustworthy enough to place in front of a qualified workflow.

FlowTrace addresses this gap in three ways:

1. **Warm start.** A FlowTrace solution is handed to Fluent through its own interpolation
   mechanism. Generated journals and PyFluent scripts open the user's *original* case, so the
   physics set up in Fluent is unchanged. A built-in study measures how many iterations the
   field saves. It uses FlowTrace's own pressure-based solver as a measured proxy, and Ansys
   Fluent when a licence is available.
2. **Measured, not assumed, hardware placement.** For a 1024² lid-driven cavity, the MAC
   solver on a laptop GPU (RTX 4060) was measured 22× faster per time step than the multi-core
   CPU path. For grids of 128² and smaller the CPU is faster, and the planner chooses
   accordingly. Fused CUDA kernels, compiled at run time with NVRTC, replace launch-bound
   vectorised code in the stencils. They were ported only after profiling, and they come with
   equality tests against the vectorised path.
3. **Evidence.** A single command, `flowtrace benchmark`, regenerates every number the project
   publishes, including those quoted in this paper. It writes a JSON record, an HTML/PDF report
   and a Markdown table.

# Verification and validation

Code verification uses the method of manufactured solutions [@roache2002] on systematically
refined, smoothly distorted meshes. The structured solver shows second-order convergence for
velocity, pressure and temperature. The unstructured solver shows second order for velocity
and temperature on warped quadrilaterals and triangles. Pressure converges at second order on
quadrilaterals and at an order between 1.2 and 1.5 on triangles.

The manufactured solutions also exposed a defect that conventional validation had missed. The
unstructured solver agreed with the lid-driven cavity data of @ghia1982 to within 0.007, yet its
error on triangular meshes did not decrease under refinement. The cause was the pairing of a
least-squares cell pressure gradient in the momentum equation with a face-flux divergence in the
continuity equation, two operators that are not discrete adjoints. Replacing the pressure force
by a Green–Gauss face-pressure form restored second-order accuracy. This illustrates why
FlowTrace treats verification as part of its test suite and not as a one-off exercise.

Discretisation uncertainty is reported with the grid convergence index [@celik2008]. Validation
covers:

* laminar cavity flows [@ghia1982] and natural convection [@devahldavis1983];
* a turbulent channel at Re_τ = 392 against direct numerical simulation [@moser1999], where SST
  is within 5 % of the DNS mean velocity at all compared points;
* the zero-pressure-gradient flat plate against skin-friction correlations [@white1974];
* the backward-facing step of @driver1985.

The validation report in the documentation lists every case, its acceptance criterion and the
measured result, including the known deviations of the k-ε models.

# Implementation notes

The unstructured discretisation stores Fluent's face-based data model directly, and it
precomputes all geometric operators as sparse incidence matrices. Every per-iteration operation
is therefore a vectorised gather/scatter or a sparse product on the selected device. Pressure
corrections use algebraic multigrid: PyAMG [@bell2022] on the CPU, and a smoothed-aggregation
implementation whose cycles run on the GPU. A local web dashboard (standard library only, works
offline, English and Persian interfaces) provides case setup, live residuals and fields, the
execution plan with its measured reasons, saved projects and exportable run reports.

# Acknowledgements

The author thanks the maintainers of NumPy, SciPy, Numba, CuPy and PyAMG. Ansys and Fluent are
trademarks of ANSYS, Inc. FlowTrace is an independent project, compatible with but not affiliated
with or endorsed by ANSYS, Inc.

# References
