# Trademarks and licence

## Trademarks

Ansys and Fluent are registered trademarks of ANSYS, Inc. or its subsidiaries in the United
States and other countries.

FlowTrace is an independent open-source project. It is **not affiliated with, sponsored by or
endorsed by ANSYS, Inc.** The names appear in this project only to describe compatibility
(nominative use): FlowTrace reads and writes file formats that Ansys Fluent uses, and it
generates journals and PyFluent scripts for a Fluent installation the user is licensed to run.
The project does not use Ansys logos, and it ships no Ansys software, documentation or data.

The HDF5 case/data layout implemented here follows the open-source VTK reader
(`vtkFLUENTCFFReader`). No proprietary specification was used.

## Third-party data

* Lid-driven cavity centreline velocities: Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387–411.
* Natural convection: de Vahl Davis, *Int. J. Numer. Meth. Fluids* 3 (1983) 249–264.
* Turbulent channel: 16 points sampled from `chan395.means`, Moser, Kim & Mansour,
  *Phys. Fluids* 11 (1999) 943–945. The data are distributed by the authors for free use with
  citation.
* Backward-facing step reattachment length: Driver & Seegmiller, *AIAA J.* 23 (1985) 163–171.
* Flat-plate correlations: F. M. White, *Viscous Fluid Flow* (1974); Schultz-Grunow (1940).

## Licence

FlowTrace is released under the [MIT licence](https://opensource.org/license/mit).

## Using FlowTrace for engineering decisions

FlowTrace comes without warranty (see the licence). The [validation report](validation/index.md)
documents what has been checked and against what. Results for other flows must be verified for
your application, for example with a grid-convergence study, a comparison with test data, or a
finished Fluent run started from the FlowTrace field.
