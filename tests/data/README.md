# Test data

`SquareCylinder-Coarse.msh`: the coarsest mesh of the project author's square-cylinder
mesh-independence set. It is an ASCII Fluent mesh written by GAMBIT in 2016: 5 824 quads,
zones `inlet` (velocity-inlet), `outlet` (pressure-outlet), `SideWalls`, `BluffWalls`, plus
an interior zone. It is included with the author's permission as a real-world regression case
(`tests/test_square_cylinder.py`). The finer meshes of the set are used by
`benchmarks/square_cylinder.py` but are not stored in the repository.
