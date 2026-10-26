# Square cylinder in a channel (GAMBIT meshes)

A real Fluent-format mesh set: the project author's four-mesh independence study of a
square cylinder, written by GAMBIT in 2016 (`SquareCylinder-{Coarse,Medium,Fine,Finest}.msh`,
5 824 to 22 984 quads). The square has side D = 4, centred 5 D downstream of the inlet, with
side walls at ±6 D (blockage D/H = 8.3 %) and the outlet 14.5 D behind the rear face.

FlowTrace reads the files as they are. It detects the zones without user input (velocity-inlet,
pressure-outlet, `SideWalls`, `BluffWalls`, an interior zone) and runs steady SIMPLEC at
U = 1, ρ = 1, μ = 4/Re. The coarsest mesh is part of the test suite
(`tests/test_square_cylinder.py`).

What this case exposed:

* **SIMPLEC diverged on these meshes**, even at Re = 4. About 9 000 faces of the Finest mesh
  are 44° non-orthogonal, with skew that does not alternate in sign. The stability guard now
  recovers from this automatically (see the changelog).
* The planner's 10-iteration window chose the GPU on Finest, where the full run was ~20 %
  slower than the CPU. See [Native kernels](../user-guide/native.md) for the current numbers.

## Mesh study at Re = 40

--8<-- "benchmarks/results/square_cylinder_meshes.md"

The recirculation length is grid-converged. Cd still drifts by a few tenths of a percent per
refinement level, with an irregular observed order. That is typical of the sharp corners,
which are flow singularities, so the GCI of Cd is an estimate rather than a bound. The no-slip
side walls of the original mesh raise Cd by about 11 % and shorten the bubble by about 18 %
compared with symmetry (slip) side walls.

## Physical consistency: Reynolds-number sweep (symmetry side walls)

Criteria were fixed before the runs:

* **P1:** Lr/D is linear in Re (R² ≥ 0.99). Sen, Mittal & Biswas (2011, *IJNMF* 67) report a
  linear growth of the bubble and the onset of steady separation at Re ≈ 1.15.
* **P2:** Cd ∝ Reⁿ with n = −0.66 ± 0.10 (same source).

--8<-- "benchmarks/results/square_cylinder_sweep.md"

P1 is the stronger check. Extrapolating the fitted line to zero length gives the separation
onset, a number the solver was never tuned to, and it lands next to the published 1.15.

P2 fails, and that failure is reported as it is. The domain test below shows that blockage
does not explain it: the exponent barely moves when the blockage drops from 8.3 % to 2.5 %.
The likely reason is the criterion itself. As far as we can tell from the abstract, −0.66
is a fit over a much wider Re range, while the local slope between Re = 10 and 40 is
shallower. For the circular cylinder, the classic Dennis & Chang (1970) values give about −0.45
over the same range. That explanation came after the result, so P2 stays **unresolved**.

## Domain-size effect (generated meshes, same resolution)

--8<-- "benchmarks/results/square_cylinder_domain.md"

The user-like generated domain reproduces the GAMBIT meshes. That validates the generator
before it is used for the large domain. In the large, strongly stretched domain, SIMPLEC
converges slowly; the iteration counts above show whether it reached the tolerance.

## Reproduce

```bash
python benchmarks/square_cylinder.py meshes --dir "<folder with the four .msh files>"
python benchmarks/square_cylinder.py sweep  --dir "<folder>" --mesh Fine
python benchmarks/square_cylinder.py domain
flowtrace batch <folder>/SquareCylinder-*.msh --mu 0.1 --tol 1e-7 --max-iter 8000   # all four at once
```
