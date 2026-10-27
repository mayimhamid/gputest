# Command line

`flowtrace <command> --help` lists every option. Running `flowtrace` with no command starts the
dashboard (`flowtrace ui`, local only; see [Dashboard](dashboard.md#security) for `--host` and
`--allow-remote`).

## Exit status and errors

| Status | Meaning |
|---:|---|
| 0 | success |
| 1 | the run failed (for example it diverged: the error line gives the iteration, the mesh's non-orthogonality and the stability guard's state), or a result fails its check (`benchmark`, `validate`, a failed case in `batch`) |
| 2 | usage or input error: bad option, missing or unreadable file, malformed mesh, unknown zone name, backend not available, not enough memory. `flowtrace quality` also returns 2 when a quality metric is below Fluent's limits |
| 130 | interrupted (Ctrl-C) |

Input errors and diverged runs print one line, `flowtrace <command>: error: <cause>`, instead of
a Python traceback. That applies whether the error is found before the run starts or by the
solver. An unexpected fault keeps its traceback. Set `FLOWTRACE_DEBUG=1` to see the traceback
in every case (for a bug report).

## Inspecting the machine and meshes

```bash
flowtrace probe                      # CPU, RAM, GPUs, libraries, available backends
flowtrace bench                      # micro-benchmarks of every backend
flowtrace gpu-report --out results/  # GPU vs CPU correctness + ms/step per grid (Markdown + JSON)
flowtrace mesh case.msh              # zones, periodic pairs and the mesh quality report
flowtrace quality case.cas.h5        # quality only; exit code 2 if below Fluent's limits
flowtrace h5-inspect case.cas.h5     # HDF5 tree (debugging files from other tools)
```

## Structured (MAC) cases

```bash
flowtrace run cavity --re 1000 --n 256 [--backend cupy] [--export out/]
flowtrace run natural_convection --ra 1e5 --n 128
flowtrace run cylinder --re 100 --n-per-d 24 --t-end 150
flowtrace run step --re 400 --mode steady-simplec    # same case through the SIMPLEC solver
```

Presets: `cavity`, `natural_convection`, `channel`, `cylinder`, `step`, `rayleigh_benard`,
`taylor_green`.

## Unstructured (Fluent mesh) cases

```bash
flowtrace solve mesh.msh --bc inlet=inlet:1.0 --bc outlet=outlet --mu 1.8e-5 --rho 1.2 --export out/
```

| Option | Meaning |
|---|---|
| `--bc zone=kind[:values[:T\|q]]` | `inlet=inlet:1.0:T300`, `lid=wall:1,0`, `hot=wall::T1`, `out=outlet:0` |
| `--turbulence` | `laminar`, `k-omega-sst`, `k-epsilon-realizable`, `k-epsilon-standard` |
| `--energy`, `--k`, `--cp` | energy equation, fluid conductivity, heat capacity |
| `--solid zone:k[:q]` | conjugate heat transfer: solid cell zone, conductivity, volumetric source |
| `--mrf zone:omega[:x0,y0]` | moving reference frame for a cell zone |
| `--periodic a:b` | translational periodic pair (zone b = zone a shifted) |
| `--body-force fx,fy` | uniform force per volume (e.g. the driving pressure gradient of a periodic channel) |
| `--max-iter`, `--tol` | iteration limit and scaled-residual tolerance |

`.msh`, legacy `.cas` (gzipped or not) and CFF `.cas.h5`/`.msh.h5` files are all accepted.

### Automatic options and mesh units

```bash
flowtrace solve examples/elbow-mix.msh.txt --auto --units auto --rho 998.2 --mu 1.003e-3 --k 0.6 --cp 4182     --bc inf1=inlet:0.4:T293 --bc inf2=inlet:1.2:T313 --bc out=outlet:0
```

`--auto` prints one line per decision (Re and the estimated y+ choose the turbulence model, different
inlet temperatures switch the energy equation on, a skewed mesh gets first-order energy transport so
that the temperature stays between the boundary values). `--units` is the length unit of the mesh file
(Fluent: General > Scale); `auto` reads an extent of 20 or more as millimetres, which the log states.
The fluid properties and boundary values are never changed.

## Many cases at once: `flowtrace batch`

A mesh-independence study, a Reynolds-number sweep or a set of design variants is a group of
independent solves. A single small or medium case cannot keep the machine busy: its CPU
iteration runs on one core, and on the GPU it sits near the launch floor. `flowtrace batch`
runs the cases concurrently. One GPU worker takes the cases the cost model says the GPU is
faster for, largest first. Every other physical core runs one case on the CPU.

```bash
# mesh-independence study: four meshes, same physics
flowtrace batch Coarse.msh Medium.msh Fine.msh Finest.msh --mu 0.1 --tol 1e-7 --max-iter 8000 --out runs/

# Reynolds sweep on one mesh (viscosity = rho U D / Re)
flowtrace batch Fine.msh --set viscosity=0.4,0.2,0.1333,0.1 --bc SideWalls=symmetry --out sweep/
```

| Option | Meaning |
|---|---|
| all `solve` options | physics and boundary conditions shared by every case |
| `--set key=v1,v2,..` | sweep a case parameter (repeat for a full factorial) |
| `--cpu-workers N` | concurrent CPU cases (default: physical cores, minus one that drives the GPU) |
| `--no-gpu` | CPU workers only |
| `--out DIR` | `<case>.ip` (Fluent interpolation file), `<case>.npz` (all fields) and `batch.json` |

The summary lists each case's device, iterations, convergence and the **force on every wall
zone** (pressure + viscous, per unit depth in 2-D). The same report is available from Python as
`solver.wall_forces()`, the equivalent of Fluent's *Report > Forces*. It is built from the
momentum equation's own wall terms, so a periodic channel driven by a body force balances
exactly (test `test_wall_forces_balance_the_driving_body_force`).

Each case runs in its own process with an explicit backend. Its result therefore does not
depend on what ran beside it, apart from round-off noise of about 1e-8 between processes.

## Compressible flow: `flowtrace compressible`

Density-based solver (HLLC, Euler / Navier–Stokes / k-ω SST) on any Fluent mesh, including
polyhedral and axisymmetric ones; see [the theory page](../theory/compressible.md).

```bash
flowtrace compressible nozzle.msh --case nozzle.json --export out/
```

| Option | Meaning |
|---|---|
| `--case FILE` | JSON of `CompressibleCase` settings with a `boundaries` map (zone → kind and values) |
| `--backend` | `numpy`, `numba`, `cupy` (default: automatic) |
| `--export DIR` | `<mesh>.ip` for Fluent's *File → Interpolate* and `<mesh>.npz` |

From Python, `BatchJob(name, mesh, case, solver="density-based")` runs compressible cases in
`run_batch` next to pressure-based ones, on every CPU core and the GPU.

## Fluent warm start

```bash
flowtrace fluent-warmstart case.cas.h5 --turbulence k-omega-sst --bc inlet=inlet:10 --out ws/
```

This writes `flowtrace.ip`, `flowtrace.dat.h5` (for HDF5 input), `warmstart.jou`,
`warmstart_pyfluent.py`, `report.json` and a README. See [Fluent workflow](fluent.md).

## Verification and validation

```bash
flowtrace benchmark --list
flowtrace benchmark --tier quick --out benchmarks/results       # ~2 min
flowtrace benchmark --tier full --out benchmarks/results        # turbulence + warm start, ~40 min
flowtrace benchmark --only mms_simplec_tri,channel_retau395_sst
flowtrace warmstart-study --cases cavity,step --out ws_study/   # FlowTrace proxy (+ Fluent if licensed)
flowtrace parse-transcript cold.trn warm.trn                    # iterations from Fluent transcripts
flowtrace validate                                              # the original short suite
```
