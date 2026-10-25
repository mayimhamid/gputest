# Command line

`flowtrace <command> --help` lists every option. Running `flowtrace` with no command starts the
dashboard.

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

## Fluent warm start

```bash
flowtrace fluent-warmstart case.cas.h5 --turbulence k-omega-sst --bc inlet=inlet:10 --out ws/
```

This writes `flowtrace.ip`, `flowtrace.dat.h5` (for HDF5 input), `warmstart.jou`,
`warmstart_pyfluent.py`, `report.json` and a README. See [Fluent workflow](fluent.md).

## Verification and validation

```bash
flowtrace benchmark --list
flowtrace benchmark --tier quick --out benchmarks/results       # ~3 min
flowtrace benchmark --tier full --out benchmarks/results        # turbulence + warm start, ~1 h
flowtrace benchmark --only mms_simplec_tri,channel_retau395_sst
flowtrace warmstart-study --cases cavity,step --out ws_study/   # FlowTrace proxy (+ Fluent if licensed)
flowtrace parse-transcript cold.trn warm.trn                    # iterations from Fluent transcripts
flowtrace validate                                              # the original short suite
```
