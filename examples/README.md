# Examples

| File | What to do with it |
|---|---|
| `heated_duct_tri.msh` | Fluent ASCII mesh (2 400 skewed triangles) with zones `inlet` (velocity-inlet), `outlet` (pressure-outlet), `floor`, `ceiling` (walls). Drag it onto the dashboard's *Fluent mesh* tab, tick *Energy*, set `floor` thermal to `T=1` and `inlet` to `T=0`, run. |
| `cavity_quad.msh` | Fluent **binary** mesh of a 40×40 square cavity with zone `lid`. Set `lid` to *wall* with value `1,0` (moving lid). |

Command-line equivalents:

```bash
python -m flowtrace mesh examples/heated_duct_tri.msh
python -m flowtrace solve examples/heated_duct_tri.msh --energy --mu 0.02 --k 0.02 \
    --bc inlet=inlet:1.0:T0 --bc floor=wall::T1 --export out_duct/
python -m flowtrace solve examples/cavity_quad.msh --mu 0.01 --bc lid=wall:1,0 --export out_cavity/
```

The `--export` directory contains the Fluent hand-off package (`flowtrace.msh`,
`flowtrace.ip`, `run_fluent.jou`, `run_pyfluent.py`, `flowtrace_cells.csv`).
