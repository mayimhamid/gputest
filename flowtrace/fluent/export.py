"""Files that Ansys Fluent reads directly.

* Interpolation file (.ip, version 3, ASCII): File > Interpolate > Read and
  Interpolate. Fluent maps the point cloud onto *its own* mesh, so a FlowTrace
  solution can initialise any Fluent case of the same geometry ("warm start").
* Profile file (.prof, point profile): boundary-condition profiles, e.g. an
  inlet velocity/temperature distribution computed by FlowTrace.
* Journal (.jou): TUI commands to read the mesh/case, interpolate the
  FlowTrace data and iterate, for ``fluent 2ddp -g -i run.jou`` batch runs.
* PyFluent script (.py): the same workflow through the Settings API.
"""

from __future__ import annotations

import io
import os
from typing import Dict, Optional, Sequence

import numpy as np

FLUENT_FIELD_NAMES = {
    "x-velocity": "x-velocity",
    "y-velocity": "y-velocity",
    "z-velocity": "z-velocity",
    "pressure": "pressure",
    "temperature": "temperature",
}


def _block(values: np.ndarray, per_line: int = 5) -> str:
    v = np.asarray(values, dtype=np.float64).ravel()
    out = io.StringIO()
    out.write("(")
    for i in range(0, v.size, per_line):
        out.write(" ".join(f"{x:.10e}" for x in v[i:i + per_line]))
        out.write("\n")
    out.write(")\n")
    return out.getvalue()


def write_interpolation_file(path: str, points: np.ndarray, fields: Dict[str, np.ndarray]) -> str:
    """Write a Fluent interpolation file (format version 3, ASCII).

    ``points``: (n, 2|3) coordinates (cell centroids); ``fields``: Fluent
    variable names -> (n,) values. Non-finite values are dropped point-wise.
    """
    pts = np.asarray(points, dtype=np.float64)
    n, dim = pts.shape
    if dim not in (2, 3):
        raise ValueError("points must be (n, 2) or (n, 3)")
    names = [FLUENT_FIELD_NAMES.get(k, k) for k in fields]
    vals = [np.asarray(v, dtype=np.float64).ravel() for v in fields.values()]
    for v in vals:
        if v.size != n:
            raise ValueError("every field needs one value per point")
    ok = np.all(np.isfinite(pts), axis=1)
    for v in vals:
        ok &= np.isfinite(v)
    pts = pts[ok]
    vals = [v[ok] for v in vals]
    with open(path, "w", encoding="ascii") as fh:
        fh.write("3\n")
        fh.write(f"{dim}\n")
        fh.write(f"{pts.shape[0]}\n")
        fh.write(f"{len(names)}\n")
        for nm in names:
            fh.write(nm + "\n")
        for d in range(dim):
            fh.write(_block(pts[:, d]))
        for v in vals:
            fh.write(_block(v))
    return path


def read_interpolation_file(path: str):
    """Parse a version-3 ASCII interpolation file -> (points, {name: values})."""
    with open(path, encoding="ascii") as fh:
        txt = fh.read()
    head, _, rest = txt.partition("(")
    lines = head.split()
    version, dim, n, k = (int(x) for x in lines[:4])
    if version != 3:
        raise ValueError("only version-3 ASCII interpolation files are supported")
    names = lines[4:4 + k]
    blocks = ("(" + rest).split(")")
    arrays = []
    for b in blocks:
        b = b.strip()
        if b.startswith("("):
            arrays.append(np.array(b[1:].split(), dtype=np.float64))
    pts = np.stack(arrays[:dim], axis=1)
    return pts, {nm: arrays[dim + i] for i, nm in enumerate(names)}


def write_profile(path: str, name: str, points: np.ndarray, fields: Dict[str, np.ndarray]) -> str:
    """Write a Fluent point profile (.prof)."""
    pts = np.asarray(points, dtype=np.float64)
    n, dim = pts.shape
    axes = "xyz"[:dim]
    with open(path, "w", encoding="ascii") as fh:
        fh.write(f"(({name} point {n})\n")
        for d, ax in enumerate(axes):
            fh.write(f"({ax}\n" + "\n".join(f"{x:.10e}" for x in pts[:, d]) + "\n)\n")
        for fname, v in fields.items():
            v = np.asarray(v, dtype=np.float64).ravel()
            fh.write(f"({FLUENT_FIELD_NAMES.get(fname, fname)}\n" + "\n".join(f"{x:.10e}" for x in v) + "\n)\n")
        fh.write(")\n")
    return path


def write_journal(path: str, mesh_file: str, ip_file: Optional[str], iterations: int = 200,
                  case_out: str = "flowtrace_result.cas.h5", energy: bool = False,
                  viscous: str = "laminar", extra_commands: Sequence[str] = ()) -> str:
    """Write a Fluent TUI journal for a batch run (``fluent 2ddp -g -i run.jou``).

    TUI prompts can differ slightly between Fluent releases; the journal keeps
    to long-standing commands and the PyFluent script is the robust alternative.
    """
    q = lambda p: '"' + str(p).replace("\\", "/") + '"'  # noqa: E731
    lines = [
        "; FlowTrace -> Ansys Fluent warm-start journal",
        "; run with:  fluent 2ddp -g -t4 -i " + os.path.basename(path),
        f"/file/read-case {q(mesh_file)}",
        "/mesh/check",
    ]
    # Blank lines accept the default answer of any follow-up prompt a given
    # Fluent release may ask, and are ignored at the top-level prompt.
    if viscous == "laminar":
        lines += ["/define/models/viscous/laminar yes", ""]
    if energy:
        lines += ["/define/models/energy yes", "", "", "", ""]
    lines += list(extra_commands)
    lines += ["/solve/initialize/initialize-flow", ""]
    if ip_file:
        lines += [f"/file/interpolate/read-data {q(ip_file)}", "", ""]
    lines += [
        f"/solve/iterate {int(iterations)}",
        f"/file/write-case-data {q(case_out)}",
        "/exit yes",
    ]
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_pyfluent_script(path: str, mesh_file: str, ip_file: Optional[str], dim: int = 2,
                          iterations: int = 200, energy: bool = False, processors: int = 4,
                          boundaries: Optional[dict] = None, density: float = 1.0,
                          viscosity: float = 1e-3) -> str:
    """Write a stand-alone PyFluent script reproducing the FlowTrace case in Fluent."""
    bc_lines = []
    for name, spec in (boundaries or {}).items():
        kind = spec.get("kind")
        if kind == "inlet" and spec.get("normal_speed") is not None:
            bc_lines.append(
                f'    bc.velocity_inlet["{name}"].momentum.velocity_magnitude.value = {float(spec["normal_speed"])}')
        elif kind == "wall" and spec.get("velocity"):
            v = spec["velocity"]
            bc_lines.append(f'    w = bc.wall["{name}"].momentum')
            bc_lines.append('    w.wall_motion = "Moving Wall"')
            bc_lines.append('    w.relative = False')
            speed = float(np.linalg.norm(v))
            bc_lines.append(f'    w.speed = {speed}')
            if speed > 0:
                bc_lines.append(f'    w.direction.value = {[float(x) / speed for x in v] + [0.0] * (3 - len(v))}')
        elif kind == "outlet":
            bc_lines.append(f'    bc.pressure_outlet["{name}"].momentum.gauge_pressure.value = {float(spec.get("pressure", 0.0))}')
    thermal = [f"{name}: {spec.get('thermal', 'temperature') if spec.get('kind') != 'inlet' else 'inlet temperature'}"
               f" = {spec.get('value', 0.0)}"
               for name, spec in (boundaries or {}).items()
               if energy and (spec.get("thermal") in ("temperature", "flux") or spec.get("kind") == "inlet")]
    bc_block = "\n".join(bc_lines) if bc_lines else "    pass"
    thermal_block = "\n".join(f"#   {t}" for t in thermal) if thermal else "#   (none)"
    src = f'''"""Generated by FlowTrace: reproduce and finish the case in Ansys Fluent via PyFluent.

pip install ansys-fluent-core   (needs a licensed Fluent installation)
"""
import ansys.fluent.core as pyfluent

MESH = r"{mesh_file}"
IP_FILE = {repr(ip_file) if ip_file else None}

solver = pyfluent.launch_fluent(mode="solver", dimension={dim}, precision="double",
                                processor_count={processors})
s = solver.settings
s.file.read_case(file_name=MESH)
s.setup.models.viscous.model = "laminar"
{"s.setup.models.energy.enabled = True" if energy else "# energy model off"}
try:
    air = s.setup.materials.fluid["air"]
    air.density.value = {density}
    air.viscosity.value = {viscosity}
except Exception as exc:  # material API differs between releases
    print("material update skipped:", exc)
bc = s.setup.boundary_conditions
# Thermal conditions used by FlowTrace (thermal BC names differ between Fluent
# releases, so set these in the Boundary Conditions panel if needed):
{thermal_block}
try:
{bc_block}
except Exception as exc:
    print("boundary-condition update skipped:", exc)
s.solution.initialization.hybrid_initialize()
if IP_FILE:
    s.file.interpolate.read_data(file_name=IP_FILE)
s.solution.run_calculation.iterate(iter_count={int(iterations)})
s.file.write(file_type="case-data", file_name="flowtrace_result.cas.h5")
solver.exit()
'''
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return path
