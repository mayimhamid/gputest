"""``flowtrace fluent-warmstart``: pre-solve a Fluent case and hand the field back to Fluent.

Typical use: an engineer already has a Fluent case (mesh with named zones,
physics set up). FlowTrace reads the mesh from the same file, solves the flow
quickly with its pressure-based solver, and writes

* ``flowtrace.ip``           interpolation file (any Fluent mesh of the geometry can read it),
* ``flowtrace.dat.h5``       cell data on the *same* mesh (only for ``.cas.h5``/``.msh.h5`` input; experimental),
* ``warmstart.jou``          journal: read the original case, initialise, interpolate, iterate, save,
* ``warmstart_pyfluent.py``  the same through PyFluent,
* ``report.json``            settings, residual history, mass/heat balance and mesh quality.

The generated scripts read the **original** case file, so Fluent keeps the
physics the engineer set up; FlowTrace only supplies the initial field.
Ansys and Fluent are trademarks of ANSYS, Inc.; FlowTrace is an independent
project, compatible with but not affiliated with or endorsed by ANSYS, Inc.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict


DISCLAIMER = ("Ansys and Fluent are registered trademarks of ANSYS, Inc. FlowTrace is an independent project; "
              "it is compatible with, but not affiliated with or endorsed by, ANSYS, Inc.")

_VISCOUS_TUI = {"laminar": "/define/models/viscous/laminar yes",
                "k-omega-sst": "/define/models/viscous/kw-sst yes",
                "k-epsilon-realizable": "/define/models/viscous/ke-realizable yes",
                "k-epsilon-standard": "/define/models/viscous/ke-standard yes"}
_VISCOUS_PYFLUENT = {"laminar": "laminar", "k-omega-sst": "k-omega", "k-epsilon-realizable": "k-epsilon",
                     "k-epsilon-standard": "k-epsilon"}


def write_warmstart_scripts(out_dir: str, case_file: str, ip_file: str, case, iterations: int = 500,
                            setup_physics: bool = False, dim: int = 2) -> Dict[str, str]:
    q = lambda p: '"' + str(p).replace("\\", "/") + '"'  # noqa: E731
    files = {}
    jou = [f"; FlowTrace warm start for {os.path.basename(case_file)}", f"; {DISCLAIMER}",
           f"/file/read-case {q(case_file)}"]
    if setup_physics:
        jou.append(_VISCOUS_TUI[case.turbulence])
        if case.energy:
            jou += ["/define/models/energy yes", "", "", "", ""]
    jou += ["/solve/initialize/initialize-flow", "", f"/file/interpolate/read-data {q(ip_file)}", "", "",
            f"/solve/iterate {int(iterations)}", "/file/write-case-data \"warmstart_result.cas.h5\"", "/exit yes"]
    p = os.path.join(out_dir, "warmstart.jou")
    with open(p, "w", encoding="ascii") as fh:
        fh.write("\n".join(jou) + "\n")
    files["jou"] = p
    viscous = _VISCOUS_PYFLUENT[case.turbulence]
    sub = ""
    if case.turbulence == "k-omega-sst":
        sub = '    s.setup.models.viscous.k_omega_model = "sst"\n'
    elif case.turbulence == "k-epsilon-realizable":
        sub = '    s.setup.models.viscous.k_epsilon_model = "realizable"\n'
    physics = (f'    s.setup.models.viscous.model = "{viscous}"\n{sub}'
               + ('    s.setup.models.energy.enabled = True\n' if case.energy else "")) if setup_physics else "    pass\n"
    src = f'''"""FlowTrace warm start through PyFluent (needs ansys-fluent-core and a licensed Fluent).

{DISCLAIMER}
"""
import ansys.fluent.core as pyfluent

CASE = r"{case_file}"
IP_FILE = r"{ip_file}"
ITERATIONS = {int(iterations)}

solver = pyfluent.launch_fluent(mode="solver", dimension={dim}, precision="double", processor_count=4)
s = solver.settings
s.file.read_case(file_name=CASE)
try:  # physics as used by FlowTrace (only when --setup-physics was given)
{physics}except Exception as exc:  # Settings API names move between releases
    print("physics setup skipped:", exc)
s.solution.initialization.hybrid_initialize()
s.file.interpolate.read_data(file_name=IP_FILE)
s.solution.run_calculation.iterate(iter_count=ITERATIONS)
s.file.write(file_type="case-data", file_name="warmstart_result.cas.h5")
solver.exit()
'''
    p = os.path.join(out_dir, "warmstart_pyfluent.py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src)
    files["py"] = p
    return files


def run_warmstart(path: str, case, out_dir: str, iterations_fluent: int = 500, setup_physics: bool = False,
                  allow_bad_mesh: bool = False, log=print) -> dict:
    from ..hardware.planner import plan_unstructured
    from ..mesh.fluent_h5 import is_h5, read_any_mesh, write_data_h5
    from ..mesh.quality import mesh_quality
    from ..unstructured.case import default_boundaries
    from .export import write_interpolation_file

    t0 = time.perf_counter()
    mesh, fmt = read_any_mesh(path)
    log(f"Read {path} ({fmt}): {mesh.dim}D, {mesh.n_cells:,} cells, {mesh.n_faces:,} faces")
    qual = mesh_quality(mesh)
    log(qual.summary())
    if not qual.ok and not allow_bad_mesh:
        raise RuntimeError("mesh quality below Fluent's recommendations; fix the mesh or pass --allow-bad-mesh")
    bnd = default_boundaries(mesh)
    for k, v in case.boundaries.items():
        if k not in bnd:
            raise ValueError(f"--bc names zone {k!r}, which is not a boundary zone of this mesh "
                             f"({', '.join(sorted(bnd))})")
        bnd[k] = v
    case.boundaries = bnd
    t1 = time.perf_counter()  # solve time includes the planner's timed iterations
    plan, solver = plan_unstructured(mesh, case, log=log)
    log(f"Backend: {plan.backend} ({plan.device}), {plan.reason}")
    hist = []
    last = 0.0
    while not solver.finished:
        solver.step()
        r = solver.residuals()["scaled"]
        hist.append({"iteration": solver.iteration, **r})
        if time.perf_counter() - last > 2.0 or solver.finished:
            last = time.perf_counter()
            log(f"  it {solver.iteration:>6}  " + "  ".join(f"{k}={v:.2e}" for k, v in r.items()))
    solve_s = time.perf_counter() - t1
    for ev in getattr(solver, "guard_events", []):
        log(f"Stability guard: diverging at iteration {ev['detected_at']}, continuing from the state of iteration "
            f"{ev['restored_to']} with pressure relaxation {ev['relax_p']}")
    os.makedirs(out_dir, exist_ok=True)
    f = solver.fields()
    keep = ("x-velocity", "y-velocity", "z-velocity", "pressure", "temperature", "turb-kinetic-energy",
            "specific-diss-rate", "turb-diss-rate")
    fields = {k: v for k, v in f.items() if k in keep}
    files = {"ip": write_interpolation_file(os.path.join(out_dir, "flowtrace.ip"), solver.cell_centres(), fields)}
    if is_h5(path):
        files["dat_h5"] = write_data_h5(os.path.join(out_dir, "flowtrace.dat.h5"), mesh, fields, solver.iteration)
    files.update(write_warmstart_scripts(out_dir, os.path.abspath(path), os.path.abspath(files["ip"]), case,
                                         iterations_fluent, setup_physics, mesh.dim))
    report = {"input": os.path.abspath(path), "format": fmt, "cells": mesh.n_cells, "dim": mesh.dim,
              "iterations": solver.iteration, "converged": solver.converged, "solve_seconds": solve_s,
              "total_seconds": time.perf_counter() - t0, "plan": plan.to_dict(),
              "stability_rollbacks": list(getattr(solver, "guard_events", [])), "case": _jsonable(case.to_dict()),
              "mass_balance": solver.mass_balance(), "heat_flow": solver.heat_flow() if case.energy else {},
              "quality": qual.to_dict(), "residual_history": hist[-500:], "files": files, "disclaimer": DISCLAIMER}
    p = os.path.join(out_dir, "report.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=1)
    files["report"] = p
    readme = os.path.join(out_dir, "README_WARMSTART.txt")
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write("FlowTrace warm start\n====================\n"
                 f"Input case: {os.path.abspath(path)}\n"
                 f"FlowTrace: {solver.iteration} iterations ({'converged' if solver.converged else 'not converged'}),"
                 f" {solve_s:.1f} s\n\n"
                 "Fluent (batch):  fluent 2ddp -g -t4 -i warmstart.jou   (3ddp for 3-D)\n"
                 "PyFluent:        python warmstart_pyfluent.py\n"
                 "GUI:             read your case, Initialize, then File > Interpolate > Read and Interpolate flowtrace.ip\n\n"
                 + DISCLAIMER + "\n")
    files["readme"] = readme
    report["files"] = files
    return report


def _jsonable(o):
    from ..runtime.engine import _jsonable as j

    return j(o)
