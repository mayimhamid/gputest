"""Run a density-based case on a Fluent mesh (shared by ``flowtrace compressible`` and the batch runner)."""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import numpy as np

from .case import CompressibleCase, default_comp_boundaries
from .solver import CompressibleSolver

FIELDS_FOR_FLUENT = ("pressure", "x-velocity", "y-velocity", "z-velocity", "temperature", "density",
                     "turb-kinetic-energy", "specific-diss-rate")


def build_case(mesh, case: Optional[dict] = None, backend: Optional[str] = None) -> CompressibleCase:
    kw = dict(case or {})
    bnd = default_comp_boundaries(mesh)
    for k, v in (kw.pop("boundaries", None) or {}).items():
        if k not in bnd:
            raise ValueError(f"zone {k!r} is not a boundary zone of the mesh ({', '.join(sorted(bnd))})")
        bnd[k] = v
    if backend is not None:
        kw["backend"] = backend
    return CompressibleCase(boundaries=bnd, **kw)


def solve(mesh, case: CompressibleCase, out_dir: str = "", name: str = "compressible",
          callback: Optional[Callable] = None, backend=None) -> dict:
    """Iterate to convergence (or max_iter / t_end); optionally write <name>.ip and <name>.npz."""
    solver = CompressibleSolver(mesh, case, backend=backend)
    t0 = time.perf_counter()
    while not solver.finished:
        solver.step()
        if callback is not None and solver.iteration % case.monitor_interval == 0:
            callback(solver)
    res = {"cells": mesh.n_cells, "device": solver.be.device, "iterations": solver.iteration,
           "converged": bool(solver.converged), "solve_seconds": time.perf_counter() - t0,
           "residuals": solver.residuals()["scaled"], "mass_balance": solver.mass_balance(),
           "describe": solver.describe()}
    if out_dir:
        from ..fluent.export import write_interpolation_file

        os.makedirs(out_dir, exist_ok=True)
        f = solver.fields()
        base = os.path.join(out_dir, name)
        res["ip"] = write_interpolation_file(base + ".ip", solver.cell_centres(),
                                             {k: v for k, v in f.items() if k in FIELDS_FOR_FLUENT})
        np.savez_compressed(base + ".npz", centres=solver.cell_centres(), **f)
        res["npz"] = base + ".npz"
    res["solver"] = solver
    return res
