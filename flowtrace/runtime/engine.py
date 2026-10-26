"""Simulation jobs: plan -> (FMG) -> iterate -> monitor -> validate -> export.

A :class:`Job` runs in its own thread and publishes JSON-able events
(status, plan, residuals, frames, log lines, results) to an :class:`EventHub`
that any number of listeners (the web UI over Server-Sent Events, the CLI)
can subscribe to. The solver thread never waits for listeners: frames are
throttled by wall-clock time and downsampled before leaving the device.
"""

from __future__ import annotations

import base64
import os
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Optional

import numpy as np

from ..hardware.backend import get_backend
from .checkpoint import CheckpointWriter


class EventHub:
    def __init__(self, history: int = 4000):
        self._subs = set()
        self._lock = threading.Lock()
        self.residuals = deque(maxlen=history)
        self.log = deque(maxlen=500)
        self.latest = {}

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=256)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict) -> None:
        typ = event.get("type")
        if typ == "residuals":
            self.residuals.append(event)
        elif typ == "log":
            self.log.append(event)
        else:
            self.latest[typ] = event
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # slow listener: drop its oldest event rather than stall the solver
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except queue.Empty:
                    pass

    def snapshot(self) -> dict:
        return {"residuals": list(self.residuals), "log": list(self.log), "latest": dict(self.latest)}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode("ascii")


def _display_range(field: str, a: np.ndarray):
    """Colour range robust to isolated spikes (e.g. vorticity at staircase corners)."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return 0.0, 1.0
    if field == "vorticity":
        m = float(np.percentile(np.abs(v), 99.0)) or float(np.abs(v).max()) or 1.0
        return -m, m
    lo, hi = np.percentile(v, [0.1, 99.9]) if v.size > 1000 else (v.min(), v.max())
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max()) + 1e-30
    return float(lo), float(hi)


def _downsample(a: np.ndarray, max_n: int) -> np.ndarray:
    """Block-average a 2-D (nx, ny) array so that both sides are <= max_n (NaN aware)."""
    fx = max(1, int(np.ceil(a.shape[0] / max_n)))
    fy = max(1, int(np.ceil(a.shape[1] / max_n)))
    if fx == 1 and fy == 1:
        return a
    nx, ny = a.shape[0] // fx * fx, a.shape[1] // fy * fy
    b = a[:nx, :ny].reshape(nx // fx, fx, ny // fy, fy)
    with np.errstate(invalid="ignore"):
        cnt = np.sum(np.isfinite(b), axis=(1, 3))
        s = np.nansum(b, axis=(1, 3))
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


class Job:
    """One simulation run. ``spec`` is the JSON request coming from the UI/CLI."""

    def __init__(self, spec: dict, hub: EventHub, workdir: str, meshes: Optional[dict] = None):
        self.id = uuid.uuid4().hex[:8]
        self.spec = spec
        self.hub = hub
        self.workdir = workdir
        self.meshes = meshes or {}
        self.kind = spec.get("kind", "structured")
        self.status = "created"
        self.error = None
        self.solver = None
        self.plan = None
        self.result = {}
        self.view_field = spec.get("field")
        self.frame_interval = float(spec.get("frame_interval", 0.25))
        self.max_frame = int(spec.get("max_frame", 384))
        self._stop = threading.Event()
        self._thread = None
        self.started = None
        self.finished_at = None
        self.probe_log = {"t": [], "v": [], "u": []}
        self.mesh = None

    # ------------------------------------------------------------------
    def log(self, msg: str, level: str = "info"):
        self.hub.publish({"type": "log", "level": level, "msg": msg, "t": time.time()})

    def set_status(self, status: str, **extra):
        self.status = status
        self.hub.publish({"type": "status", "status": status, "job": self.id, "kind": self.kind, **extra})

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f"flowtrace-job-{self.id}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run(self):
        self.started = time.time()
        try:
            if self.kind == "structured":
                self._run_structured()
            else:
                self._run_unstructured()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(traceback.format_exc(), "error")
            self.set_status("error", error=self.error)
        finally:
            self.finished_at = time.time()

    # ---------------- structured (MAC) ----------------
    def _structured_case(self):
        from ..structured.case import StructuredCase, make_preset

        spec = self.spec
        if spec.get("preset"):
            case = make_preset(spec["preset"], **spec.get("params", {}))
        else:
            case = StructuredCase.from_dict(spec["case"])
        for k, v in (spec.get("overrides") or {}).items():
            if not hasattr(case, k):
                raise ValueError(f"unknown case field {k!r}")
            setattr(case, k, v)
        return case

    def _run_structured(self):
        from ..hardware.planner import plan_structured
        from ..structured.sequencing import coarse_case, prolong
        from ..structured.solver import MACSolver

        case = self._structured_case()
        self.case = case
        self.set_status("planning", name=case.name)
        self.log(f"Case: {case.name}  grid {case.nx}x{case.ny}  ({case.cells:,} cells)")
        if self.spec.get("mode") == "steady-simplec":
            return self._run_structured_as_simplec(case)
        plan, solver = plan_structured(case, autotune=self.spec.get("autotune", True))
        self.plan = plan.to_dict()
        self.hub.publish({"type": "plan", "plan": self.plan, "describe": solver.describe()})
        for p in plan.placements:
            self.log(f"[plan] {p.stage}: {p.device} | {p.method} - {p.reason}")
        for n in plan.notes:
            self.log(f"[plan] {n}")
        use_fmg = self.spec.get("fmg", True) and plan.fmg
        if use_fmg:
            prev = None
            for f in plan.fmg:
                cc = coarse_case(case, f)
                cs = MACSolver(cc, backend=solver.be, dtype=solver.dtype)
                if prev is not None:
                    prolong(prev, cs)
                self.log(f"[FMG] level 1/{f}: {cc.nx}x{cc.ny}")
                self._loop(cs, label=f"FMG 1/{f}", publish_frames=True)
                if self._stop.is_set():
                    break
                prev = cs
            if prev is not None and not self._stop.is_set():
                prolong(prev, solver)
                solver.step_count = 0
                self.log("[FMG] prolongated to the fine grid")
        self.solver = solver
        if not self._stop.is_set():
            self._loop(solver, label="fine")
        self._finish_structured(solver)

    def _run_structured_as_simplec(self, case):
        from ..fluent.convert import structured_boundary_specs, structured_mesh
        from ..structured.solver import MACSolver
        from ..unstructured.case import UnstructuredCase

        probe = MACSolver(case, backend=get_backend("numpy"))  # masks only, cheap
        mesh = structured_mesh(case, probe.solid_h)
        ucase = UnstructuredCase(
            name=case.name + " (SIMPLEC)", boundaries=structured_boundary_specs(case),
            density=1.0, viscosity=case.nu, energy=case.energy, conductivity=case.kappa, cp=1.0,
            buoyancy=list(case.buoyancy) + [0.0], t_ref=case.t_ref, max_iter=int(self.spec.get("max_iter", 5000)),
            tol=float(self.spec.get("tol", 1e-5)), backend=self.spec.get("backend", "auto"),
            init_T=case.init_T,
        )
        self.mesh = mesh
        self._run_unstructured(mesh=mesh, ucase=ucase)

    def _finish_structured(self, solver):
        from ..validation.checks import validate_structured

        res = {"steps": solver.step_count, "time": solver.t, "converged": solver.converged,
               "describe": solver.describe(), "timers": dict(solver.timers)}
        try:
            res["validation"] = validate_structured(solver, self.probe_log)
            for ch in res["validation"]["checks"]:
                mark = "PASS" if ch.get("pass") else "CHECK"
                ref = f" (ref {ch['reference']:g})" if ch.get("reference") is not None else ""
                self.log(f"[validation] {mark} {ch['name']}: {ch['value']:.4g} {ch.get('unit', '')}{ref}")
        except Exception as exc:
            self.log(f"validation skipped: {exc}", "warn")
        self.result = res
        self._publish_frame(solver, force=True)
        status = "stopped" if self._stop.is_set() else ("converged" if solver.converged else "finished")
        self.set_status(status, result=_jsonable(res))

    # ---------------- shared iteration loop ----------------
    def _loop(self, solver, label: str, publish_frames: bool = True):
        c = solver.case
        mon = max(1, int(getattr(c, "monitor_interval", 10) or 10))
        ckpt_every = float(self.spec.get("checkpoint_seconds", 0) or 0)
        writer = CheckpointWriter(os.path.join(self.workdir, "checkpoints", self.id)) if ckpt_every > 0 else None
        last_frame = 0.0
        last_ckpt = time.time()
        self.set_status("running", label=label)
        t_start = time.perf_counter()
        steps0 = solver.step_count
        cells = getattr(solver, "nc", None) or (solver.nx * solver.ny)
        while not solver.finished and not self._stop.is_set():
            solver.step()
            k = solver.step_count
            if k % mon == 0 or solver.finished:
                r = solver.residuals()
                elapsed = time.perf_counter() - t_start
                rate = (k - steps0) / elapsed if elapsed > 0 else 0.0
                ev = {"type": "residuals", "label": label, "step": k, "t": getattr(solver, "t", k),
                      "dt": getattr(solver, "dt", None), "scaled": r["scaled"], "raw": r["raw"],
                      "steps_per_s": rate, "mcells_per_s": rate * cells / 1e6}
                if hasattr(solver, "kinetic_energy"):
                    ev["ke"] = solver.kinetic_energy()
                    nu = solver.nusselt("left") if solver.energy else None
                    if nu is not None:
                        ev["nusselt"] = nu
                    if solver.case.probes:
                        pv = solver.sample_probes()[0]
                        ev["probe"] = pv
                        self.probe_log["t"].append(solver.t)
                        self.probe_log["u"].append(pv[0])
                        self.probe_log["v"].append(pv[1])
                if hasattr(solver, "last_linear"):
                    ev["linear"] = solver.last_linear
                ev["timers"] = dict(solver.timers)
                self.hub.publish(_jsonable(ev))
                if not all(np.isfinite(v) for v in r["raw"].values()):
                    raise FloatingPointError("solution diverged (non-finite residuals)")
            now = time.time()
            if publish_frames and now - last_frame >= self.frame_interval:
                self._publish_frame(solver)
                last_frame = now
            if writer and now - last_ckpt >= ckpt_every:
                writer.submit(f"step{k:08d}", solver.state_dict(), {"case": _jsonable(c.to_dict()), "label": label})
                last_ckpt = now
        if writer:
            writer.submit(f"step{solver.step_count:08d}", solver.state_dict(),
                          {"case": _jsonable(c.to_dict()), "label": label})
            writer.close()
            self.log(f"checkpoints: {len(writer.written)} written to {writer.directory} "
                     f"({writer.bytes_written / 1e6:.1f} MB in {writer.seconds:.2f} s, overlapped)")

    def _publish_frame(self, solver, force: bool = False):
        if hasattr(solver, "nx"):
            self.hub.publish(self.structured_frame(solver, self.view_field))
        elif solver.dim == 2:
            self.hub.publish(self.unstructured_frame(solver, self.view_field))

    def structured_frame(self, solver, field: Optional[str] = None) -> dict:
        names = ["velocity-magnitude", "x-velocity", "y-velocity", "pressure", "vorticity", "stream-function"]
        if solver.energy:
            names.insert(0, "temperature")
        field = field if field in names else names[0]
        f = solver.fields([field])[field]
        img = _downsample(f, self.max_frame)
        u, v = solver.fields(["x-velocity", "y-velocity"]).values()
        nv = 64
        uu, vv = _downsample(u, nv), _downsample(v, nv)
        lo, hi = _display_range(field, f)
        return {"type": "frame", "mesh": "structured", "field": field, "fields": names,
                "nx": img.shape[0], "ny": img.shape[1], "lx": solver.case.lx, "ly": solver.case.ly,
                "data": _b64(img.T), "min": lo, "max": hi,
                "vnx": uu.shape[0], "vny": uu.shape[1], "u": _b64(uu.T), "v": _b64(vv.T),
                "step": solver.step_count, "t": solver.t, "grid": [solver.nx, solver.ny]}

    def unstructured_frame(self, solver, field: Optional[str] = None) -> dict:
        f = solver.fields()
        names = list(f.keys())
        field = field if field in names else ("temperature" if "temperature" in f else "velocity-magnitude")
        vals = f[field]
        lo, hi = _display_range(field, vals)
        return {"type": "frame", "mesh": "unstructured", "field": field, "fields": names,
                "data": _b64(vals), "min": lo, "max": hi,
                "step": solver.iteration, "t": solver.iteration, "cells": solver.nc}

    # ---------------- unstructured (Fluent mesh) ----------------
    def _run_unstructured(self, mesh=None, ucase=None):
        from ..hardware.memory import fmt_bytes
        from ..hardware.planner import plan_unstructured
        from ..unstructured.case import UnstructuredCase, default_boundaries

        if mesh is None:
            mesh = self.meshes.get(self.spec.get("mesh_id"))
            if mesh is None:
                raise ValueError("no mesh uploaded (mesh_id not found)")
        self.mesh = mesh
        if ucase is None:
            cd = dict(self.spec.get("case") or {})
            bnd = default_boundaries(mesh)
            for k, v in (cd.pop("boundaries", None) or {}).items():
                bnd[k] = v
            ucase = UnstructuredCase(boundaries=bnd, **cd)
        self.case = ucase
        self.set_status("planning", name=ucase.name)
        self.log(f"Mesh {mesh.name}: {mesh.n_cells:,} cells, {mesh.n_faces:,} faces, {mesh.dim}D")
        uplan, solver = plan_unstructured(mesh, ucase, log=self.log)
        be = get_backend(uplan.backend)
        est = uplan.memory.get(uplan.backend)
        if est:
            self.log(f"Memory budget: ~{fmt_bytes(est['required'])} of {fmt_bytes(est['available'])} free on {be.device}")
        for name, why in uplan.skipped.items():
            self.log(f"Planner: {name} {why}", "warn" if "fit" in why else "info")
        self.log(f"Planner: {be.name} ({be.device}), {uplan.reason}")
        self.solver = solver
        desc = solver.describe()
        plan = {
            "backend": be.name, "device": be.device, "precision": desc["precision"],
            "pressure": desc["pressure_solver"], "diffusion": "implicit",
            "memory": uplan.memory, "measured_ms_per_step": uplan.measured_ms_per_iteration, "fmg": [],
            "placements": [
                {"stage": "Mesh import & FV geometry", "device": "CPU", "method": "vectorised NumPy (once)",
                 "reason": f"{desc['setup_seconds']} s, max non-orthogonality {desc['max_non_orthogonality_deg']} deg"},
                {"stage": "Gradients, assembly, momentum", "device": be.device.upper(),
                 "method": "least-squares gradients, CSR assembly, Jacobi-BiCGSTAB", "reason": uplan.reason},
                {"stage": "Pressure correction", "device": be.device.upper(), "method": desc["pressure_solver"],
                 "reason": "AMG makes the elliptic solve mesh-size independent"},
                {"stage": "AMG hierarchy setup", "device": "CPU", "method": "smoothed aggregation, rebuilt every 10 its",
                 "reason": "irregular graph work stays on the host"},
            ],
            "notes": [f"{ucase.algorithm}, {ucase.scheme}, relax u={ucase.relax_u} p={desc.get('relax_p', ucase.relax_p)}"],
        }
        self.plan = plan
        self.hub.publish({"type": "plan", "plan": plan, "describe": desc})
        if mesh.dim == 2:
            self.hub.publish(self.mesh_geometry_event(mesh))
        self._loop(solver, label="SIMPLE")
        res = {"iterations": solver.iteration, "converged": solver.converged, "describe": desc,
               "timers": dict(solver.timers), "mass_balance": solver.mass_balance(),
               "stability_rollbacks": list(getattr(solver, "guard_events", []))}
        self.result = res
        for ev in res["stability_rollbacks"]:
            self.log(f"Stability guard: diverging at iteration {ev['detected_at']}, continuing from the state of iteration "
                     f"{ev['restored_to']} with pressure relaxation {ev['relax_p']}", "warn")
        mb = res["mass_balance"]
        net = sum(mb.values())
        self.log(f"Mass balance: net {net:.3e} ({', '.join(f'{k} {v:.4g}' for k, v in mb.items())})")
        if mesh.dim == 2:
            self.hub.publish(self.unstructured_frame(solver, self.view_field))
        status = "stopped" if self._stop.is_set() else ("converged" if solver.converged else "finished")
        self.set_status(status, result=_jsonable(res))

    @staticmethod
    def mesh_geometry_event(mesh) -> dict:
        ptr, loops = mesh.cell_polygons_2d()
        lo, hi = mesh.bounding_box()
        return {"type": "geometry", "nodes": _b64(mesh.nodes[:, :2].ravel()),
                "ptr": base64.b64encode(ptr.astype(np.int32).tobytes()).decode(),
                "loops": base64.b64encode(loops.astype(np.int32).tobytes()).decode(),
                "bbox": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])], "cells": mesh.n_cells}

    # ---------------- exports ----------------
    def export_package(self, outdir: str) -> dict:
        """Write the Fluent hand-off package for this job; returns {kind: path}."""
        from ..fluent.convert import structured_boundary_specs, structured_mesh, structured_solution_points
        from ..fluent.export import write_interpolation_file, write_journal, write_pyfluent_script
        from ..mesh.fluent_msh import write_fluent_mesh

        os.makedirs(outdir, exist_ok=True)
        s = self.solver
        if s is None:
            raise RuntimeError("no solution yet")
        base = "flowtrace"
        files = {}
        if hasattr(s, "nx"):
            mesh = structured_mesh(s.case, s.solid_h)
            sp = structured_solution_points(s)
            pts, fields = sp["points"], sp["fields"]
            bnd = structured_boundary_specs(s.case)
            energy = s.energy
            nu = s.case.nu
        else:
            mesh = s.mesh
            pts = s.cell_centres()
            f = s.fields()
            fields = {k: v for k, v in f.items() if k in ("x-velocity", "y-velocity", "z-velocity", "pressure", "temperature",
                                                          "turb-kinetic-energy", "specific-diss-rate", "turb-diss-rate")}
            bnd = {k: _jsonable(v.__dict__) for k, v in s.zone_specs.items()}
            energy = s.energy
            nu = s.case.viscosity / s.case.density
        files["msh"] = os.path.join(outdir, base + ".msh")
        write_fluent_mesh(mesh, files["msh"])
        files["ip"] = write_interpolation_file(os.path.join(outdir, base + ".ip"), pts, fields)
        files["jou"] = write_journal(os.path.join(outdir, "run_fluent.jou"), base + ".msh", base + ".ip",
                                     energy=energy)
        files["py"] = write_pyfluent_script(os.path.join(outdir, "run_pyfluent.py"), base + ".msh", base + ".ip",
                                            dim=mesh.dim, energy=energy, boundaries=bnd, viscosity=nu)
        csv = os.path.join(outdir, base + "_cells.csv")
        cols = ["x", "y", "z"][:pts.shape[1]] + list(fields)
        np.savetxt(csv, np.column_stack([pts] + [fields[k] for k in fields]), delimiter=",",
                   header=",".join(cols), comments="")
        files["csv"] = csv
        readme = os.path.join(outdir, "README_FLUENT.txt")
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(
                "FlowTrace -> Ansys Fluent hand-off\n"
                "==================================\n"
                f"{base}.msh          mesh with named zones (File > Read > Mesh)\n"
                f"{base}.ip           solution for File > Interpolate > Read and Interpolate (warm start)\n"
                "run_fluent.jou        batch:  fluent 2ddp -g -t4 -i run_fluent.jou\n"
                "run_pyfluent.py       PyFluent:  python run_pyfluent.py\n"
                f"{base}_cells.csv     cell-centre values for post-processing\n\n"
                "Boundary conditions used by FlowTrace (set the same in Fluent):\n"
                + "\n".join(f"  {k}: {v}" for k, v in bnd.items()) + "\n")
        files["readme"] = readme
        return files


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(o))
    return o
