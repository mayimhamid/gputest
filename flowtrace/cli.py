"""Command-line interface.

  flowtrace ui                         start the dashboard (default)
  flowtrace probe                      show what FlowTrace sees on this machine
  flowtrace bench                      micro-benchmark every backend
  flowtrace run cavity --re 1000 --n 256 [--backend cupy] [--export out/]
  flowtrace mesh path/to/mesh.msh      inspect a Fluent mesh
  flowtrace solve path/to/mesh.msh --bc inlet=inlet:1.0 --bc outlet=outlet --nu 0.01 --export out/
  flowtrace validate                   run the validation suite (cavity, convection, Taylor-Green ...)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _print_plan(plan):
    print(f"\nExecution plan -> backend {plan.backend} ({plan.device}), {plan.precision}")
    if plan.measured_ms_per_step:
        print("  measured ms/step: " + ", ".join(f"{k} {v}" for k, v in plan.measured_ms_per_step.items()))
    for p in plan.placements:
        print(f"  - {p.stage:<38} {p.device:<18} {p.method}")
        print(f"      {p.reason}")
    for n in plan.notes:
        print(f"  * {n}")


def cmd_probe(args):
    from .hardware.backend import available_backends
    from .hardware.memory import fmt_bytes
    from .hardware.probe import probe_system

    p = probe_system()
    print(f"OS        {p.os}  (Python {p.python})")
    print(f"CPU       {p.cpu_model}  {p.physical_cores}C/{p.logical_cores}T  SIMD: {' '.join(p.simd) or '-'}")
    print(f"RAM       {fmt_bytes(p.ram_available)} free / {fmt_bytes(p.ram_total)}")
    for g in p.gpus:
        print(f"GPU {g.index}     {g.name}  {fmt_bytes(g.vram_total)}  cc {g.compute_capability}  "
              f"tensor={g.tensor_cores} rt={g.rt_cores} fp64=1/{round(1 / g.fp64_rate)}  usable={g.usable}")
    if not p.gpus:
        print("GPU       none detected")
    print(f"Backends  {', '.join(available_backends())}")
    print("Libraries " + ", ".join(f"{k}={v}" for k, v in p.libraries.items()))
    for n in p.notes:
        print(f"note: {n}")


def cmd_bench(args):
    from .hardware.bench import run_benchmarks

    res = run_benchmarks(use_cache=False)
    print(json.dumps(res["results"], indent=2))


def cmd_run(args):
    from .hardware.planner import plan_structured
    from .runtime.engine import EventHub, Job

    params = {}
    for k in ("re", "ra", "pr", "n", "nx", "ny", "n_per_d", "n_per_h"):
        v = getattr(args, k, None)
        if v is not None:
            params[k] = v
    overrides = {"backend": args.backend, "precision": args.precision}
    if args.t_end is not None:
        overrides["t_end"] = args.t_end
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    spec = {"kind": "structured", "preset": args.preset, "params": params, "overrides": overrides,
            "fmg": not args.no_fmg, "mode": args.mode, "checkpoint_seconds": args.checkpoint}
    hub = EventHub()
    q = hub.subscribe()
    job = Job(spec, hub, args.workdir)
    job.start()
    last = 0.0
    while job.running or not q.empty():
        try:
            ev = q.get(timeout=0.5)
        except Exception:
            continue
        t = ev.get("type")
        if t == "log":
            print(ev["msg"])
        elif t == "residuals" and time.time() - last > 1.0:
            last = time.time()
            res = "  ".join(f"{k}={v:.2e}" for k, v in ev["scaled"].items())
            print(f"[{ev['label']}] step {ev['step']:>7}  t={ev['t']:.4g}  {res}  {ev['mcells_per_s']:.2f} Mcell/s")
        elif t == "status" and ev["status"] in ("converged", "finished", "stopped", "error"):
            print(f"status: {ev['status']}")
    job.join()
    if job.error:
        sys.exit(1)
    if args.export:
        files = job.export_package(args.export)
        print("Fluent package:")
        for k, v in files.items():
            print(f"  {k:<6} {v}")


def cmd_mesh(args):
    from .mesh.fluent_msh import read_fluent_mesh
    from .mesh.unstructured import compute_geometry

    t0 = time.perf_counter()
    m = read_fluent_mesh(args.path)
    g = compute_geometry(m)
    s = m.summary()
    print(f"{s['name']}: {s['dim']}D, {s['cells']:,} cells, {s['faces']:,} faces, {s['nodes']:,} nodes "
          f"(read in {time.perf_counter() - t0:.2f} s)")
    print(f"volume {g.total_volume:.6g}, left-handed faces repaired: {g.flipped_faces}")
    for z in s["zones"]:
        print(f"  zone {z['id']:>4}  {z['kind']:<4} {z['type']:<18} {z['name']:<24} {z['count']:,}")


def _parse_bc(items):
    out = {}
    for it in items or []:
        name, _, rest = it.partition("=")
        parts = rest.split(":")
        kind = parts[0]
        spec = {"kind": kind}
        if len(parts) > 1 and parts[1]:
            nums = [float(x) for x in parts[1].split(",")]
            if kind == "inlet":
                if len(nums) == 1:
                    spec["normal_speed"] = nums[0]
                else:
                    spec["velocity"] = nums
            elif kind == "wall":
                spec["velocity"] = nums
            elif kind in ("outlet", "pressure"):
                spec["pressure"] = nums[0]
        if len(parts) > 2 and parts[2]:
            t = parts[2]
            if t.startswith("q"):
                spec["thermal"], spec["value"] = "flux", float(t[1:])
            else:
                spec["thermal"], spec["value"] = "temperature", float(t.lstrip("T"))
        out[name] = spec
    return out


def cmd_solve(args):
    from .mesh.fluent_msh import read_fluent_mesh
    from .runtime.engine import EventHub, Job

    mesh = read_fluent_mesh(args.path)
    case = {"name": os.path.basename(args.path), "boundaries": _parse_bc(args.bc), "density": args.rho,
            "viscosity": args.mu, "energy": args.energy, "conductivity": args.k, "cp": args.cp,
            "max_iter": args.max_iter, "tol": args.tol, "backend": args.backend, "algorithm": args.algorithm}
    hub = EventHub()
    q = hub.subscribe()
    job = Job({"kind": "unstructured", "mesh_id": "m", "case": case}, hub, args.workdir, meshes={"m": mesh})
    job.start()
    last = 0.0
    while job.running or not q.empty():
        try:
            ev = q.get(timeout=0.5)
        except Exception:
            continue
        if ev.get("type") == "log":
            print(ev["msg"])
        elif ev.get("type") == "residuals" and time.time() - last > 1.0:
            last = time.time()
            print(f"it {ev['step']:>6}  " + "  ".join(f"{k}={v:.2e}" for k, v in ev["scaled"].items()))
        elif ev.get("type") == "status" and ev["status"] in ("converged", "finished", "stopped", "error"):
            print(f"status: {ev['status']}")
    job.join()
    if job.error:
        sys.exit(1)
    if args.export:
        for k, v in job.export_package(args.export).items():
            print(f"  {k:<6} {v}")


def cmd_validate(args):
    from .validation.suite import run_suite

    ok = run_suite(quick=not args.full, backend=args.backend)
    sys.exit(0 if ok else 1)


def cmd_ui(args):
    from .ui.server import serve

    serve(args.host, args.port, args.workdir, open_browser=not args.no_browser)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default="flowtrace_runs", help="runs, checkpoints and exports go here")
    ap = argparse.ArgumentParser(prog="flowtrace", description="FlowTrace heterogeneous CFD solver",
                                 parents=[common])
    sub = ap.add_subparsers(dest="cmd")
    _add = sub.add_parser

    def add_parser(name, **kw):
        return _add(name, parents=[common], **kw)

    sub.add_parser = add_parser
    p = sub.add_parser("ui", help="start the web dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_ui)
    sub.add_parser("probe", help="describe this machine").set_defaults(func=cmd_probe)
    sub.add_parser("bench", help="micro-benchmark the backends").set_defaults(func=cmd_bench)
    p = sub.add_parser("run", help="run a built-in structured case")
    p.add_argument("preset", choices=["cavity", "natural_convection", "channel", "cylinder", "step",
                                      "rayleigh_benard", "taylor_green"])
    for k in ("re", "ra", "pr"):
        p.add_argument(f"--{k}", type=float)
    for k in ("n", "nx", "ny", "n_per_d", "n_per_h", "max_steps"):
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int)
    p.add_argument("--t-end", dest="t_end", type=float)
    p.add_argument("--backend", default="auto")
    p.add_argument("--precision", default="auto")
    p.add_argument("--mode", default="transient", choices=["transient", "steady-simplec"])
    p.add_argument("--no-fmg", action="store_true")
    p.add_argument("--checkpoint", type=float, default=0.0, help="seconds between async checkpoints")
    p.add_argument("--export", help="write the Fluent hand-off package to this directory")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("mesh", help="inspect a Fluent mesh")
    p.add_argument("path")
    p.set_defaults(func=cmd_mesh)
    p = sub.add_parser("solve", help="solve on a Fluent mesh with SIMPLE/SIMPLEC")
    p.add_argument("path")
    p.add_argument("--bc", action="append", help="zone=kind[:values[:T|q]]  e.g. inlet=inlet:1.0:T300  lid=wall:1,0")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--cp", type=float, default=1.0)
    p.add_argument("--energy", action="store_true")
    p.add_argument("--algorithm", default="SIMPLEC")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=3000)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--backend", default="auto")
    p.add_argument("--export")
    p.set_defaults(func=cmd_solve)
    p = sub.add_parser("validate", help="run the validation suite")
    p.add_argument("--full", action="store_true")
    p.add_argument("--backend", default="auto")
    p.set_defaults(func=cmd_validate)
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        args = ap.parse_args(["ui"] + (argv or []))
    args.func(args)


if __name__ == "__main__":
    main()
