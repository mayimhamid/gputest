"""Command-line interface.

  flowtrace ui                         start the dashboard (default)
  flowtrace probe                      show what FlowTrace sees on this machine
  flowtrace bench                      micro-benchmark every backend
  flowtrace run cavity --re 1000 --n 256 [--backend cupy] [--export out/]
  flowtrace mesh path/to/mesh.msh      inspect a Fluent mesh (.msh/.cas/.cas.h5) + quality report
  flowtrace quality path/to/mesh.msh   orthogonal quality / skewness / aspect ratio (exit 2 if critical)
  flowtrace solve path/to/mesh.msh --bc inlet=inlet:1.0 --bc outlet=outlet --mu 0.01 --export out/
  flowtrace fluent-warmstart case.cas.h5 --turbulence k-omega-sst --out ws/   pre-solve for Fluent
  flowtrace batch a.msh b.msh --set viscosity=0.1,0.05 --out runs/   many cases at once on CPU cores + GPU
  flowtrace validate                   run the validation suite (cavity, convection, Taylor-Green ...)
  flowtrace benchmark [--tier full]    reproducible benchmarks + HTML/PDF report (every published number)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


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
    from .mesh.fluent_h5 import read_any_mesh
    from .mesh.quality import mesh_quality

    t0 = time.perf_counter()
    m, fmt = read_any_mesh(args.path)
    s = m.summary()
    print(f"{s['name']} ({fmt}): {s['dim']}D, {s['cells']:,} cells, {s['faces']:,} faces, {s['nodes']:,} nodes "
          f"(read in {time.perf_counter() - t0:.2f} s)")
    for z in s["zones"]:
        print(f"  zone {z['id']:>4}  {z['kind']:<4} {z['type']:<18} {z['name']:<24} {z['count']:,}")
    for p in m.periodic:
        print(f"  periodic pair: zone {p['zone_a']} <-> {p['zone_b']}, translation {np.round(p['translation'], 6)}")
    print()
    print(mesh_quality(m).summary())


def cmd_quality(args):
    from .mesh.fluent_h5 import read_any_mesh
    from .mesh.quality import mesh_quality

    m, _ = read_any_mesh(args.path)
    q = mesh_quality(m)
    if args.json:
        print(json.dumps(q.to_dict(), indent=1))
    else:
        print(q.summary())
    sys.exit(0 if q.ok else 2)


def cmd_h5_inspect(args):
    from .mesh.fluent_h5 import describe_h5

    print(describe_h5(args.path, args.max))


def _apply_mesh_options(mesh, args):
    from .mesh.unstructured import make_periodic

    for spec in args.periodic or []:
        a, _, b = spec.partition(":")
        make_periodic(mesh, a, b)
    return mesh


def _case_from_args(args, name):
    from .unstructured.case import UnstructuredCase

    solids = {}
    for s in args.solid or []:
        parts = s.split(":")
        solids[parts[0]] = {"conductivity": float(parts[1]), "source": float(parts[2]) if len(parts) > 2 else 0.0}
    mrf = {}
    for s in args.mrf or []:
        parts = s.split(":")
        rot = {"omega": float(parts[1])}
        if len(parts) > 2:
            rot["origin"] = [float(x) for x in parts[2].split(",")]
        mrf[parts[0]] = rot
    return UnstructuredCase(name=name, boundaries=_parse_bc(args.bc), density=args.rho, viscosity=args.mu,
                            energy=args.energy or bool(solids), conductivity=args.k, cp=args.cp,
                            max_iter=args.max_iter, tol=args.tol, backend=args.backend, algorithm=args.algorithm,
                            turbulence=args.turbulence, solids=solids, mrf=mrf,
                            body_force=[float(x) for x in args.body_force.split(",")] + [0.0] * 2
                            if args.body_force else [0.0, 0.0, 0.0])


def cmd_fluent_warmstart(args):
    from .fluent.warmstart import run_warmstart
    from .mesh.fluent_h5 import read_any_mesh  # noqa: F401  (fails early without h5py for .h5 input)

    case = _case_from_args(args, os.path.basename(args.path))
    rep = run_warmstart(args.path, case, args.out, iterations_fluent=args.fluent_iterations,
                        setup_physics=args.setup_physics, allow_bad_mesh=args.allow_bad_mesh)
    print(f"\nFlowTrace: {rep['iterations']} iterations, {'converged' if rep['converged'] else 'NOT converged'}, "
          f"{rep['solve_seconds']:.1f} s")
    for k, v in rep["files"].items():
        print(f"  {k:<7} {v}")


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


def cmd_batch(args):
    import itertools
    from dataclasses import asdict

    from .runtime.batch import BatchJob, run_batch

    sweeps = []
    for item in args.set or []:
        key, _, vals = item.partition("=")
        sweeps.append([(key.strip(), _num(v)) for v in vals.split(",") if v.strip()])
    jobs = []
    for path in args.paths:
        stem = os.path.basename(path).split(".")[0]
        for combo in itertools.product(*sweeps) if sweeps else [()]:
            name = stem + "".join(f"_{k}={v:g}" if isinstance(v, float) else f"_{k}={v}" for k, v in combo)
            case = asdict(_case_from_args(args, name))
            case.pop("backend", None)
            for k, v in combo:
                if k not in case:
                    raise SystemExit(f"--set {k}: not a case parameter (e.g. viscosity, density, max_iter, tol)")
                case[k] = v
            jobs.append(BatchJob(name=name, mesh=os.path.abspath(path), case=case,
                                 out_dir=os.path.abspath(args.out) if args.out else ""))
    res = run_batch(jobs, cpu_workers=args.cpu_workers, use_gpu=False if args.no_gpu else None)
    print()
    print(f"{'case':<40} {'device':<6} {'cells':>8} {'its':>6} {'conv':>5} {'s':>7}  forces on walls (total)")
    for r in res["results"]:
        if r.get("error"):
            print(f"{r['name']:<40} FAILED  {r['error']}")
            continue
        forces = "; ".join(f"{z} " + ",".join(f"{c:.5g}" for c in f["total"]) for z, f in r["wall_forces"].items())
        print(f"{r['name']:<40} {r['device']:<6} {r['cells']:>8,} {r['iterations']:>6} {str(r['converged']):>5} "
              f"{r['seconds']:>7.1f}  {forces}")
    print(f"wall time {res['wall_seconds']:.1f} s")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "batch.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
    if any(r.get("error") for r in res["results"]):
        sys.exit(1)


def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v.strip()


def cmd_solve(args):
    from dataclasses import asdict

    from .mesh.fluent_h5 import read_any_mesh
    from .runtime.engine import EventHub, Job

    mesh, _ = read_any_mesh(args.path)
    _apply_mesh_options(mesh, args)
    case = asdict(_case_from_args(args, os.path.basename(args.path)))
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


def cmd_benchmark(args):
    from .validation.benchmarks import REGISTRY, run_benchmarks

    if args.list:
        for b in REGISTRY.values():
            print(f"{b.id:<36} {','.join(b.tiers):<11} {b.category:<17} {b.title}")
        return
    ids = [s for s in (args.only or "").split(",") if s] or None
    data = run_benchmarks(ids=ids, tier=args.tier, backend=args.backend, out_dir=args.out)
    s = data["summary"]
    print(f"\n{s['passed']}/{s['total']} benchmarks passed" + (f"; report in {args.out}" if args.out else ""))
    sys.exit(0 if s["passed"] == s["total"] else 1)


def cmd_warmstart_study(args):
    from .fluent.warmstart_study import CASES, run_study

    cases = [s for s in (args.cases or "").split(",") if s] or None
    if cases:
        bad = set(cases) - set(CASES)
        if bad:
            sys.exit(f"unknown case(s) {sorted(bad)}; choose from {sorted(CASES)}")
    data = run_study(args.out, cases=cases, backend=args.backend, fluent=not args.no_fluent, tol=args.tol)
    print(f"\nresults: {os.path.join(args.out, 'results.md')}")
    for r in data["fluent"]:
        if r["status"] != "measured":
            print(f"Fluent {r['case']}: not run - {r['message']}")


def cmd_parse_transcript(args):
    from .fluent.warmstart_study import parse_transcript

    res = {}
    for p in args.paths:
        with open(p, encoding="utf-8", errors="ignore") as fh:
            res[p] = parse_transcript(fh.read())
        r = res[p]
        print(f"{p}: {r['iterations']} iterations, converged={r['converged']}")
    if len(args.paths) == 2 and all(res[p]["iterations"] for p in args.paths):
        a, b = (res[p]["iterations"] for p in args.paths)
        print(f"ratio {args.paths[0]} / {args.paths[1]} = {a / b:.2f}")


def cmd_gpu_report(args):
    from .hardware.gpu_report import run_gpu_report

    sizes = tuple(int(s) for s in args.sizes.split(","))
    data = run_gpu_report(args.out, sizes=sizes)
    if data.get("gpu_status"):
        print(data["gpu_status"])
    print(f"GPU log written to {args.out}")


def cmd_ui(args):
    from .ui.server import serve

    serve(args.host, args.port, args.workdir, open_browser=not args.no_browser)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default="flowtrace_runs", help="runs, checkpoints and exports go here")
    from . import __version__

    ap = argparse.ArgumentParser(prog="flowtrace", description="FlowTrace heterogeneous CFD solver",
                                 parents=[common])
    ap.add_argument("--version", action="version", version=f"FlowTrace {__version__}")
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
    p = sub.add_parser("mesh", help="inspect a Fluent mesh (.msh/.cas/.cas.h5/.msh.h5) and its quality")
    p.add_argument("path")
    p.set_defaults(func=cmd_mesh)
    p = sub.add_parser("quality", help="mesh quality report (orthogonal quality, skewness, aspect ratio)")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_quality)
    p = sub.add_parser("h5-inspect", help="print the group/dataset tree of an HDF5 (.cas.h5/.dat.h5) file")
    p.add_argument("path")
    p.add_argument("--max", type=int, default=400)
    p.set_defaults(func=cmd_h5_inspect)

    def physics_args(p):
        p.add_argument("path")
        p.add_argument("--bc", action="append", help="zone=kind[:values[:T|q]]  e.g. inlet=inlet:1.0:T300  lid=wall:1,0")
        p.add_argument("--rho", type=float, default=1.0)
        p.add_argument("--mu", type=float, default=0.01)
        p.add_argument("--k", type=float, default=0.01, help="fluid thermal conductivity")
        p.add_argument("--cp", type=float, default=1.0)
        p.add_argument("--energy", action="store_true")
        p.add_argument("--turbulence", default="laminar",
                       choices=["laminar", "k-omega-sst", "k-epsilon-realizable", "k-epsilon-standard"])
        p.add_argument("--solid", action="append", help="zone:conductivity[:volumetric source]  (conjugate heat transfer)")
        p.add_argument("--mrf", action="append", help="cell zone:omega[:x0,y0[,z0]]  (moving reference frame)")
        p.add_argument("--periodic", action="append", help="zoneA:zoneB  translational periodic pair")
        p.add_argument("--body-force", dest="body_force", help="fx,fy[,fz] per unit volume (e.g. -dp/dx)")
        p.add_argument("--algorithm", default="SIMPLEC")
        p.add_argument("--max-iter", dest="max_iter", type=int, default=3000)
        p.add_argument("--tol", type=float, default=1e-5)
        p.add_argument("--backend", default="auto")

    p = sub.add_parser("solve", help="solve on a Fluent mesh with SIMPLE/SIMPLEC")
    physics_args(p)
    p.add_argument("--export")
    p.set_defaults(func=cmd_solve)
    p = sub.add_parser("batch", help="solve many meshes / parameter sets at once on every CPU core and the GPU")
    p.add_argument("paths", nargs="+", help="Fluent meshes/cases")
    physics_args(p)
    p.add_argument("--set", action="append", help="sweep a case parameter: key=v1,v2,..  (e.g. viscosity=0.1,0.05)")
    p.add_argument("--out", default="", help="write <case>.ip, <case>.npz and batch.json here")
    p.add_argument("--cpu-workers", dest="cpu_workers", type=int, default=None,
                   help="concurrent CPU cases (default: physical cores, minus one when a GPU is used)")
    p.add_argument("--no-gpu", dest="no_gpu", action="store_true")
    p.set_defaults(func=cmd_batch)
    p = sub.add_parser("fluent-warmstart", help="pre-solve a Fluent case; write .ip (+.dat.h5), journal, PyFluent script")
    physics_args(p)
    p.add_argument("--out", default="flowtrace_warmstart")
    p.add_argument("--fluent-iterations", dest="fluent_iterations", type=int, default=500)
    p.add_argument("--setup-physics", dest="setup_physics", action="store_true",
                   help="also set the viscous/energy models in the generated Fluent scripts")
    p.add_argument("--allow-bad-mesh", dest="allow_bad_mesh", action="store_true")
    p.set_defaults(func=cmd_fluent_warmstart)
    p = sub.add_parser("validate", help="run the validation suite")
    p.add_argument("--full", action="store_true")
    p.add_argument("--backend", default="auto")
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("benchmark", help="reproducible benchmarks -> results.json + HTML/PDF/Markdown report")
    p.add_argument("--tier", default="quick", choices=["quick", "full"])
    p.add_argument("--only", help="comma-separated benchmark ids (see --list)")
    p.add_argument("--list", action="store_true", help="list the benchmarks and exit")
    p.add_argument("--backend", default="auto")
    p.add_argument("--out", default="benchmarks/results", help="output directory ('' for none)")
    p.set_defaults(func=cmd_benchmark)
    p = sub.add_parser("warmstart-study", help="cold vs warm start: FlowTrace proxy (+ Ansys Fluent via PyFluent)")
    p.add_argument("--cases", help="comma-separated: cavity,channel,step,cylinder,heated")
    p.add_argument("--out", default="benchmarks/fluent_warmstart")
    p.add_argument("--backend", default="auto")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--no-fluent", action="store_true", help="skip the Fluent runs even if PyFluent is installed")
    p.set_defaults(func=cmd_warmstart_study)
    p = sub.add_parser("gpu-report", help="GPU vs CPU correctness and speed log for this machine")
    p.add_argument("--out", default="benchmarks/results")
    p.add_argument("--sizes", default="64,128,256,512,1024")
    p.set_defaults(func=cmd_gpu_report)
    p = sub.add_parser("parse-transcript", help="count iterations in Fluent transcripts (e.g. cold.trn warm.trn)")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_parse_transcript)
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        args = ap.parse_args(["ui"] + (argv or []))
    args.func(args)


if __name__ == "__main__":
    main()
