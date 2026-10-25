"""Local web server for the FlowTrace dashboard (standard library only).

Endpoints (JSON unless noted):
  GET  /                      dashboard (static files)
  GET  /api/system            hardware profile, backends, cached benchmarks, PyFluent status
  POST /api/benchmark         run micro-benchmarks on every backend
  GET  /api/presets           built-in cases and their parameters
  POST /api/estimate          memory budget for a case before running it
  POST /api/run               start a job      {kind, preset|case|mesh_id, params, overrides, ...}
  POST /api/stop              stop the running job
  POST /api/view              change the displayed field {field}
  GET  /api/state             current job, plan, residual history, log
  GET  /api/events            Server-Sent Events stream (status, plan, residuals, frame, geometry, log)
  POST /api/mesh              upload a Fluent mesh (.msh/.cas/.cas.h5/.msh.h5; raw body, header X-Filename)
  GET  /api/export/<kind>     download msh | ip | jou | py | csv | zip for the current job
  GET  /api/projects          saved projects in <workdir>/projects
  POST /api/projects          save {name, spec, ui} (+ the uploaded mesh and the last result)
  GET  /api/projects/<name>   open a project (re-registers its mesh)
  POST /api/report            self-contained HTML report of the current run {image: PNG data URL}
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ..runtime.engine import EventHub, Job, _jsonable

STATIC = os.path.join(os.path.dirname(__file__), "static")


class AppState:
    def __init__(self, workdir: str):
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.hub = EventHub()
        self.job = None
        self.meshes = {}
        self.mesh_info = {}
        self.lock = threading.Lock()
        self._profile = None
        self._bench = None
        self.mesh_files = {}  # mesh id -> (file name, original upload bytes), saved with projects

    @property
    def project_dir(self) -> str:
        d = os.path.join(self.workdir, "projects")
        os.makedirs(d, exist_ok=True)
        return d

    def profile(self):
        from ..hardware.probe import probe_system

        if self._profile is None:
            self._profile = probe_system(self.workdir)
        return self._profile


def preset_catalog():
    import inspect

    from ..structured.case import PRESETS

    out = []
    for name, fn in PRESETS.items():
        sig = inspect.signature(fn)
        params = {k: v.default for k, v in sig.parameters.items() if k != "kw" and v.default is not inspect._empty}
        case = fn()
        out.append({"id": name, "title": case.name.split(" Re=")[0].split(" Ra=")[0], "params": params,
                    "description": case.description, "steady": case.steady_tol is not None,
                    "energy": case.energy, "obstacles": bool(case.obstacles),
                    "defaults": _jsonable(case.to_dict())})
    return out


def make_handler(app: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FlowTrace"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # ---------------- helpers ----------------
        def _json(self, obj, status=200):
            body = json.dumps(_jsonable(obj)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, msg, status=400):
            self._json({"error": msg}, status)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _json_body(self):
            raw = self._body()
            return json.loads(raw.decode() or "{}")

        def _file(self, path, download_name=None, ctype=None):
            with open(path, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype or mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            if download_name:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.end_headers()
            self.wfile.write(data)

        # ---------------- routing ----------------
        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    return self._file(os.path.join(STATIC, "index.html"), ctype="text/html; charset=utf-8")
                if path.startswith("/static/"):
                    fp = os.path.normpath(os.path.join(STATIC, path[len("/static/"):]))
                    if not fp.startswith(STATIC) or not os.path.isfile(fp):
                        return self._error("not found", 404)
                    return self._file(fp)
                if path == "/api/system":
                    return self._system()
                if path == "/api/presets":
                    return self._json({"presets": preset_catalog()})
                if path == "/api/state":
                    return self._state()
                if path == "/api/events":
                    return self._events()
                if path.startswith("/api/export/"):
                    return self._export(path.rsplit("/", 1)[-1])
                if path == "/api/projects":
                    return self._json({"projects": list_projects(app)})
                if path.startswith("/api/projects/"):
                    return self._json(open_project(app, path.rsplit("/", 1)[-1]))
                if path.startswith("/api/mesh/"):
                    mid = path.rsplit("/", 1)[-1]
                    mesh = app.meshes.get(mid)
                    if mesh is None:
                        return self._error("unknown mesh", 404)
                    if mesh.dim != 2:
                        return self._json({"dim": 3})
                    return self._json(Job.mesh_geometry_event(mesh))
                return self._error("not found", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._error(f"{type(exc).__name__}: {exc}", 500)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/run":
                    return self._run()
                if path == "/api/stop":
                    if app.job:
                        app.job.stop()
                    return self._json({"ok": True})
                if path == "/api/view":
                    body = self._json_body()
                    job = app.job
                    if job:
                        job.view_field = body.get("field")
                        if job.solver is not None and not job.running:
                            job._publish_frame(job.solver, force=True)
                    return self._json({"ok": True})
                if path == "/api/benchmark":
                    from ..hardware.bench import run_benchmarks

                    if app.job and app.job.running:
                        return self._error("a simulation is running; benchmark later", 409)
                    app._bench = run_benchmarks(use_cache=False)
                    return self._json(app._bench)
                if path == "/api/estimate":
                    return self._estimate()
                if path == "/api/mesh":
                    return self._upload_mesh()
                if path == "/api/projects":
                    return self._json(save_project(app, self._json_body()))
                if path == "/api/report":
                    return self._report()
                return self._error("not found", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._error(f"{type(exc).__name__}: {exc}", 500)

        # ---------------- handlers ----------------
        def _system(self):
            from ..fluent.bridge import pyfluent_status
            from ..hardware.backend import available_backends
            from ..hardware.bench import load_cached

            prof = app.profile()
            self._json({"profile": prof.to_dict(), "backends": available_backends(),
                        "bench": app._bench or load_cached(), "pyfluent": pyfluent_status().__dict__,
                        "workdir": app.workdir})

        def _state(self):
            job = app.job
            snap = app.hub.snapshot()
            self._json({
                "job": None if job is None else {
                    "id": job.id, "kind": job.kind, "status": job.status, "error": job.error,
                    "plan": job.plan, "result": job.result, "running": job.running,
                    "field": job.view_field, "spec": job.spec,
                },
                "residuals": snap["residuals"][-2000:], "log": snap["log"][-300:],
                "meshes": app.mesh_info,
            })

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = app.hub.subscribe()
            try:
                # replay the latest state so a (re)connecting browser is in sync
                for ev in app.hub.latest.values():
                    self._sse(ev)
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self._sse(ev)
                    except Exception:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                app.hub.unsubscribe(q)

        def _sse(self, ev):
            data = json.dumps(_jsonable(ev))
            self.wfile.write(f"event: {ev.get('type', 'message')}\ndata: {data}\n\n".encode())
            self.wfile.flush()

        def _run(self):
            spec = self._json_body()
            with app.lock:
                if app.job and app.job.running:
                    return self._error("a simulation is already running", 409)
                app.hub.residuals.clear()
                app.hub.latest.clear()
                job = Job(spec, app.hub, app.workdir, meshes=app.meshes)
                app.job = job
                job.start()
            self._json({"ok": True, "job": job.id})

        def _estimate(self):
            from ..hardware.backend import available_backends, get_backend
            from ..hardware.memory import estimate_structured, estimate_unstructured
            from ..structured.case import StructuredCase, make_preset

            spec = self._json_body()
            out = {}
            if spec.get("kind", "structured") == "structured":
                case = make_preset(spec["preset"], **spec.get("params", {})) if spec.get("preset") \
                    else StructuredCase.from_dict(spec["case"])
                for k, v in (spec.get("overrides") or {}).items():
                    setattr(case, k, v)
                for name in available_backends():
                    be = get_backend(name)
                    free, _ = be.mem_info()
                    item = 4 if (case.precision == "fp32" or (be.is_gpu and case.precision == "auto")) else 8
                    out[name] = estimate_structured(case.nx, case.ny, case.energy, item,
                                                    "mg" if case.obstacles else "fft", True,
                                                    bool(case.obstacles), be.device, free).to_dict()
                out["cells"] = case.nx * case.ny
            else:
                mesh = app.meshes.get(spec.get("mesh_id"))
                if mesh is None:
                    return self._error("unknown mesh", 404)
                for name in available_backends():
                    be = get_backend(name)
                    free, _ = be.mem_info()
                    out[name] = estimate_unstructured(mesh.n_cells, mesh.n_faces, mesh.dim, 8,
                                                      bool(spec.get("energy")), be.device, free).to_dict()
                out["cells"] = mesh.n_cells
            self._json(out)

        def _upload_mesh(self):
            name = os.path.basename(self.headers.get("X-Filename", "mesh.msh"))
            self._json(register_mesh(app, self._body(), name))

        def _report(self):
            body = self._json_body()
            job = app.job
            if job is None or job.solver is None:
                return self._error("run a case first", 409)
            data = run_report_html(app, job, body.get("image")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="flowtrace_report_{job.id}.html"')
            self.end_headers()
            self.wfile.write(data)

        def _export(self, kind):
            job = app.job
            if job is None or job.solver is None:
                return self._error("run a case first", 409)
            outdir = os.path.join(app.workdir, "export", job.id)
            files = job.export_package(outdir)
            if kind == "zip":
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in files.values():
                        zf.write(p, os.path.basename(p))
                data = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="flowtrace_{job.id}_fluent.zip"')
                self.end_headers()
                self.wfile.write(data)
                return
            if kind not in files:
                return self._error(f"unknown export {kind!r}", 404)
            return self._file(files[kind], download_name=os.path.basename(files[kind]))

    return Handler


# --------------------------------------------------------------------------
# meshes, projects and reports (module level so the tests can use them directly)
# --------------------------------------------------------------------------

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def register_mesh(app: AppState, raw: bytes, name: str) -> dict:
    """Read an uploaded mesh (.msh/.cas, gzipped or not, or CFF .cas.h5/.msh.h5) and summarise it."""
    from ..mesh.fluent_msh import read_fluent_mesh
    from ..mesh.quality import mesh_quality
    from ..unstructured.case import default_boundaries

    t0 = time.perf_counter()
    if raw[:8] == HDF5_MAGIC:
        from ..mesh.fluent_h5 import read_case_h5

        fd, tmp = tempfile.mkstemp(suffix=".h5")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            mesh = read_case_h5(tmp, name=name)
        finally:
            os.unlink(tmp)
    else:
        mesh = read_fluent_mesh(raw, name=name)
    q = mesh_quality(mesh)
    mid = uuid.uuid4().hex[:8]
    app.meshes[mid] = mesh
    app.mesh_files[mid] = (name, raw)
    info = mesh.summary()
    info.update({"id": mid, "read_seconds": round(time.perf_counter() - t0, 3),
                 "volume": q.total_volume, "repaired_faces": q.left_handed_faces_repaired,
                 "boundaries": {k: v.__dict__ for k, v in default_boundaries(mesh).items()},
                 "solid_zones": [z.name for z in mesh.cell_zones() if z.type == "solid"],
                 "quality": {"min_orthogonal_quality": q.min_orthogonal_quality, "max_skewness": q.max_skewness,
                             "max_aspect_ratio": q.max_aspect_ratio,
                             "max_non_orthogonality_deg": q.max_non_orthogonality_deg,
                             "warnings": q.warnings, "ok": q.ok}})
    app.mesh_info[mid] = info
    return info


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\- ]", "_", str(name)).strip().strip(".")
    if not s:
        raise ValueError("project name is empty")
    return s[:80]


def list_projects(app: AppState) -> list:
    out = []
    for fn in sorted(os.listdir(app.project_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(app.project_dir, fn), encoding="utf-8") as fh:
                d = json.load(fh)
            out.append({"name": d.get("name", fn[:-5]), "saved": d.get("saved"), "kind": d.get("spec", {}).get("kind"),
                        "title": d.get("title", ""), "has_result": bool(d.get("result"))})
        except (OSError, ValueError):
            continue
    return out


def save_project(app: AppState, body: dict) -> dict:
    name = _safe_name(body.get("name", ""))
    spec = body.get("spec") or {}
    proj = {"name": name, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "flowtrace": _version(),
            "title": body.get("title", ""), "spec": spec, "ui": body.get("ui") or {}}
    mid = spec.get("mesh_id")
    if spec.get("kind") == "unstructured" and mid in app.mesh_files:
        fname, raw = app.mesh_files[mid]
        ext = ".cas.h5" if raw[:8] == HDF5_MAGIC else (os.path.splitext(fname)[1] or ".msh")
        mesh_file = name + "_mesh" + ext
        with open(os.path.join(app.project_dir, mesh_file), "wb") as fh:
            fh.write(raw)
        proj["mesh_file"] = mesh_file
        proj["mesh_name"] = fname
    job = app.job

    def same(a, b):  # the displayed field is a view setting, not part of the case
        return {k: v for k, v in (a or {}).items() if k != "field"} == {k: v for k, v in (b or {}).items() if k != "field"}

    if job is not None and not job.running and job.result and same(job.spec, spec):
        proj["result"] = _jsonable(job.result)
    path = os.path.join(app.project_dir, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(proj), fh, indent=1)
    return {"ok": True, "name": name, "path": path}


def open_project(app: AppState, name: str) -> dict:
    from urllib.parse import unquote

    name = _safe_name(unquote(name))
    path = os.path.join(app.project_dir, name + ".json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no project named {name!r}")
    with open(path, encoding="utf-8") as fh:
        proj = json.load(fh)
    out = {"project": proj}
    if proj.get("mesh_file"):
        with open(os.path.join(app.project_dir, proj["mesh_file"]), "rb") as fh:
            info = register_mesh(app, fh.read(), proj.get("mesh_name", proj["mesh_file"]))
        proj["spec"]["mesh_id"] = info["id"]
        out["mesh"] = info
    return out


def _version() -> str:
    from .. import __version__

    return __version__


def _svg_residuals(history: list, width: int = 640, height: int = 260) -> str:
    """Residual history as an inline SVG (log scale); no plotting library needed."""
    import math

    if not history:
        return "<p>No residual history.</p>"
    keys = list(history[-1].get("scaled", {}).keys())
    pts = {k: [(i, ev.get("scaled", {}).get(k)) for i, ev in enumerate(history)] for k in keys}
    vals = [math.log10(v) for k in keys for _, v in pts[k] if v and v > 0]
    if not vals:
        return "<p>No residual history.</p>"
    lo, hi = math.floor(min(vals)), math.ceil(max(vals))
    hi = max(hi, lo + 1)
    n = max(len(history) - 1, 1)
    L, R, T, B = 48, 10, 10, 26
    X = lambda i: L + (width - L - R) * i / n  # noqa: E731
    Y = lambda lv: T + (height - T - B) * (hi - lv) / (hi - lo)  # noqa: E731
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Residual history" '
           f'style="max-width:100%;background:#fff;border:1px solid #ddd;border-radius:6px">']
    for e in range(lo, hi + 1):
        y = Y(e)
        out.append(f'<line x1="{L}" x2="{width - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eee"/>'
                   f'<text x="4" y="{y + 4:.1f}" font-size="10" fill="#555">1e{e}</text>')
    for c, k in enumerate(keys):
        d = " ".join(f"{X(i):.1f},{Y(math.log10(v)):.1f}" for i, v in pts[k] if v and v > 0)
        col = colours[c % len(colours)]
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.4" points="{d}"/>')
        out.append(f'<text x="{L + 8 + 110 * c}" y="{height - 6}" font-size="11" fill="{col}">{k}</text>')
    out.append("</svg>")
    return "".join(out)


def run_report_html(app: AppState, job, image=None) -> str:
    """Self-contained HTML report of the current run (opens without FlowTrace)."""
    import html as H

    esc = lambda x: H.escape(str(x))  # noqa: E731
    res = _jsonable(job.result or {})
    plan = job.plan or {}
    prof = app.profile()
    parts = []

    def table(d):
        return "<table>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in d.items()) + "</table>"

    parts.append(f"<h1>FlowTrace run report</h1><p class='m'>Job {esc(job.id)} &middot; status <b>{esc(job.status)}</b>"
                 f" &middot; FlowTrace {esc(_version())} &middot; {esc(time.strftime('%Y-%m-%d %H:%M'))}</p>")
    parts.append("<h2>Machine</h2>" + table({"CPU": prof.cpu_model,
                                             "GPU": prof.best_gpu.name if prof.best_gpu else "none", "OS": prof.os}))
    spec = dict(job.spec)
    spec.pop("field", None)
    parts.append("<h2>Case</h2><pre>" + esc(json.dumps(_jsonable(spec), indent=1)) + "</pre>")
    if plan:
        pl = plan.get("placements", [])
        parts.append("<h2>Execution plan</h2><p>Backend <b>" + esc(plan.get("backend")) + "</b>, precision "
                     + esc(plan.get("precision")) + ". Measured ms/step: " + esc(plan.get("measured_ms_per_step") or "-")
                     + "</p><table><tr><th>Stage</th><th>Device</th><th>Method</th><th>Reason</th></tr>"
                     + "".join(f"<tr><td>{esc(p.get('stage'))}</td><td>{esc(p.get('device'))}</td>"
                               f"<td>{esc(p.get('method'))}</td><td>{esc(p.get('reason'))}</td></tr>" for p in pl)
                     + "</table>")
    parts.append("<h2>Convergence</h2>" + _svg_residuals(list(app.hub.residuals)))
    summary = {k: res[k] for k in ("steps", "iterations", "time", "converged") if k in res}
    if summary:
        parts.append(table(summary))
    checks = (res.get("validation") or {}).get("checks") or []
    if checks:
        parts.append("<h2>Validation</h2><table><tr><th>Check</th><th>Value</th><th>Reference</th><th>Result</th></tr>"
                     + "".join(f"<tr><td>{esc(c.get('name'))}</td><td>{esc(c.get('value'))} {esc(c.get('unit', ''))}</td>"
                               f"<td>{esc(c.get('reference', '-'))}</td><td>{'pass' if c.get('pass') else 'check'}</td></tr>"
                               for c in checks) + "</table>")
    if res.get("mass_balance"):
        mb = res["mass_balance"]
        parts.append("<h2>Mass balance</h2>" + table({**mb, "net": sum(mb.values())}))
    if isinstance(image, str) and image.startswith("data:image/png;base64,"):
        parts.append(f"<h2>Field</h2><img alt='Field snapshot' src='{image}' style='max-width:100%'>")
    log = [ev.get("msg", "") for ev in list(app.hub.log)[-60:]]
    parts.append("<h2>Log (last lines)</h2><pre>" + esc("\n".join(log)) + "</pre>")
    parts.append("<p class='m'>Ansys and Fluent are trademarks of ANSYS, Inc. FlowTrace is an independent project, "
                 "not affiliated with or endorsed by ANSYS, Inc.</p>")
    css = ("body{font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;"
           "color:#1d2433;background:#fff}h1{font-size:24px}h2{font-size:17px;margin-top:28px;border-top:1px solid #e3e6ec;"
           "padding-top:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #eceff4;"
           "padding:4px 6px;text-align:left;vertical-align:top}pre{background:#f6f8fb;padding:8px;overflow:auto;"
           "font-size:12px}.m{color:#5b6475}")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><title>FlowTrace Run Report</title>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{css}</style></head><body>"
            + "".join(parts) + "</body></html>")


def serve(host: str = "127.0.0.1", port: int = 8765, workdir: str = "flowtrace_runs", open_browser: bool = True):
    app = AppState(workdir)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"FlowTrace dashboard: {url}  (Ctrl+C to quit)")
    if open_browser:
        try:
            import webbrowser

            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.job:
            app.job.stop()
        httpd.server_close()
    return app


def start_background(host="127.0.0.1", port=0, workdir="flowtrace_runs"):
    """Start the server in a daemon thread (tests / notebooks). Returns (httpd, app, url)."""
    app = AppState(workdir)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, app, f"http://{host}:{httpd.server_address[1]}/"
