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
  POST /api/mesh              upload a Fluent mesh (raw body, header X-Filename)
  GET  /api/export/<kind>     download msh | ip | jou | py | csv | zip for the current job
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import threading
import time
import uuid
import zipfile
from http import HTTPStatus
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


def unstructured_preset_catalog():
    import inspect

    from ..unstructured.presets import DESCRIPTIONS, PRESETS

    out = []
    for name, fn in PRESETS.items():
        sig = inspect.signature(fn)
        params = {k: v.default for k, v in sig.parameters.items() if k != "kw" and v.default is not inspect._empty}
        _mesh, case, _info = fn()
        title, desc = DESCRIPTIONS.get(name, (case.name, ""))
        out.append({"id": name, "title": title, "params": params, "description": desc,
                    "defaults": _jsonable(case.to_dict())})
    return out


def make_handler(app: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FlowTrace/0.1"

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
                    if not hasattr(app, "_catalog"):
                        app._catalog = {"presets": preset_catalog(),
                                        "unstructured_presets": unstructured_preset_catalog()}
                    return self._json(app._catalog)
                if path == "/api/state":
                    return self._state()
                if path == "/api/events":
                    return self._events()
                if path.startswith("/api/export/"):
                    return self._export(path.rsplit("/", 1)[-1])
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
            from ..mesh.fluent_msh import read_fluent_mesh
            from ..mesh.unstructured import compute_geometry
            from ..unstructured.case import default_boundaries

            name = self.headers.get("X-Filename", "mesh.msh")
            raw = self._body()
            t0 = time.perf_counter()
            mesh = read_fluent_mesh(raw, name=os.path.basename(name))
            geo = compute_geometry(mesh)
            mid = uuid.uuid4().hex[:8]
            app.meshes[mid] = mesh
            info = mesh.summary()
            info.update({"id": mid, "read_seconds": round(time.perf_counter() - t0, 3),
                         "volume": geo.total_volume, "repaired_faces": geo.flipped_faces,
                         "boundaries": {k: v.__dict__ for k, v in default_boundaries(mesh).items()}})
            app.mesh_info[mid] = info
            self._json(info)

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
