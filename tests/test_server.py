import json
import time
import urllib.request

from flowtrace.ui.server import start_background


def _get(url):
    req = urllib.request.Request(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=30) as r:
        return r.status, r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=60) as r:
        return json.loads(r.read())


def test_server_end_to_end(tmp_path):
    httpd, app, url = start_background(workdir=str(tmp_path))
    try:
        st, body = _get(url)
        assert st == 200 and b"FlowTrace" in body
        sysinfo = json.loads(_get(url + "api/system")[1])
        assert "numpy" in sysinfo["backends"]
        presets = json.loads(_get(url + "api/presets")[1])["presets"]
        assert {p["id"] for p in presets} >= {"cavity", "natural_convection", "cylinder"}
        est = _post(url + "api/estimate", {"kind": "structured", "preset": "cavity", "params": {"n": 64}})
        assert est["cells"] == 4096
        r = _post(url + "api/run", {"kind": "structured", "preset": "cavity", "params": {"re": 100, "n": 32},
                                    "overrides": {"max_steps": 300}, "fmg": False})
        assert r["ok"]
        for _ in range(240):
            state = json.loads(_get(url + "api/state")[1])
            if state["job"]["status"] in ("converged", "finished", "error", "stopped"):
                break
            time.sleep(0.25)
        assert state["job"]["status"] in ("converged", "finished"), state["job"]
        assert state["residuals"]
        st, data = _get(url + "api/export/zip")
        assert st == 200 and data[:2] == b"PK"
        st, data = _get(url + "api/export/ip")
        assert data.splitlines()[0] == b"3"
        # run report: self-contained HTML with the residual chart
        req = urllib.request.Request(url + "api/report", data=json.dumps({"image": None}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=60) as r:
            html = r.read().decode()
        assert "<svg" in html and "Execution plan" in html and "trademarks of ANSYS" in html
        # projects: save (with the finished result) and list
        spec = state["job"]["spec"]
        saved = _post(url + "api/projects", {"name": "cavity test", "spec": spec, "ui": {"source": "structured"}})
        assert saved["ok"]
        projects = json.loads(_get(url + "api/projects")[1])["projects"]
        assert projects[0]["name"] == "cavity test" and projects[0]["has_result"]
        opened = json.loads(_get(url + "api/projects/cavity%20test")[1])
        assert opened["project"]["spec"]["preset"] == "cavity"
        # static assets for the i18n / theme front end
        st, js = _get(url + "static/i18n.js")
        assert st == 200 and "فارسی".encode() in js
        # built-in unstructured case: validation and wall report in the result
        presets_all = json.loads(_get(url + "api/presets")[1])
        assert {p["id"] for p in presets_all["unstructured_presets"]} >= {"turbulent_channel", "flat_plate"}
        r = _post(url + "api/run", {"kind": "unstructured", "preset": "radiation_slab", "params": {"nx": 20}})
        assert r["ok"]
        for _ in range(240):
            state = json.loads(_get(url + "api/state")[1])
            if state["job"]["status"] in ("converged", "finished", "error", "stopped"):
                break
            time.sleep(0.25)
        assert state["job"]["status"] == "converged", state["job"]
        checks = state["job"]["result"]["validation"]["checks"]
        assert all(c["pass"] for c in checks), checks
        assert "hot" in state["job"]["result"]["wall_report"]
        # the report of this second job must not contain the first job's log or residuals
        req = urllib.request.Request(url + "api/report", data=json.dumps({"image": None}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=60) as r:
            html2 = r.read().decode()
        assert "Radiation slab" in html2 or "P-1" in html2
        assert "Lid-driven cavity" not in html2.split("Log of job")[1]
        assert "local continuity" in html2
        # solver diagnostics reach the result, the residual events, the log and the report
        diag = state["job"]["result"]["diagnostics"]
        assert diag["available"] and diag["health"]["finite"] and "energy" in diag
        mine = [ev for ev in state["residuals"] if ev.get("job") == state["job"]["id"]]
        assert any("diagnostics" in ev for ev in mine) and "diagnostics" in mine[-1]
        assert mine[-1]["diagnostics"]["iteration"] == diag["iteration"]
        assert any("[diagnostics] iteration" in ev.get("msg", "") for ev in state["log"])
        assert "Solver diagnostics (last iteration)" in html2 and "componentwise backward errors" in html2
    finally:
        httpd.shutdown()


def test_mesh_upload_quality_and_project_with_mesh(tmp_path):
    import pytest

    from flowtrace.mesh.unstructured import cartesian_mesh_2d, set_cell_zone
    from flowtrace.ui.server import AppState, open_project, register_mesh, save_project

    h5 = pytest.importorskip("h5py")  # noqa: F841
    from flowtrace.mesh.fluent_h5 import write_case_h5

    app = AppState(str(tmp_path))
    m = set_cell_zone(cartesian_mesh_2d(8, 8), lambda x: x[:, 0] < 0.25, "block", "solid")
    p = tmp_path / "m.cas.h5"
    write_case_h5(m, str(p))
    info = register_mesh(app, p.read_bytes(), "m.cas.h5")
    assert info["cells"] == 64 and info["solid_zones"] == ["block"] and info["quality"]["ok"]
    spec = {"kind": "unstructured", "mesh_id": info["id"], "case": {"name": "m"}}
    save_project(app, {"name": "with mesh", "spec": spec})
    assert (tmp_path / "projects" / "with mesh_mesh.cas.h5").exists()
    out = open_project(app, "with mesh")
    assert out["mesh"]["cells"] == 64 and out["project"]["spec"]["mesh_id"] == out["mesh"]["id"]


def test_request_allowed_rules():
    from flowtrace.ui.server import request_allowed

    assert request_allowed("127.0.0.1", "127.0.0.1:8765", None)
    assert request_allowed("127.0.0.1", "localhost:8765", "http://localhost:8765")
    assert request_allowed("::1", "[::1]:8765", "http://[::1]:8765")
    assert not request_allowed("127.0.0.1", "evil.example:8765", None)  # DNS rebinding
    assert not request_allowed("127.0.0.1", "127.0.0.1:8765", "http://evil.example")  # cross-site request
    assert not request_allowed("127.0.0.1", "127.0.0.1:8765", "null")  # sandboxed / file:// page
    assert request_allowed("0.0.0.0", "192.168.1.5:8765", "http://192.168.1.5:8765")  # --allow-remote
    assert not request_allowed("0.0.0.0", "192.168.1.5:8765", "http://evil.example")


def test_server_refuses_dns_rebinding_and_cross_site_requests(tmp_path):
    import urllib.error

    httpd, app, url = start_background(workdir=str(tmp_path))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def status(path, headers, data=None):
        req = urllib.request.Request(url + path, data=data, headers=headers)
        try:
            with opener.open(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert status("api/system", {}) == 200
        assert status("static/i18n.js", {}) == 200
        assert status("static/../server.py", {}) == 404  # static files stay inside static/
        assert status("api/system", {"Host": "evil.example"}) == 403
        body = json.dumps({"kind": "structured", "preset": "cavity"}).encode()
        assert status("api/run", {"Origin": "http://evil.example", "Content-Type": "text/plain"}, body) == 403
        assert app.job is None  # the forged request started nothing
        port = httpd.server_address[1]
        assert status("api/stop", {"Origin": f"http://127.0.0.1:{port}"}, b"{}") == 200  # the page's own requests
    finally:
        httpd.shutdown()


def test_ui_refuses_a_network_host_without_allow_remote():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "flowtrace", "ui", "--host", "0.0.0.0", "--no-browser"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "--allow-remote" in r.stderr and "Traceback" not in r.stderr
