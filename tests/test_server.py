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
        presets_all = json.loads(_get(url + "api/presets")[1])
        presets = presets_all["presets"]
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
    finally:
        httpd.shutdown()
