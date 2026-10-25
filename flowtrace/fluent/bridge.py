"""Live link to a licensed Ansys Fluent session through PyFluent.

FlowTrace never needs Fluent to run, but when ``ansys-fluent-core`` and a
Fluent installation are present this bridge can

1. launch (or connect to) Fluent,
2. read the mesh/case,
3. interpolate the FlowTrace solution as the initial field (warm start),
4. iterate and report Fluent's own residual count,
5. write the finished case/data.

All Settings-API calls are guarded: API names move between PyFluent
releases, so each step falls back to the equivalent TUI command.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BridgeStatus:
    available: bool
    version: Optional[str] = None
    message: str = ""
    log: List[str] = field(default_factory=list)


def pyfluent_status() -> BridgeStatus:
    try:
        pf = importlib.import_module("ansys.fluent.core")
    except Exception:
        return BridgeStatus(False, None, "ansys-fluent-core is not installed (pip install ansys-fluent-core)")
    return BridgeStatus(True, getattr(pf, "__version__", "unknown"),
                        "PyFluent installed; launching requires a licensed Ansys Fluent installation")


class FluentBridge:
    def __init__(self, dimension: int = 2, processors: int = 4, precision: str = "double",
                 ui_mode: str = "no_gui", connect_ip: Optional[str] = None, connect_port: Optional[int] = None,
                 connect_password: Optional[str] = None):
        st = pyfluent_status()
        if not st.available:
            raise RuntimeError(st.message)
        self.pf = importlib.import_module("ansys.fluent.core")
        self.log: List[str] = []
        if connect_ip:
            self.session = self.pf.connect_to_fluent(ip=connect_ip, port=connect_port, password=connect_password)
            self._log(f"connected to Fluent at {connect_ip}:{connect_port}")
        else:
            self.session = self.pf.launch_fluent(mode="solver", dimension=dimension, precision=precision,
                                                 processor_count=processors, ui_mode=ui_mode)
            self._log(f"launched Fluent {dimension}D {precision} on {processors} processes")

    def _log(self, msg: str):
        self.log.append(msg)

    def _tui(self, command: str):
        """Run a TUI command string (works across releases)."""
        self.session.scheme_eval.scheme_eval(f'(ti-menu-load-string "{command}")')
        self._log(f"TUI: {command}")

    @property
    def settings(self):
        return getattr(self.session, "settings", self.session)

    def read(self, path: str):
        try:
            self.settings.file.read_case(file_name=path)
        except Exception:
            self._tui(f'/file/read-case \\"{path}\\"')
        self._log(f"read {path}")

    def set_laminar(self, energy: bool = False):
        try:
            self.settings.setup.models.viscous.model = "laminar"
            if energy:
                self.settings.setup.models.energy.enabled = True
        except Exception:
            self._tui("/define/models/viscous/laminar yes")
            if energy:
                self._tui("/define/models/energy yes")

    def interpolate(self, ip_file: str):
        try:
            self.settings.solution.initialization.hybrid_initialize()
        except Exception:
            self._tui("/solve/initialize/initialize-flow")
        try:
            self.settings.file.interpolate.read_data(file_name=ip_file)
        except Exception:
            self._tui(f'/file/interpolate/read-data \\"{ip_file}\\"')
        self._log(f"interpolated FlowTrace data from {ip_file}")

    def iterate(self, n: int):
        try:
            self.settings.solution.run_calculation.iterate(iter_count=int(n))
        except Exception:
            self._tui(f"/solve/iterate {int(n)}")
        self._log(f"iterated {n}")

    def write(self, path: str):
        try:
            self.settings.file.write(file_type="case-data", file_name=path)
        except Exception:
            self._tui(f'/file/write-case-data \\"{path}\\"')
        self._log(f"wrote {path}")

    def close(self):
        try:
            self.session.exit()
        except Exception:
            pass

    def warm_start(self, mesh_or_case: str, ip_file: str, iterations: int = 200, energy: bool = False,
                   out_file: str = "flowtrace_result.cas.h5") -> List[str]:
        self.read(mesh_or_case)
        self.set_laminar(energy)
        self.interpolate(ip_file)
        self.iterate(iterations)
        self.write(out_file)
        return self.log
