"""Discover what the target machine can do.

Everything here is best-effort and dependency-light: each probe falls back to
the next source of truth (psutil -> OS interfaces -> conservative defaults) and
never raises. The result is a plain dataclass that the planner, the memory
budget and the UI all consume.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total: int  # bytes
    vram_free: int  # bytes
    compute_capability: str = ""
    multiprocessors: int = 0
    mem_bandwidth_gbs: float = 0.0  # theoretical peak, 0 when unknown
    driver: str = ""
    source: str = ""  # which probe produced this record
    usable: bool = False  # a CUDA array backend (CuPy) can drive it
    tensor_cores: bool = False
    rt_cores: bool = False
    fp64_rate: float = 1.0 / 32.0  # FP64 throughput relative to FP32

    @property
    def cc_major(self) -> int:
        try:
            return int(self.compute_capability.split(".")[0])
        except ValueError:
            return 0

    @property
    def is_datacenter_fp64(self) -> bool:
        return self.fp64_rate >= 0.25


@dataclass
class SystemProfile:
    os: str
    python: str
    machine: str
    cpu_model: str
    logical_cores: int
    physical_cores: int
    simd: List[str]
    ram_total: int
    ram_available: int
    disk_free: int
    disk_path: str
    on_battery: Optional[bool]
    gpus: List[GPUInfo] = field(default_factory=list)
    libraries: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        usable = [g for g in self.gpus if g.usable]
        if not usable:
            return None
        return max(usable, key=lambda g: (g.vram_total, g.multiprocessors))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["best_gpu"] = asdict(self.best_gpu) if self.best_gpu else None
        return d


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------

def _cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg  # type: ignore

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def _physical_cores(logical: int) -> int:
    try:
        import psutil  # type: ignore

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    if os.path.exists("/proc/cpuinfo"):
        try:
            pairs = set()
            phys = core = None
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                    elif not line.strip():
                        if core is not None:
                            pairs.add((phys, core))
                        phys = core = None
            if core is not None:
                pairs.add((phys, core))
            if pairs:
                return len(pairs)
        except Exception:
            pass
    return logical


def _simd_features() -> List[str]:
    wanted = ["SSE42", "AVX", "AVX2", "FMA3", "AVX512F", "AVX512_SKX", "NEON", "ASIMD", "SVE"]
    feats = None
    for mod in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            feats = importlib.import_module(mod).__cpu_features__
            break
        except Exception:
            continue
    if not feats:
        return []
    return [f for f in wanted if feats.get(f)]


# --------------------------------------------------------------------------
# Memory / disk / power
# --------------------------------------------------------------------------

def _ram() -> tuple:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        if os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    k, v = line.split(":", 1)
                    info[k] = int(v.strip().split()[0]) * 1024
            return info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            total = int(out.stdout.strip())
            return total, total // 2
    except Exception:
        pass
    return 0, 0


def _on_battery() -> Optional[bool]:
    try:
        import psutil  # type: ignore

        batt = psutil.sensors_battery()
        if batt is None:
            return None
        return not batt.power_plugged
    except Exception:
        return None


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

_DATACENTER_FP64 = re.compile(
    r"(A100|A30|H100|H200|GH200|B100|B200|GB200|V100|P100|TITAN V|MI\d{2,3})", re.IGNORECASE
)


def _classify_gpu(g: GPUInfo) -> GPUInfo:
    name = g.name.upper()
    major = g.cc_major
    gtx16 = "GTX 16" in name
    g.tensor_cores = major >= 7 and not gtx16
    g.rt_cores = ("RTX" in name or "QUADRO RTX" in name) or (
        major >= 8 and any(k in name for k in ("A2000", "A4000", "A5000", "A6000", "L40", "L4"))
    )
    if _DATACENTER_FP64.search(name):
        g.fp64_rate = 0.5
    elif major >= 8:
        g.fp64_rate = 1.0 / 64.0
    else:
        g.fp64_rate = 1.0 / 32.0
    return g


def _gpus_from_cupy() -> List[GPUInfo]:
    import cupy  # type: ignore

    rt = cupy.cuda.runtime
    out = []
    for i in range(rt.getDeviceCount()):
        props = rt.getDeviceProperties(i)
        name = props.get("name", b"GPU")
        name = name.decode() if isinstance(name, bytes) else str(name)
        with cupy.cuda.Device(i):
            free, total = cupy.cuda.Device(i).mem_info
        clock_khz = props.get("memoryClockRate", 0) or 0
        bus_bits = props.get("memoryBusWidth", 0) or 0
        bw = 2.0 * clock_khz * 1e3 * bus_bits / 8.0 / 1e9 if clock_khz and bus_bits else 0.0
        g = GPUInfo(
            index=i,
            name=name,
            vram_total=int(total),
            vram_free=int(free),
            compute_capability=f"{props.get('major', 0)}.{props.get('minor', 0)}",
            multiprocessors=int(props.get("multiProcessorCount", 0)),
            mem_bandwidth_gbs=round(bw, 1),
            driver=str(rt.driverGetVersion()),
            source="cupy",
            usable=True,
        )
        out.append(_classify_gpu(g))
    return out


def _gpus_from_nvidia_smi() -> List[GPUInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = "index,name,memory.total,memory.free,compute_cap,driver_version"
    res = subprocess.run(
        [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if res.returncode != 0:
        # Older drivers do not know compute_cap.
        fields = "index,name,memory.total,memory.free,driver_version"
        res = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    out = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            idx, name, tot, free, cc, drv = parts
        elif len(parts) == 5:
            idx, name, tot, free, drv = parts
            cc = ""
        else:
            continue
        g = GPUInfo(
            index=int(idx),
            name=name,
            vram_total=int(float(tot)) * 1024 * 1024,
            vram_free=int(float(free)) * 1024 * 1024,
            compute_capability=cc,
            driver=drv,
            source="nvidia-smi",
        )
        out.append(_classify_gpu(g))
    return out


def _cupy_works() -> bool:
    try:
        import cupy  # type: ignore

        a = cupy.arange(8, dtype=cupy.float32)
        return float((a * 2).sum()) == 56.0
    except Exception:
        return False


def _probe_gpus(notes: List[str]) -> List[GPUInfo]:
    if _cupy_works():
        try:
            return _gpus_from_cupy()
        except Exception as exc:  # pragma: no cover - depends on driver
            notes.append(f"CuPy present but device query failed: {exc}")
    try:
        gpus = _gpus_from_nvidia_smi()
    except Exception:
        gpus = []
    if gpus:
        notes.append(
            "NVIDIA GPU detected but no working CuPy; install the matching wheel "
            "(pip install cupy-cuda12x) to enable the GPU backend."
        )
    return gpus


# --------------------------------------------------------------------------
# Libraries
# --------------------------------------------------------------------------

def _library_versions() -> dict:
    libs = {}
    for mod, label in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("numba", "numba"),
        ("cupy", "cupy"),
        ("pyamg", "pyamg"),
        ("psutil", "psutil"),
        ("ansys.fluent.core", "pyfluent"),
    ):
        try:
            m = importlib.import_module(mod)
            libs[label] = getattr(m, "__version__", "installed")
        except Exception:
            libs[label] = None
    return libs


def probe_system(workdir: Optional[str] = None) -> SystemProfile:
    """Return a :class:`SystemProfile` for the current machine. Never raises."""
    notes: List[str] = []
    logical = os.cpu_count() or 1
    total, avail = _ram()
    workdir = workdir or os.getcwd()
    try:
        disk_free = shutil.disk_usage(workdir).free
    except Exception:
        disk_free = 0
    gpus = _probe_gpus(notes)
    on_batt = _on_battery()
    if on_batt:
        notes.append("Running on battery: laptop CPUs/GPUs usually throttle; plug in for benchmarks.")
    prof = SystemProfile(
        os=f"{platform.system()} {platform.release()}",
        python=sys.version.split()[0],
        machine=platform.machine(),
        cpu_model=_cpu_model(),
        logical_cores=logical,
        physical_cores=_physical_cores(logical),
        simd=_simd_features(),
        ram_total=total,
        ram_available=avail,
        disk_free=disk_free,
        disk_path=os.path.abspath(workdir),
        on_battery=on_batt,
        gpus=gpus,
        libraries=_library_versions(),
        notes=notes,
    )
    best = prof.best_gpu
    if best is not None and not best.is_datacenter_fp64:
        notes.append(
            f"{best.name}: consumer-class FP64 rate (~1/{round(1 / best.fp64_rate)} of FP32); "
            "FlowTrace defaults to FP32 fields on this GPU."
        )
    if prof.libraries.get("numba") is None:
        notes.append("Numba not installed: CPU stencils run as vectorised NumPy (pip install numba).")
    return prof
