"""Memory budgeting: estimate before allocating, refuse before crashing.

M_required = M_mesh + M_fields + M_solver + M_preconditioner + M_temporary,
compared with what the target device actually has free, with a safety margin
for allocator fragmentation and library workspaces (cuFFT/cuBLAS plans).

The estimates are upper bounds on the CPU (benchmarks/memory_estimate.py), so a
case that does not fit is refused before anything large is allocated: the planner
skips a device it does not fit on, and raises MemoryError when it fits on none.
``FLOWTRACE_MEMORY_CHECK=off`` lets such a run start anyway (at the user's risk).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List

SAFETY = 1.25  # allocator fragmentation, FFT/BLAS workspaces, CUDA context
OVERRIDE = "FLOWTRACE_MEMORY_CHECK"  # =off: never refuse a case on its memory estimate


def _check_disabled() -> bool:
    return os.environ.get(OVERRIDE, "").strip().lower() in ("off", "0", "no", "false")


# Largest measured peak / estimate (unstructured SIMPLEC, CPU, 4k cells to the 3.2M-cell rocket,
# benchmarks/memory_estimate.py). Between this fraction of the upper bound and the bound itself a
# CPU run is started with a warning: no measured case would have exceeded the free RAM, and the
# operating system pages rather than failing. GPU memory cannot page, so the GPU keeps the bound.
MEASURED_PEAK_FRACTION = 0.85


def _verdict(req: int, available: int, device: str, tight_ok: bool = False):
    """(fits, advice) for an estimate of ``req`` bytes against ``available`` free bytes."""
    if available <= 0 or req <= available:
        return True, []
    need = f"Needs up to ~{_fmt(req)} but only {_fmt(available)} free on {device}."
    if tight_ok and device == "cpu" and MEASURED_PEAK_FRACTION * req <= available:
        return True, [need, f"Measured peaks were at most {MEASURED_PEAK_FRACTION:.2f} of this bound "
                            f"(~{_fmt(MEASURED_PEAK_FRACTION * req)}), so the run starts; close other "
                            "programs if the system starts paging."]
    if _check_disabled():
        return True, [need, f"Memory check disabled ({OVERRIDE}=off): running anyway."]
    return False, [need, f"To try anyway, set {OVERRIDE}=off."]


@dataclass
class MemoryEstimate:
    device: str
    breakdown: Dict[str, int]
    required: int
    available: int
    fits: bool
    utilisation: float
    advice: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def estimate_structured(nx: int, ny: int, energy: bool, itemsize: int, pressure: str,
                        implicit: bool, obstacles: bool, device: str, available: int) -> MemoryEstimate:
    ng = (nx + 5) * (ny + 5)  # ghosted field size
    n = nx * ny
    nf = 4 if energy else 3  # U, V, P, T
    nh = 3 if energy else 2  # Adams-Bashforth history
    bd = {
        "fields": nf * ng * itemsize,
        "history": nh * n * itemsize,
        "temporaries": 18 * n * itemsize,  # flux, RHS and projection work arrays (peak)
        "masks": (6 * n * itemsize) if obstacles else 0,
    }
    if pressure == "mg":
        bd["pressure_solver"] = int(1.34 * n * (9 * itemsize + 2))  # cx, cy, shift, diag, inv, colours...
        bd["krylov_vectors"] = 5 * n * itemsize
    else:
        dense = (nx * nx + ny * ny) * 2 * itemsize if pressure == "eig" else 0
        bd["pressure_solver"] = dense + 3 * n * itemsize * (2 if pressure == "fft" else 1)
    if implicit:
        bd["diffusion_solvers"] = (3 if energy else 2) * ((nx * nx + ny * ny) * 2 * itemsize + 2 * n * itemsize)
    raw = sum(bd.values())
    req = int(raw * SAFETY)
    fits, advice = _verdict(req, available, device)
    if not fits:
        hints = ["Run on the CPU backend (system RAM) or reduce the grid." if device == "gpu"
                 else "Reduce the grid resolution."]
        if itemsize == 8:
            hints.append("precision='fp32' halves field memory, but check that FP32 can meet the steady "
                         "tolerance (docs/user-guide/gpu.md, Precision).")
        advice[1:1] = hints
    return MemoryEstimate(device=device, breakdown=bd, required=req, available=int(available), fits=fits,
                          utilisation=(req / available) if available else 0.0, advice=advice)


def estimate_unstructured(n_cells: int, n_faces: int, dim: int, itemsize: int, energy: bool,
                          device: str, available: int, amg: bool = True) -> MemoryEstimate:
    nnz = n_cells + 2 * n_faces
    bd = {
        "mesh_geometry": n_faces * (2 * 8 + (3 * dim - 3) * itemsize) + n_cells * (dim + 1) * itemsize,
        "fields": n_cells * (dim + 2 + (1 if energy else 0)) * itemsize * 3,  # value, previous, gradients
        "gradients": n_cells * dim * (dim + 2) * itemsize,
        "face_data": n_faces * 8 * itemsize,
        "sparse_matrices": 2 * nnz * (itemsize + 4) + 2 * (n_cells + 1) * 4,
        "preconditioner": (int(1.6 * nnz * (itemsize + 4))) if amg else n_cells * itemsize,
        "krylov_vectors": 8 * n_cells * itemsize,
    }
    # The terms above leave out geometry and FV-mesh intermediates, the incidence and pattern
    # matrices (and their native-kernel copies), the per-iteration momentum, relaxed and
    # pressure matrices, the AMG set-up and the gradient lists. With this term, measured peaks
    # (CPU: 4k-65k cells 2-D and 3-D, laminar, energy and SST, as traced allocations; and the
    # 3.2M-cell, 10.9M-face polyhedral rocket as resident-set growth) are 0.66-0.81x of the
    # estimate (benchmarks/memory_estimate.py; rocket 0.76). The GPU path stores the same
    # arrays; its CuPy pool overhead has not been measured.
    modelled = sum(bd.values())
    bd["measured_overhead"] = int(1.0 * modelled + 8e6)
    raw = sum(bd.values())
    req = int(raw * SAFETY)
    fits, advice = _verdict(req, available, device, tight_ok=True)
    if not fits:  # an upper bound: say so, and what would help
        advice[1:1] = [f"This is an upper bound (measured CPU peaks were at most {MEASURED_PEAK_FRACTION:.2f} of it).",
                       "Run on the CPU (system RAM) or use a coarser mesh." if device == "gpu"
                       else "Use a coarser mesh, close other programs, or use a machine with more memory."]
    return MemoryEstimate(device=device, breakdown=bd, required=req, available=int(available), fits=fits,
                          utilisation=(req / available) if available else 0.0, advice=advice)


fmt_bytes = _fmt
