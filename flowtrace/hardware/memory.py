"""Memory budgeting: estimate before allocating, refuse before crashing.

M_required = M_mesh + M_fields + M_solver + M_preconditioner + M_temporary,
compared with what the target device actually has free, with a safety margin
for allocator fragmentation and library workspaces (cuFFT/cuBLAS plans).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

SAFETY = 1.25  # allocator fragmentation, FFT/BLAS workspaces, CUDA context


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
    fits = req <= available if available > 0 else True
    advice = []
    if not fits:
        advice.append(f"Needs ~{_fmt(req)} but only {_fmt(available)} free on {device}.")
        if itemsize == 8:
            advice.append("Switch precision to fp32 (halves field memory).")
        if device == "gpu":
            advice.append("Run on the CPU backend (system RAM) or reduce the grid.")
        else:
            advice.append("Reduce the grid resolution.")
    return MemoryEstimate(device=device, breakdown=bd, required=req, available=int(available), fits=fits,
                          utilisation=(req / available) if available else 0.0, advice=advice)


def estimate_unstructured(n_cells: int, n_faces: int, dim: int, itemsize: int, energy: bool,
                          device: str, available: int, amg: bool = True) -> MemoryEstimate:
    nnz = n_cells + 2 * n_faces
    bd = {
        "mesh_geometry": n_faces * (2 * 8 + (3 * dim + 4) * itemsize) + n_cells * (dim + 1) * itemsize,
        "fields": n_cells * (dim + 2 + (1 if energy else 0)) * itemsize * 3,  # value, previous, gradients
        "gradients": n_cells * dim * (dim + 2) * itemsize,
        "face_data": n_faces * 8 * itemsize,
        "sparse_matrices": 2 * nnz * (itemsize + 4) + 2 * (n_cells + 1) * 4,
        "preconditioner": (int(1.6 * nnz * (itemsize + 4))) if amg else n_cells * itemsize,
        "krylov_vectors": 8 * n_cells * itemsize,
    }
    raw = sum(bd.values())
    req = int(raw * SAFETY)
    fits = req <= available if available > 0 else True
    advice = [] if fits else [f"Needs ~{_fmt(req)} but only {_fmt(available)} free on {device}."]
    return MemoryEstimate(device=device, breakdown=bd, required=req, available=int(available), fits=fits,
                          utilisation=(req / available) if available else 0.0, advice=advice)


fmt_bytes = _fmt
