"""FlowTrace: a heterogeneous, hardware-aware CFD solver.

FlowTrace profiles the machine it runs on, places every stage of a CFD solve
(stencils, pressure solve, reductions, I/O) on the device and algorithm that is
measured to be fastest, and exchanges meshes and solutions with Ansys Fluent.
"""

import os as _os

# Numba's OpenMP workers busy-wait after every parallel kernel by default. The
# solvers interleave JIT kernels with NumPy/SciPy calls, and spinning workers
# starve those calls of cores (measured 10x slow-downs), so ask the OpenMP
# runtimes (GNU and Intel) to sleep instead. Must happen before numba loads.
_os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
_os.environ.setdefault("KMP_BLOCKTIME", "0")

__version__ = "0.3.0"

__all__ = ["__version__"]
