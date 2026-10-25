"""Grid Convergence Index (Roache 1994; procedure of Celik et al., J. Fluids Eng. 130, 2008).

Given a quantity on three systematically refined grids (fine ``phi1``, medium
``phi2``, coarse ``phi3``) with refinement ratios ``r21 = h2/h1`` and
``r32 = h3/h2``, compute the observed order, the Richardson extrapolation and
the fine-grid GCI (safety factor 1.25 for three-grid studies).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class GCIResult:
    phi: tuple  # (fine, medium, coarse)
    h: tuple
    observed_order: float
    extrapolated: float
    approx_rel_error: float  # e_a^21 = |(phi1 - phi2) / phi1|
    extrap_rel_error: float  # e_ext^21 = |(phi_ext - phi1) / phi_ext|
    gci_fine: float  # relative, e.g. 0.004 = 0.4 %
    gci_medium: float
    asymptotic_ratio: float  # ~1 in the asymptotic range
    oscillatory: bool

    def to_dict(self) -> dict:
        return asdict(self)


def gci(phi1: float, phi2: float, phi3: float, h1: float, h2: float, h3: float, fs: float = 1.25,
        max_iter: int = 100) -> GCIResult:
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = phi2 - phi1, phi3 - phi2
    if e21 == 0 or e32 == 0:
        raise ValueError("identical solutions on two grids: GCI undefined")
    s = math.copysign(1.0, e32 / e21)
    oscillatory = s < 0
    # fixed-point iteration for the order with non-constant refinement ratio
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(max_iter):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    phi_ext = (r21 ** p * phi1 - phi2) / (r21 ** p - 1.0)
    ea21 = abs((phi1 - phi2) / phi1)
    eext = abs((phi_ext - phi1) / phi_ext)
    gci21 = fs * ea21 / (r21 ** p - 1.0)
    ea32 = abs((phi2 - phi3) / phi2)
    gci32 = fs * ea32 / (r32 ** p - 1.0)
    # asymptotic range check with absolute GCIs (Roache): GCI_32 / (r^p GCI_21) ~ 1
    ratio = (abs(e32) / (r32 ** p - 1.0)) / (r21 ** p * abs(e21) / (r21 ** p - 1.0))
    return GCIResult((phi1, phi2, phi3), (h1, h2, h3), p, phi_ext, ea21, eext, gci21, gci32, ratio, oscillatory)
