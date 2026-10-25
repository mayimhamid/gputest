"""Node distributions for graded structured blocks (wall-resolved RANS meshes)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def geometric(n: int, length: float, first: float, start: float = 0.0) -> np.ndarray:
    """n cells on [start, start+length], first cell ``first``, constant growth ratio."""
    if n < 1:
        raise ValueError("need at least one cell")
    if first * n >= length:  # uniform (or shrinking) would be needed
        return start + np.linspace(0.0, length, n + 1)
    r = brentq(lambda q: first * (q ** n - 1) / (q - 1) - length, 1.0 + 1e-12, 10.0)
    h = first * r ** np.arange(n)
    x = np.concatenate([[0.0], np.cumsum(h)])
    x *= length / x[-1]
    return start + x


def growth_ratio(nodes: np.ndarray) -> float:
    h = np.diff(nodes)
    return float(np.max(np.maximum(h[1:] / h[:-1], h[:-1] / h[1:])))


def two_sided(n: int, length: float, first: float, start: float = 0.0) -> np.ndarray:
    """Clustered towards both ends (n even): mirrored geometric halves."""
    half = geometric(n // 2, 0.5 * length, first)
    x = np.concatenate([half, length - half[::-1][1:]])
    return start + x


def first_cell_height(y_plus: float, re_length: float, cf: float, length: float = 1.0) -> float:
    """Wall-normal size of the first cell for a target y+ (cf: skin-friction estimate)."""
    u_tau_over_u = np.sqrt(0.5 * cf)
    return y_plus * length / (re_length * u_tau_over_u)
