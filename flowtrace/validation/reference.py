"""Published reference data used to validate FlowTrace.

* Ghia, Ghia & Shin (1982), J. Comput. Phys. 48:387-411 -- lid-driven cavity
  centreline velocities (Tables I and II).
* de Vahl Davis (1983), Int. J. Numer. Meth. Fluids 3:249-264 -- natural
  convection in a differentially heated square cavity (Pr = 0.71).
"""

import numpy as np

GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                   0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U = {
    100: [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581,
          -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000],
    400: [1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477,
          -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000],
    1000: [1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080,
           -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000],
}
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
                   0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_V = {
    100: [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454,
          0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000],
    400: [0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 0.05186,
          0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000],
    1000: [0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526,
           0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000],
}

# de Vahl Davis (1983): mean hot-wall Nusselt number and midline velocity maxima
DVD = {
    1e3: {"nu": 1.118, "umax": 3.649, "y_umax": 0.813, "vmax": 3.697, "x_vmax": 0.178},
    1e4: {"nu": 2.243, "umax": 16.178, "y_umax": 0.823, "vmax": 19.617, "x_vmax": 0.119},
    1e5: {"nu": 4.519, "umax": 34.73, "y_umax": 0.855, "vmax": 68.59, "x_vmax": 0.066},
    1e6: {"nu": 8.800, "umax": 64.63, "y_umax": 0.850, "vmax": 219.36, "x_vmax": 0.0379},
}


def ghia(re: float):
    """Return ((y, u), (x, v)) Ghia data for the closest tabulated Re, or None."""
    key = min(GHIA_U, key=lambda k: abs(k - re))
    if abs(key - re) / key > 0.05:
        return None
    return (GHIA_Y, np.array(GHIA_U[key])), (GHIA_X, np.array(GHIA_V[key]))


def compare_ghia(re, y_sim, u_sim, x_sim, v_sim) -> dict:
    """Max/RMS deviation of simulated centreline profiles from Ghia et al."""
    ref = ghia(re)
    if ref is None:
        return {}
    (gy, gu), (gx, gv) = ref
    yy = np.concatenate([[0.0], y_sim, [1.0]])
    uu = np.concatenate([[0.0], u_sim, [1.0]])
    xx = np.concatenate([[0.0], x_sim, [1.0]])
    vv = np.concatenate([[0.0], v_sim, [0.0]])
    du = np.interp(gy, yy, uu) - gu
    dv = np.interp(gx, xx, vv) - gv
    return {
        "u_max_err": float(np.abs(du).max()),
        "u_rms_err": float(np.sqrt(np.mean(du ** 2))),
        "v_max_err": float(np.abs(dv).max()),
        "v_rms_err": float(np.sqrt(np.mean(dv ** 2))),
    }
