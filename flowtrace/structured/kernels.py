"""Fused multi-core CPU kernels (Numba) for the hot loops of the MAC solver.

Each kernel computes one output value per thread-row with no temporaries, so
the stencils run at close to memory bandwidth instead of paying NumPy's
one-pass-per-operator cost. They are bit-for-bit equivalent in algorithm to
the vectorised NumPy/CuPy path in ``solver.py`` (verified by the test suite)
and are used only when the ``numba`` backend is selected.

Scheme codes: 0 central, 1 upwind, 2 QUICK, 3 van Leer TVD.
Boundary patch modes: 0 none, 1 fixed value, 2 adjacent interior value.
"""

from __future__ import annotations

try:
    import numba
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba optional
    HAVE_NUMBA = False

SCHEME_CODES = {"central": 0, "upwind": 1, "quick": 2, "vanleer": 3}
G = 2

if HAVE_NUMBA:
    _opts = dict(parallel=True, fastmath=False, cache=True, nogil=True)

    @njit(inline="always", cache=True)
    def _tvd(c, u, d):
        d1 = c - u
        d2 = d - c
        return c + 0.5 * (d1 * abs(d2) + abs(d1) * d2) / (abs(d1) + abs(d2) + 1e-30)

    @njit(inline="always", cache=True)
    def _face(a, b, aa, bb, w, scheme):
        if scheme == 0:
            return 0.5 * (a + b)
        if scheme == 1:
            return a if w >= 0 else b
        if scheme == 2:
            if w >= 0:
                return 0.75 * a + 0.375 * b - 0.125 * aa
            return 0.75 * b + 0.375 * a - 0.125 * bb
        if w >= 0:
            return _tvd(a, aa, b)
        return _tvd(b, bb, a)

    @njit(**_opts)
    def conv_u_nb(U, V, out, nx, ny, dx, dy, scheme, bmode, bval, tmode, tval):
        for i in prange(nx + 1):
            I = i + G
            for j in range(ny):
                J = j + G
                # x-fluxes at the cell centres either side of face i
                a, b = U[I, J], U[I + 1, J]
                w = 0.5 * (a + b)
                fe = w * _face(a, b, U[I - 1, J], U[I + 2, J], w, scheme)
                a, b = U[I - 1, J], U[I, J]
                w = 0.5 * (a + b)
                fw = w * _face(a, b, U[I - 2, J], U[I + 1, J], w, scheme)
                # y-fluxes at the corners above/below
                w = 0.5 * (V[I - 1, J + 1] + V[I, J + 1])
                if j == ny - 1 and tmode != 0:
                    gn = w * (tval if tmode == 1 else U[I, J])
                else:
                    gn = w * _face(U[I, J], U[I, J + 1], U[I, J - 1], U[I, J + 2], w, scheme)
                w = 0.5 * (V[I - 1, J] + V[I, J])
                if j == 0 and bmode != 0:
                    gs = w * (bval if bmode == 1 else U[I, J])
                else:
                    gs = w * _face(U[I, J - 1], U[I, J], U[I, J - 2], U[I, J + 1], w, scheme)
                out[i, j] = (fe - fw) / dx + (gn - gs) / dy

    @njit(**_opts)
    def conv_v_nb(U, V, out, nx, ny, dx, dy, scheme, lmode, lval, rmode, rval):
        for i in prange(nx):
            I = i + G
            for j in range(ny + 1):
                J = j + G
                a, b = V[I, J], V[I, J + 1]
                w = 0.5 * (a + b)
                gn = w * _face(a, b, V[I, J - 1], V[I, J + 2], w, scheme)
                a, b = V[I, J - 1], V[I, J]
                w = 0.5 * (a + b)
                gs = w * _face(a, b, V[I, J - 2], V[I, J + 1], w, scheme)
                w = 0.5 * (U[I + 1, J - 1] + U[I + 1, J])
                if i == nx - 1 and rmode != 0:
                    fe = w * (rval if rmode == 1 else V[I, J])
                else:
                    fe = w * _face(V[I, J], V[I + 1, J], V[I - 1, J], V[I + 2, J], w, scheme)
                w = 0.5 * (U[I, J - 1] + U[I, J])
                if i == 0 and lmode != 0:
                    fw = w * (lval if lmode == 1 else V[I, J])
                else:
                    fw = w * _face(V[I - 1, J], V[I, J], V[I - 2, J], V[I + 1, J], w, scheme)
                out[i, j] = (fe - fw) / dx + (gn - gs) / dy

    @njit(**_opts)
    def conv_T_nb(U, V, T, out, nx, ny, dx, dy, scheme, modes, vals):
        # modes/vals: left, right, bottom, top
        for i in prange(nx):
            I = i + G
            for j in range(ny):
                J = j + G
                w = U[I + 1, J]
                if i == nx - 1 and modes[1] != 0:
                    fe = w * (vals[1] if modes[1] == 1 else T[I, J])
                else:
                    fe = w * _face(T[I, J], T[I + 1, J], T[I - 1, J], T[I + 2, J], w, scheme)
                w = U[I, J]
                if i == 0 and modes[0] != 0:
                    fw = w * (vals[0] if modes[0] == 1 else T[I, J])
                else:
                    fw = w * _face(T[I - 1, J], T[I, J], T[I - 2, J], T[I + 1, J], w, scheme)
                w = V[I, J + 1]
                if j == ny - 1 and modes[3] != 0:
                    gn = w * (vals[3] if modes[3] == 1 else T[I, J])
                else:
                    gn = w * _face(T[I, J], T[I, J + 1], T[I, J - 1], T[I, J + 2], w, scheme)
                w = V[I, J]
                if j == 0 and modes[2] != 0:
                    gs = w * (vals[2] if modes[2] == 1 else T[I, J])
                else:
                    gs = w * _face(T[I, J - 1], T[I, J], T[I, J - 2], T[I, J + 1], w, scheme)
                out[i, j] = (fe - fw) / dx + (gn - gs) / dy

    @njit(**_opts)
    def lap_nb(A, out, nxa, nya, dx, dy):
        idx2 = 1.0 / (dx * dx)
        idy2 = 1.0 / (dy * dy)
        for i in prange(nxa):
            I = i + G
            for j in range(nya):
                J = j + G
                c = A[I, J]
                out[i, j] = (A[I + 1, J] - 2.0 * c + A[I - 1, J]) * idx2 + (A[I, J + 1] - 2.0 * c + A[I, J - 1]) * idy2

    @njit(**_opts)
    def laplacian5_numba(p, out):
        n, m = out.shape
        for i in prange(n):
            for j in range(m):
                out[i, j] = p[i + 2, j + 1] + p[i, j + 1] + p[i + 1, j + 2] + p[i + 1, j] - 4.0 * p[i + 1, j + 1]

    # ---------------- multigrid ----------------
    @njit(**_opts)
    def rbgs_nb(x, b, cx, cy, inv_diag, colour):
        nx, ny = x.shape
        for i in prange(nx):
            for j in range((i + colour) % 2, ny, 2):
                s = b[i, j]
                if i > 0:
                    s += cx[i, j] * x[i - 1, j]
                if i < nx - 1:
                    s += cx[i + 1, j] * x[i + 1, j]
                if j > 0:
                    s += cy[i, j] * x[i, j - 1]
                if j < ny - 1:
                    s += cy[i, j + 1] * x[i, j + 1]
                x[i, j] = s * inv_diag[i, j]

    @njit(**_opts)
    def apply_nb(x, cx, cy, diag, out):
        nx, ny = x.shape
        for i in prange(nx):
            for j in range(ny):
                s = diag[i, j] * x[i, j]
                if i > 0:
                    s -= cx[i, j] * x[i - 1, j]
                if i < nx - 1:
                    s -= cx[i + 1, j] * x[i + 1, j]
                if j > 0:
                    s -= cy[i, j] * x[i, j - 1]
                if j < ny - 1:
                    s -= cy[i, j + 1] * x[i, j + 1]
                out[i, j] = s
else:  # pragma: no cover
    conv_u_nb = conv_v_nb = conv_T_nb = lap_nb = laplacian5_numba = rbgs_nb = apply_nb = None
