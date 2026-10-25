"""Fused CUDA C++ kernels for the MAC solver's hot loops (GPU backend).

Profiling the CuPy path on an RTX 4060 Laptop GPU showed the time step to be
launch-bound: the vectorised convection terms issue dozens of small
elementwise kernels per component. These kernels compute each output point
in one thread with the same arithmetic as the Numba CPU kernels in
``kernels.py`` (central / upwind / QUICK / van Leer, boundary patches), so a
whole convection term is a single launch.

They are compiled at run time by NVRTC through ``cupy.RawModule`` (no
compiler needed on the user's machine) for float32 and float64. If anything
fails (no CuPy, old driver, compile error) ``available()`` returns False and
the solver keeps the vectorised CuPy path; ``FLOWTRACE_NO_CUDA_KERNELS=1``
forces that fallback (used by the equality tests).
"""

from __future__ import annotations

import os

_SRC = r"""
template<typename T> __device__ __forceinline__ T tvd(T c, T u, T d) {
    T d1 = c - u, d2 = d - c;
    return c + (T)0.5 * (d1 * fabs(d2) + fabs(d1) * d2) / (fabs(d1) + fabs(d2) + (T)1e-30);
}
template<typename T> __device__ __forceinline__ T face(T a, T b, T aa, T bb, T w, int s) {
    if (s == 0) return (T)0.5 * (a + b);
    if (s == 1) return w >= 0 ? a : b;
    if (s == 2) return w >= 0 ? (T)0.75 * a + (T)0.375 * b - (T)0.125 * aa
                              : (T)0.75 * b + (T)0.375 * a - (T)0.125 * bb;
    return w >= 0 ? tvd(a, aa, b) : tvd(b, bb, a);
}
#define G 2
template<typename T>
__device__ void conv_u(const T* U, const T* V, T* out, int nx, int ny, int su, int sv, T dx, T dy,
                       int scheme, int bmode, T bval, int tmode, T tval) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)(nx + 1) * ny) return;
    int i = idx / ny, j = idx % ny, I = i + G, J = j + G;
#define Uat(a, b) U[(long)(a) * su + (b)]
#define Vat(a, b) V[(long)(a) * sv + (b)]
    T a = Uat(I, J), b = Uat(I + 1, J), w = (T)0.5 * (a + b);
    T fe = w * face(a, b, Uat(I - 1, J), Uat(I + 2, J), w, scheme);
    a = Uat(I - 1, J); b = Uat(I, J); w = (T)0.5 * (a + b);
    T fw = w * face(a, b, Uat(I - 2, J), Uat(I + 1, J), w, scheme);
    w = (T)0.5 * (Vat(I - 1, J + 1) + Vat(I, J + 1));
    T gn = (j == ny - 1 && tmode != 0) ? w * (tmode == 1 ? tval : Uat(I, J))
                                       : w * face(Uat(I, J), Uat(I, J + 1), Uat(I, J - 1), Uat(I, J + 2), w, scheme);
    w = (T)0.5 * (Vat(I - 1, J) + Vat(I, J));
    T gs = (j == 0 && bmode != 0) ? w * (bmode == 1 ? bval : Uat(I, J))
                                  : w * face(Uat(I, J - 1), Uat(I, J), Uat(I, J - 2), Uat(I, J + 1), w, scheme);
    out[idx] = (fe - fw) / dx + (gn - gs) / dy;
#undef Uat
#undef Vat
}
template<typename T>
__device__ void conv_v(const T* U, const T* V, T* out, int nx, int ny, int su, int sv, T dx, T dy,
                       int scheme, int lmode, T lval, int rmode, T rval) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)nx * (ny + 1)) return;
    int i = idx / (ny + 1), j = idx % (ny + 1), I = i + G, J = j + G;
#define Uat(a, b) U[(long)(a) * su + (b)]
#define Vat(a, b) V[(long)(a) * sv + (b)]
    T a = Vat(I, J), b = Vat(I, J + 1), w = (T)0.5 * (a + b);
    T gn = w * face(a, b, Vat(I, J - 1), Vat(I, J + 2), w, scheme);
    a = Vat(I, J - 1); b = Vat(I, J); w = (T)0.5 * (a + b);
    T gs = w * face(a, b, Vat(I, J - 2), Vat(I, J + 1), w, scheme);
    w = (T)0.5 * (Uat(I + 1, J - 1) + Uat(I + 1, J));
    T fe = (i == nx - 1 && rmode != 0) ? w * (rmode == 1 ? rval : Vat(I, J))
                                       : w * face(Vat(I, J), Vat(I + 1, J), Vat(I - 1, J), Vat(I + 2, J), w, scheme);
    w = (T)0.5 * (Uat(I, J - 1) + Uat(I, J));
    T fw = (i == 0 && lmode != 0) ? w * (lmode == 1 ? lval : Vat(I, J))
                                  : w * face(Vat(I - 1, J), Vat(I, J), Vat(I - 2, J), Vat(I + 1, J), w, scheme);
    out[idx] = (fe - fw) / dx + (gn - gs) / dy;
#undef Uat
#undef Vat
}
template<typename T>
__device__ void conv_T(const T* U, const T* V, const T* Tc, T* out, int nx, int ny, int su, int sv, int st,
                       T dx, T dy, int scheme, int ml, T vl, int mr, T vr, int mb, T vb, int mt, T vt) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)nx * ny) return;
    int i = idx / ny, j = idx % ny, I = i + G, J = j + G;
#define Uat(a, b) U[(long)(a) * su + (b)]
#define Vat(a, b) V[(long)(a) * sv + (b)]
#define Tat(a, b) Tc[(long)(a) * st + (b)]
    T w = Uat(I + 1, J);
    T fe = (i == nx - 1 && mr != 0) ? w * (mr == 1 ? vr : Tat(I, J))
                                    : w * face(Tat(I, J), Tat(I + 1, J), Tat(I - 1, J), Tat(I + 2, J), w, scheme);
    w = Uat(I, J);
    T fw = (i == 0 && ml != 0) ? w * (ml == 1 ? vl : Tat(I, J))
                               : w * face(Tat(I - 1, J), Tat(I, J), Tat(I - 2, J), Tat(I + 1, J), w, scheme);
    w = Vat(I, J + 1);
    T gn = (j == ny - 1 && mt != 0) ? w * (mt == 1 ? vt : Tat(I, J))
                                    : w * face(Tat(I, J), Tat(I, J + 1), Tat(I, J - 1), Tat(I, J + 2), w, scheme);
    w = Vat(I, J);
    T gs = (j == 0 && mb != 0) ? w * (mb == 1 ? vb : Tat(I, J))
                               : w * face(Tat(I, J - 1), Tat(I, J), Tat(I, J - 2), Tat(I, J + 1), w, scheme);
    out[idx] = (fe - fw) / dx + (gn - gs) / dy;
#undef Uat
#undef Vat
#undef Tat
}
template<typename T>
__device__ void lap(const T* A, T* out, int nxa, int nya, int sa, T idx2, T idy2) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)nxa * nya) return;
    int i = idx / nya, j = idx % nya;
    long c = (long)(i + G) * sa + (j + G);
    T v = A[c];
    out[idx] = (A[c + sa] - (T)2 * v + A[c - sa]) * idx2 + (A[c + 1] - (T)2 * v + A[c - 1]) * idy2;
}
#define INST(T, S) \
extern "C" __global__ void conv_u_##S(const T* U, const T* V, T* out, int nx, int ny, int su, int sv, T dx, T dy, \
    int scheme, int bmode, T bval, int tmode, T tval) { conv_u<T>(U, V, out, nx, ny, su, sv, dx, dy, scheme, bmode, bval, tmode, tval); } \
extern "C" __global__ void conv_v_##S(const T* U, const T* V, T* out, int nx, int ny, int su, int sv, T dx, T dy, \
    int scheme, int lmode, T lval, int rmode, T rval) { conv_v<T>(U, V, out, nx, ny, su, sv, dx, dy, scheme, lmode, lval, rmode, rval); } \
extern "C" __global__ void conv_T_##S(const T* U, const T* V, const T* Tc, T* out, int nx, int ny, int su, int sv, int st, \
    T dx, T dy, int scheme, int ml, T vl, int mr, T vr, int mb, T vb, int mt, T vt) { \
    conv_T<T>(U, V, Tc, out, nx, ny, su, sv, st, dx, dy, scheme, ml, vl, mr, vr, mb, vb, mt, vt); } \
extern "C" __global__ void lap_##S(const T* A, T* out, int nxa, int nya, int sa, T idx2, T idy2) { \
    lap<T>(A, out, nxa, nya, sa, idx2, idy2); }
INST(double, f64)
INST(float, f32)
"""

_module = None
_failed = None
BLOCK = 256


def available() -> bool:
    global _module, _failed
    if os.environ.get("FLOWTRACE_NO_CUDA_KERNELS"):
        return False
    if _module is not None:
        return True
    if _failed is not None:
        return False
    try:
        import cupy

        if getattr(cupy, "__fake__", False):  # NumPy-backed test stand-in: no CUDA, don't cache
            return False
        _module = cupy.RawModule(code=_SRC, options=("--std=c++14",))
        _module.get_function("conv_u_f64")  # forces compilation
        return True
    except Exception as exc:  # pragma: no cover - depends on the CUDA install
        _failed = str(exc)
        return False


def failure_reason():
    return _failed


def _fn(name, dtype):
    import numpy as np

    return _module.get_function(f"{name}_{'f64' if np.dtype(dtype) == np.float64 else 'f32'}")


def _launch(fn, n, args):
    grid = ((n + BLOCK - 1) // BLOCK,)
    fn(grid, (BLOCK,), args)


def _scalar(dtype, v):
    import numpy as np

    return np.dtype(dtype).type(v)


def conv_u(U, V, nx, ny, dx, dy, scheme, bmode, bval, tmode, tval):
    import cupy

    out = cupy.empty((nx + 1, ny), dtype=U.dtype)
    s = lambda v: _scalar(U.dtype, v)  # noqa: E731
    _launch(_fn("conv_u", U.dtype), (nx + 1) * ny,
            (U, V, out, int(nx), int(ny), int(U.shape[1]), int(V.shape[1]), s(dx), s(dy), int(scheme),
             int(bmode), s(bval), int(tmode), s(tval)))
    return out


def conv_v(U, V, nx, ny, dx, dy, scheme, lmode, lval, rmode, rval):
    import cupy

    out = cupy.empty((nx, ny + 1), dtype=U.dtype)
    s = lambda v: _scalar(U.dtype, v)  # noqa: E731
    _launch(_fn("conv_v", U.dtype), nx * (ny + 1),
            (U, V, out, int(nx), int(ny), int(U.shape[1]), int(V.shape[1]), s(dx), s(dy), int(scheme),
             int(lmode), s(lval), int(rmode), s(rval)))
    return out


def conv_T(U, V, T, nx, ny, dx, dy, scheme, modes, vals):
    import cupy

    out = cupy.empty((nx, ny), dtype=U.dtype)
    s = lambda v: _scalar(U.dtype, v)  # noqa: E731
    _launch(_fn("conv_T", U.dtype), nx * ny,
            (U, V, T, out, int(nx), int(ny), int(U.shape[1]), int(V.shape[1]), int(T.shape[1]), s(dx), s(dy),
             int(scheme), int(modes[0]), s(vals[0]), int(modes[1]), s(vals[1]), int(modes[2]), s(vals[2]),
             int(modes[3]), s(vals[3])))
    return out


def lap(A, nxa, nya, dx, dy):
    import cupy

    out = cupy.empty((nxa, nya), dtype=A.dtype)
    _launch(_fn("lap", A.dtype), nxa * nya,
            (A, out, int(nxa), int(nya), int(A.shape[1]), _scalar(A.dtype, 1.0 / (dx * dx)),
             _scalar(A.dtype, 1.0 / (dy * dy))))
    return out
