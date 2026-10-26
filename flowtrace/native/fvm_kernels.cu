// FlowTrace finite-volume kernels for NVIDIA GPUs (CUDA C++, double precision).
//
// Compiled at first use by NVRTC through cupy.RawModule (flowtrace/native/cuda.py), so no
// host compiler is needed. The same file builds with nvcc where one is available:
//   nvcc -cubin -arch=sm_89 fvm_kernels.cu
// Arguments mirror the C++ kernels in fvm_kernels.cpp; tests/test_native.py checks both
// against the NumPy/CuPy reference.

typedef long long i64;

extern "C" __global__ void ft_face_interp(i64 nf, i64 k, const double* w, const i64* P, const i64* N,
                                          const double* phi, double* out) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nf * k) return;
    i64 f = t / k, c = t - f * k;
    double wf = w[f];
    out[t] = wf * phi[P[f] * k + c] + (1.0 - wf) * phi[N[f] * k + c];
}

extern "C" __global__ void ft_ls_gradient(i64 nc, i64 dim, const int* iptr, const int* iidx, const int* bptr,
                                          const int* bidx, const i64* P, const i64* N, const double* wl,
                                          const double* d, const double* wlb, const double* db, const double* Ginv,
                                          const double* phi, const double* phib, double* out) {
    i64 c = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double r0 = 0.0, r1 = 0.0, r2 = 0.0;
    for (int j = iptr[c]; j < iptr[c + 1]; ++j) {
        i64 f = iidx[j];
        double coef = wl[f] * (phi[N[f]] - phi[P[f]]);
        r0 += coef * d[f * dim];
        r1 += coef * d[f * dim + 1];
        if (dim == 3) r2 += coef * d[f * dim + 2];
    }
    for (int j = bptr[c]; j < bptr[c + 1]; ++j) {
        i64 f = bidx[j];
        double coef = wlb[f] * (phib[f] - phi[c]);
        r0 += coef * db[f * dim];
        r1 += coef * db[f * dim + 1];
        if (dim == 3) r2 += coef * db[f * dim + 2];
    }
    const double* G = Ginv + c * dim * dim;
    for (i64 a = 0; a < dim; ++a) {
        double s = G[a * dim] * r0 + G[a * dim + 1] * r1;
        if (dim == 3) s += G[a * dim + 2] * r2;
        out[c * dim + a] = s;
    }
}

extern "C" __global__ void ft_gg_gradient(i64 nc, i64 dim, const int* iptr, const int* iidx, const double* isgn,
                                          const int* bptr, const int* bidx, const i64* P, const i64* N,
                                          const double* w, const double* S, const double* Sb, const double* vol,
                                          const double* dxm, const double* phi, const double* phib,
                                          const double* grad, int use_grad, double* out) {
    i64 c = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double s[3] = {0.0, 0.0, 0.0};
    for (int j = iptr[c]; j < iptr[c + 1]; ++j) {
        i64 f = iidx[j], p = P[f], n = N[f];
        double wf = w[f], wn = 1.0 - wf;
        double pf = wf * phi[p] + wn * phi[n];
        if (use_grad)
            for (i64 a = 0; a < dim; ++a) pf += (wf * grad[p * dim + a] + wn * grad[n * dim + a]) * dxm[f * dim + a];
        double coef = isgn[j] * pf;
        for (i64 a = 0; a < dim; ++a) s[a] += coef * S[f * dim + a];
    }
    for (int j = bptr[c]; j < bptr[c + 1]; ++j) {
        i64 f = bidx[j];
        for (i64 a = 0; a < dim; ++a) s[a] += phib[f] * Sb[f * dim + a];
    }
    double iv = 1.0 / vol[c];
    for (i64 a = 0; a < dim; ++a) out[c * dim + a] = s[a] * iv;
}

// assembly: data must be zero on entry; one thread per cell writes its diagonal and adds the
// coefficients of the faces it owns (P side) or neighbours (N side). Cell-centric, so no atomics.
extern "C" __global__ void ft_assemble(i64 nc, const int* iptr, const int* iidx, const double* isgn,
                                       const int* slot_diag, const int* slot_PN, const int* slot_NP,
                                       const double* diag, const double* aPN, const double* aNP, double* data) {
    i64 c = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    data[slot_diag[c]] = diag[c];
    for (int j = iptr[c]; j < iptr[c + 1]; ++j) {
        i64 f = iidx[j];
        if (isgn[j] > 0) data[slot_PN[f]] += aPN[f];  // row c = P: coefficient of N
        else data[slot_NP[f]] += aNP[f];               // row c = N: coefficient of P
    }
}

// ------------------------------------------------------------------ BiCGSTAB building blocks
// Block reduction of up to two dot products into acc[0..1] (acc must be zero on entry).
__device__ __forceinline__ void block_add2(double a, double b, double* acc, int two) {
    __shared__ double sa[256], sb[256];
    int t = threadIdx.x;
    sa[t] = a;
    sb[t] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            sa[t] += sa[t + s];
            sb[t] += sb[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        atomicAdd(acc, sa[0]);
        if (two) atomicAdd(acc + 1, sb[0]);
    }
}

extern "C" __global__ void ft_dot2(i64 n, const double* a, const double* b, const double* c, const double* d,
                                   int two, double* acc) {
    double x = 0.0, y = 0.0;
    for (i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        x += a[i] * b[i];
        if (two) y += c[i] * d[i];
    }
    block_add2(x, y, acc, two);
}

// p = r + beta (p - omega v);  h = inv p
extern "C" __global__ void ft_bicg_p(i64 n, double beta, double omega, const double* r, const double* v,
                                     const double* inv, double* p, double* h) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double pi = r[i] + beta * (p[i] - omega * v[i]);
    p[i] = pi;
    h[i] = inv[i] * pi;
}

// s = r - alpha v;  hs = inv s;  acc[0] += s.s
extern "C" __global__ void ft_bicg_s(i64 n, double alpha, const double* r, const double* v, const double* inv,
                                     double* s, double* hs, double* acc) {
    double ss = 0.0;
    for (i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        double si = r[i] - alpha * v[i];
        s[i] = si;
        hs[i] = inv[i] * si;
        ss += si * si;
    }
    block_add2(ss, 0.0, acc, 0);
}

// xn = x + alpha h + omega hs;  r = s - omega t;  acc[0] += r.r, acc[1] += r_hat.r
extern "C" __global__ void ft_bicg_x(i64 n, double alpha, double omega, const double* x, const double* h,
                                     const double* hs, const double* s, const double* t, const double* r_hat,
                                     double* xn, double* r, double* acc) {
    double rr = 0.0, hr = 0.0;
    for (i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        xn[i] = x[i] + alpha * h[i] + omega * hs[i];
        double ri = s[i] - omega * t[i];
        r[i] = ri;
        rr += ri * ri;
        hr += r_hat[i] * ri;
    }
    block_add2(rr, hr, acc, 1);
}
