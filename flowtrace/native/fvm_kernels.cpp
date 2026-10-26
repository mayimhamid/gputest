// FlowTrace native finite-volume kernels (C++17 + OpenMP). See fvm_kernels.h for the API.
//
// Built at first use by flowtrace.native with g++/clang++:
//   g++ -O3 -march=native -fopenmp -shared [-static on Windows] fvm_kernels.cpp -o <cache>/flowtrace_fvm.{dll,so}
// The NumPy/SciPy code path remains the reference; tests/test_native.py checks both agree.

#include "fvm_kernels.h"

#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int64_t PAR_MIN = 4096;  // below this many rows a parallel region costs more than it saves

inline void spmv(int64_t n, const int32_t* indptr, const int32_t* indices, const double* data, const double* x,
                 double* y) {
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int32_t j = indptr[i]; j < indptr[i + 1]; ++j) s += data[j] * x[indices[j]];
        y[i] = s;
    }
}

inline double dot(int64_t n, const double* a, const double* b) {
    double s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : s) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

}  // namespace

extern "C" {

int ft_abi_version(void) { return 2; }

int ft_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void ft_csr_spmm(int64_t nrows, int64_t k, const int32_t* indptr, const int32_t* indices, const double* data,
                 const double* X, double* Y) {
    if (k == 1) {
        spmv(nrows, indptr, indices, data, X, Y);
        return;
    }
#pragma omp parallel for schedule(static) if (nrows > PAR_MIN)
    for (int64_t i = 0; i < nrows; ++i) {
        double* y = Y + i * k;
        for (int64_t c = 0; c < k; ++c) y[c] = 0.0;
        for (int32_t j = indptr[i]; j < indptr[i + 1]; ++j) {
            const double a = data[j];
            const double* x = X + static_cast<int64_t>(indices[j]) * k;
            for (int64_t c = 0; c < k; ++c) y[c] += a * x[c];
        }
    }
}

void ft_face_interp(int64_t nf, int64_t k, const double* w, const int64_t* P, const int64_t* N, const double* phi,
                    double* out) {
#pragma omp parallel for schedule(static) if (nf > PAR_MIN)
    for (int64_t f = 0; f < nf; ++f) {
        const double wf = w[f], wn = 1.0 - wf;
        const double* a = phi + P[f] * k;
        const double* b = phi + N[f] * k;
        double* o = out + f * k;
        for (int64_t c = 0; c < k; ++c) o[c] = wf * a[c] + wn * b[c];
    }
}

void ft_ls_gradient(int64_t nc, int64_t dim, const int32_t* iptr, const int32_t* iidx, const int32_t* bptr,
                    const int32_t* bidx, const int64_t* P, const int64_t* N, const double* wl, const double* d,
                    const double* wlb, const double* db, const double* Ginv, const double* phi, const double* phib,
                    double* out) {
#pragma omp parallel for schedule(static) if (nc > PAR_MIN)
    for (int64_t c = 0; c < nc; ++c) {
        double r[3] = {0.0, 0.0, 0.0};
        for (int32_t j = iptr[c]; j < iptr[c + 1]; ++j) {
            const int64_t f = iidx[j];
            const double coef = wl[f] * (phi[N[f]] - phi[P[f]]);
            for (int64_t a = 0; a < dim; ++a) r[a] += coef * d[f * dim + a];
        }
        for (int32_t j = bptr[c]; j < bptr[c + 1]; ++j) {
            const int64_t f = bidx[j];
            const double coef = wlb[f] * (phib[f] - phi[c]);
            for (int64_t a = 0; a < dim; ++a) r[a] += coef * db[f * dim + a];
        }
        const double* G = Ginv + c * dim * dim;
        for (int64_t a = 0; a < dim; ++a) {
            double s = 0.0;
            for (int64_t b = 0; b < dim; ++b) s += G[a * dim + b] * r[b];
            out[c * dim + a] = s;
        }
    }
}

void ft_gg_gradient(int64_t nc, int64_t dim, const int32_t* iptr, const int32_t* iidx, const double* isgn,
                    const int32_t* bptr, const int32_t* bidx, const int64_t* P, const int64_t* N, const double* w,
                    const double* S, const double* Sb, const double* vol, const double* dxm, const double* phi,
                    const double* phib, const double* grad, double* out) {
#pragma omp parallel for schedule(static) if (nc > PAR_MIN)
    for (int64_t c = 0; c < nc; ++c) {
        double s[3] = {0.0, 0.0, 0.0};
        for (int32_t j = iptr[c]; j < iptr[c + 1]; ++j) {
            const int64_t f = iidx[j];
            const int64_t p = P[f], n = N[f];
            const double wf = w[f], wn = 1.0 - wf;
            double pf = wf * phi[p] + wn * phi[n];
            if (grad) {
                for (int64_t a = 0; a < dim; ++a)
                    pf += (wf * grad[p * dim + a] + wn * grad[n * dim + a]) * dxm[f * dim + a];
            }
            const double coef = isgn[j] * pf;
            for (int64_t a = 0; a < dim; ++a) s[a] += coef * S[f * dim + a];
        }
        for (int32_t j = bptr[c]; j < bptr[c + 1]; ++j) {
            const int64_t f = bidx[j];
            for (int64_t a = 0; a < dim; ++a) s[a] += phib[f] * Sb[f * dim + a];
        }
        const double iv = 1.0 / vol[c];
        for (int64_t a = 0; a < dim; ++a) out[c * dim + a] = s[a] * iv;
    }
}

void ft_assemble(int64_t nnz, int64_t nc, int64_t nf, const int32_t* slot_diag, const int32_t* slot_PN,
                 const int32_t* slot_NP, const double* diag, const double* aPN, const double* aNP, double* data) {
    std::memset(data, 0, sizeof(double) * static_cast<size_t>(nnz));
    for (int64_t c = 0; c < nc; ++c) data[slot_diag[c]] = diag[c];
    // sequential: two faces may share a (P, N) slot (e.g. a cell pair joined by two faces)
    for (int64_t f = 0; f < nf; ++f) {
        data[slot_PN[f]] += aPN[f];
        data[slot_NP[f]] += aNP[f];
    }
}

int ft_bicgstab_jacobi(int64_t n, const int32_t* indptr, const int32_t* indices, const double* data, const double* b,
                       const double* inv, double* x, double tol, int64_t maxiter, int relative_to_initial,
                       double* work, int64_t* iters, double* res) {
    // work: 7 n doubles (r, r_hat, p, v, phat/shat, s, t)
    double* r = work;
    double* r_hat = work + n;
    double* p = work + 2 * n;
    double* v = work + 3 * n;
    double* h = work + 4 * n;
    double* s = work + 5 * n;
    double* t = work + 6 * n;
    spmv(n, indptr, indices, data, x, r);
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
    for (int64_t i = 0; i < n; ++i) r[i] = b[i] - r[i];
    double rnorm = std::sqrt(dot(n, r, r));
    const double b_abs = std::sqrt(dot(n, b, b));
    double bnorm = relative_to_initial ? rnorm : b_abs;
    if (bnorm == 0.0) bnorm = 1.0;
    *iters = 0;
    *res = rnorm / bnorm;
    if (rnorm / bnorm < tol || rnorm <= 1e-14 * b_abs) return 1;
    std::memcpy(r_hat, r, sizeof(double) * static_cast<size_t>(n));
    std::memset(p, 0, sizeof(double) * static_cast<size_t>(n));
    std::memset(v, 0, sizeof(double) * static_cast<size_t>(n));
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    int64_t k = 0;
    for (k = 1; k <= maxiter; ++k) {
        const double rho_new = dot(n, r_hat, r);
        if (rho_new == 0.0) break;
        const double beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
            h[i] = inv[i] * p[i];  // phat
        }
        spmv(n, indptr, indices, data, h, v);
        const double denom = dot(n, r_hat, v);
        if (denom == 0.0) break;
        alpha = rho / denom;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        const double snorm = std::sqrt(dot(n, s, s));
        if (snorm / bnorm < tol) {
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
            for (int64_t i = 0; i < n; ++i) x[i] += alpha * h[i];
            *iters = k;
            *res = snorm / bnorm;
            return 1;
        }
        // keep phat for the update: x_new = x + alpha phat + omega shat; shat goes into t's partner below
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) {
            x[i] += alpha * h[i];   // provisional: undone on breakdown below
            h[i] = inv[i] * s[i];   // shat
        }
        spmv(n, indptr, indices, data, h, t);
        const double tt = dot(n, t, t);
        omega = tt > 0.0 ? dot(n, t, s) / tt : 0.0;
        double rr = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) {
            r[i] = s[i] - omega * t[i];
            rr += r[i] * r[i];
        }
        rnorm = std::sqrt(rr);
        if (!std::isfinite(rnorm)) {
            // breakdown: return the last finite iterate (undo the provisional alpha step)
            const double a = alpha;
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
            for (int64_t i = 0; i < n; ++i) x[i] -= a * inv[i] * p[i];
            *iters = k;
            *res = NAN;
            return 0;
        }
#pragma omp parallel for schedule(static) if (n > PAR_MIN)
        for (int64_t i = 0; i < n; ++i) x[i] += omega * h[i];
        if (rnorm / bnorm < tol || rnorm <= 1e-14 * b_abs) {
            *iters = k;
            *res = rnorm / bnorm;
            return 1;
        }
        if (omega == 0.0) break;
    }
    if (k > maxiter) k = maxiter;
    *iters = k;
    *res = rnorm / bnorm;
    return 0;
}


void ft_hllc_flux(int64_t nf, int64_t dim, double g, const double* WL, const double* WR, const double* n,
                  const double* A, double* out) {
    const int64_t nv = dim + 2;
#pragma omp parallel for schedule(static) if (nf > PAR_MIN)
    for (int64_t f = 0; f < nf; ++f) {
        const double* L = WL + f * nv;
        const double* R = WR + f * nv;
        const double* nn = n + f * dim;
        double* o = out + f * nv;
        const double rL = L[0], pL = L[nv - 1], rR = R[0], pR = R[nv - 1];
        double unL = 0, unR = 0, qL = 0, qR = 0;
        for (int64_t i = 0; i < dim; ++i) {
            unL += L[1 + i] * nn[i];
            unR += R[1 + i] * nn[i];
            qL += L[1 + i] * L[1 + i];
            qR += R[1 + i] * R[1 + i];
        }
        const double cL = std::sqrt(g * pL / rL), cR = std::sqrt(g * pR / rR);
        const double EL = pL / (g - 1.0) + 0.5 * rL * qL, ER = pR / (g - 1.0) + 0.5 * rR * qR;
        const double sl = std::sqrt(rL), sr = std::sqrt(rR), w = sl / (sl + sr);
        double un_roe = 0, q_roe = 0;
        for (int64_t i = 0; i < dim; ++i) {
            const double u = w * L[1 + i] + (1.0 - w) * R[1 + i];
            un_roe += u * nn[i];
            q_roe += u * u;
        }
        const double H_roe = w * (EL + pL) / rL + (1.0 - w) * (ER + pR) / rR;
        const double c_roe = std::sqrt(std::fmax((g - 1.0) * (H_roe - 0.5 * q_roe), 1e-30));
        const double SL = std::fmin(unL - cL, un_roe - c_roe);
        const double SR = std::fmax(unR + cR, un_roe + c_roe);
        double den = rL * (SL - unL) - rR * (SR - unR);
        if (den == 0.0) den = 1e-300;
        const double SM = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / den;
        // physical flux of the upwind side, plus S (Q* - Q) in the star regions
        const bool left = SM >= 0.0;
        const double* W = left ? L : R;
        const double r = left ? rL : rR, p = left ? pL : pR, un = left ? unL : unR, E = left ? EL : ER;
        const double S = left ? SL : SR;
        const bool star = left ? (SL < 0.0) : (SR > 0.0);
        o[0] = r * un;
        for (int64_t i = 0; i < dim; ++i) o[1 + i] = r * W[1 + i] * un + p * nn[i];
        o[nv - 1] = (E + p) * un;
        if (star) {
            double dS = S - SM;
            if (dS == 0.0) dS = 1e-300;
            double dU = S - un;
            if (dU == 0.0) dU = 1e-300;
            const double coef = r * (S - un) / dS;
            o[0] += S * (coef - r);
            for (int64_t i = 0; i < dim; ++i) o[1 + i] += S * (coef * (W[1 + i] + (SM - un) * nn[i]) - r * W[1 + i]);
            o[nv - 1] += S * (coef * (E / r + (SM - un) * (SM + p / (r * dU))) - E);
        }
        for (int64_t k = 0; k < nv; ++k) o[k] *= A[f];
    }
}
}  // extern "C"
