/* FlowTrace native finite-volume kernels: C API (C11 / C++ callers, ctypes from Python).
 *
 * All floating-point arrays are double precision, C-contiguous, row-major.
 * Face/cell index arrays (P, N, Pb) are int64; CSR index arrays are int32.
 * The "cell adjacency" of interior faces is the cells x faces CSR matrix
 * Cdiff (+1 where the cell owns the face, -1 where it is the neighbour);
 * that of boundary faces is Cb (+1 at the owner).
 *
 * Every function is thread-parallel with OpenMP over independent rows (cells or
 * faces), with no atomics; results do not depend on the thread count.
 */
#ifndef FLOWTRACE_FVM_KERNELS_H
#define FLOWTRACE_FVM_KERNELS_H

#include <stdint.h>

#ifdef _WIN32
#define FT_API __declspec(dllexport)
#else
#define FT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

FT_API int ft_abi_version(void);
FT_API int ft_max_threads(void);

/* Y[nrows, k] = A[nrows, ncols] @ X[ncols, k]   (CSR, k right-hand sides) */
FT_API void ft_csr_spmm(int64_t nrows, int64_t k, const int32_t* indptr, const int32_t* indices,
                        const double* data, const double* X, double* Y);

/* out[f, :] = w[f] * phi[P[f], :] + (1 - w[f]) * phi[N[f], :]   (k components) */
FT_API void ft_face_interp(int64_t nf, int64_t k, const double* w, const int64_t* P, const int64_t* N,
                           const double* phi, double* out);

/* Least-squares cell gradient (inverse-distance weighted), out[c, :dim]. */
FT_API void ft_ls_gradient(int64_t nc, int64_t dim,
                           const int32_t* iptr, const int32_t* iidx,  /* Cdiff adjacency */
                           const int32_t* bptr, const int32_t* bidx,  /* Cb adjacency */
                           const int64_t* P, const int64_t* N,
                           const double* wl, const double* d, const double* wlb, const double* db,
                           const double* Ginv, const double* phi, const double* phib, double* out);

/* Green-Gauss gradient (1/V) sum_f phi_f S_f with linear face values, plus the
 * skewness correction grad_f . dxm_f when grad != NULL (dxm = x_f - x_m). */
FT_API void ft_gg_gradient(int64_t nc, int64_t dim,
                           const int32_t* iptr, const int32_t* iidx, const double* isgn,
                           const int32_t* bptr, const int32_t* bidx,
                           const int64_t* P, const int64_t* N, const double* w,
                           const double* S, const double* Sb, const double* vol, const double* dxm,
                           const double* phi, const double* phib, const double* grad, double* out);

/* CSR data of a cell-cell operator from its diagonal and face coefficients. */
FT_API void ft_assemble(int64_t nnz, int64_t nc, int64_t nf, const int32_t* slot_diag, const int32_t* slot_PN,
                        const int32_t* slot_NP, const double* diag, const double* aPN, const double* aNP,
                        double* data);

/* Jacobi-preconditioned BiCGSTAB, same algorithm and stopping rules as
 * flowtrace.linalg.krylov.bicgstab. x holds x0 on entry and the solution on
 * exit. Returns 1 if converged; *iters and *res receive the iteration count
 * and the final relative residual (NaN on breakdown). */
FT_API int ft_bicgstab_jacobi(int64_t n, const int32_t* indptr, const int32_t* indices, const double* data,
                              const double* b, const double* inv_diag, double* x, double tol, int64_t maxiter,
                              int relative_to_initial, double* work, int64_t* iters, double* res);

/* HLLC flux of the compressible Euler equations (Toro, Einfeldt/Roe wave speeds) on nf faces.
 * WL, WR: primitive states (rho, u[dim], p), rows of dim+2; n: unit normals (nf, dim);
 * A: face areas (may be radius-weighted). out[f, :] = F_hllc(WL, WR, n) * A[f]. */
FT_API void ft_hllc_flux(int64_t nf, int64_t dim, double gamma, const double* WL, const double* WR,
                         const double* n, const double* A, double* out);

#ifdef __cplusplus
}
#endif

#endif
