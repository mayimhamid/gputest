"""Device-resident finite-volume data for an unstructured mesh.

Everything a cell-centred FV solver needs is precomputed once on the host and
moved to the compute backend:

* interior faces (owner P, neighbour N) and boundary faces (owner only);
* area vectors, interpolation weights, the over-relaxed orthogonal
  coefficient ``alpha = S.S / S.d`` and the non-orthogonal remainder
  ``T = S - alpha d``;
* least-squares gradient operators (inverse-distance weighted);
* sparse incidence matrices used for all face->cell scatter-adds (cuSPARSE
  friendly, no atomics), and a fixed CSR sparsity pattern with direct
  data slots for every face coefficient, so assembling a matrix each
  iteration is a handful of vectorised operations.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..mesh.unstructured import UnstructuredMesh, compute_geometry


def _batched_inverse(G: np.ndarray) -> np.ndarray:
    """Inverse of each (dim x dim) matrix; the pseudo-inverse only where one is singular.

    ``np.linalg.inv`` is several times faster than the SVD behind ``pinv`` and gives the
    same result for the non-singular least-squares matrices of an ordinary mesh."""
    out = np.empty_like(G)
    cond_ok = np.abs(np.linalg.det(G)) > 1e-300
    if cond_ok.any():
        out[cond_ok] = np.linalg.inv(G[cond_ok])
        bad = ~cond_ok | ~np.isfinite(out).all(axis=(1, 2))
    else:
        bad = cond_ok == cond_ok
    if bad.any():
        out[bad] = np.linalg.pinv(G[bad])
    return out


class _LeanGeometry:
    """Cell geometry kept in memory; the per-face arrays (7 doubles per face, 0.6 GB on a
    10.9M-face mesh) are recomputed on access, which only wall-distance set-up does."""

    def __init__(self, mesh, geo):
        self._mesh = mesh
        self.cell_centre, self.cell_volume, self.flipped_faces = geo.cell_centre, geo.cell_volume, geo.flipped_faces

    @property
    def total_volume(self) -> float:
        return float(self.cell_volume.sum())

    def __getattr__(self, name):
        if name in ("face_centre", "face_area_vec", "face_area"):
            return getattr(compute_geometry(self._mesh), name)
        raise AttributeError(name)


class FVMesh:
    def __init__(self, mesh: UnstructuredMesh, backend, dtype=np.float64):
        self.mesh = mesh
        self.be = backend
        xp = self.xp = backend.xp
        self.dtype = np.dtype(dtype)
        self.dim = dim = mesh.dim
        geo = self.geometry = compute_geometry(mesh)
        nc = self.nc = mesh.n_cells
        interior = mesh.neighbour >= 0
        boundary = ~interior
        # translational periodic pairs become interior faces: owner of the face on
        # side a, "neighbour" the owner of its image on side b, seen through the
        # translation (its centre shifted by -T). Faces on side b are dropped.
        pa = [np.asarray(p["faces_a"], dtype=np.int64) for p in mesh.periodic]
        pb = [np.asarray(p["faces_b"], dtype=np.int64) for p in mesh.periodic]
        shift = [np.repeat(np.asarray(p["translation"], dtype=float)[None, :dim], len(p["faces_a"]), axis=0)
                 for p in mesh.periodic]
        pa = np.concatenate(pa) if pa else np.zeros(0, np.int64)
        pb = np.concatenate(pb) if pb else np.zeros(0, np.int64)
        boundary[pa] = False
        boundary[pb] = False
        self.if_idx = np.concatenate([np.nonzero(interior)[0], pa])
        self.bf_idx = np.nonzero(boundary)[0]
        self.n_periodic = pa.size
        # interior-face ranges of every periodic pair (for periodic mass-flow reports)
        self.periodic, start = [], int(interior.sum())
        for p in mesh.periodic:
            n = len(p["faces_a"])
            za, zb = mesh.zones[p["zone_a"]].name, mesh.zones[p["zone_b"]].name
            self.periodic.append({"zones": (za, zb), "translation": np.asarray(p["translation"], dtype=float),
                                  "faces": np.arange(start, start + n)})
            start += n
        P = np.concatenate([mesh.owner[interior], mesh.owner[pa]])
        N = np.concatenate([mesh.neighbour[interior], mesh.owner[pb]])
        Pb = mesh.owner[boundary]
        self.n_if, self.n_bf = P.size, Pb.size
        xc = geo.cell_centre
        S = geo.face_area_vec[self.if_idx]
        Sb = geo.face_area_vec[boundary]
        xf = geo.face_centre[self.if_idx]
        xfb = geo.face_centre[boundary]
        # neighbour centre as seen from the owner (shifted only on periodic faces)
        xN = xc[N]
        if shift:
            xN[int(interior.sum()):] -= np.concatenate(shift)
        del shift
        if np.any(P == N):
            raise ValueError("a periodic face connects a cell to itself: use at least 2 cells across the period")
        d = xN - xc[P]
        db = xfb - xc[Pb]
        dd = np.einsum("fd,fd->f", d, d)
        w = np.clip(np.einsum("fd,fd->f", xN - xf, d) / dd, 0.0, 1.0)  # owner weight
        Sd = np.einsum("fd,fd->f", S, d)
        Sdb = np.einsum("fd,fd->f", Sb, db)
        SS = np.einsum("fd,fd->f", S, S)
        SSb = np.einsum("fd,fd->f", Sb, Sb)
        alpha = SS / Sd
        alpha_b = SSb / Sdb
        Tvec = S - alpha[:, None] * d
        self.non_orthogonality_deg = float(np.degrees(np.arccos(np.clip(
            Sd / np.sqrt(SS * dd), -1, 1))).max()) if P.size else 0.0

        # least-squares gradient matrices
        wl = 1.0 / dd
        wlb = 1.0 / np.einsum("fd,fd->f", db, db)
        # assembled one symmetric component at a time: no (faces, dim, dim) temporaries
        Gm = np.zeros((nc, dim, dim))
        for a in range(dim):
            for b in range(a, dim):
                o = wl * d[:, a] * d[:, b]
                Gm[:, a, b] = (np.bincount(P, o, minlength=nc) + np.bincount(N, o, minlength=nc)
                               + np.bincount(Pb, wlb * db[:, a] * db[:, b], minlength=nc))
                Gm[:, b, a] = Gm[:, a, b]
        del o
        Ginv = _batched_inverse(Gm)
        del Gm

        # incidence matrices (cells x faces)
        # host copies hold +-1 as int8 (1 byte per entry); compute copies are made below
        ones_i = np.ones(P.size, dtype=np.int8)
        self.Cown = sp.csr_matrix((ones_i, (P, np.arange(P.size))), shape=(nc, P.size))
        self.Cnei = sp.csr_matrix((ones_i, (N, np.arange(N.size))), shape=(nc, N.size))
        self.Cb = sp.csr_matrix((np.ones(Pb.size, dtype=np.int8), (Pb, np.arange(Pb.size))), shape=(nc, Pb.size))
        del ones_i
        self.Cdiff = (self.Cown - self.Cnei).tocsr()  # net outflow of face quantities

        # CSR sparsity pattern of cell-cell operators
        rows = np.concatenate([np.arange(nc), P, N])
        cols = np.concatenate([np.arange(nc), N, P])
        pat = sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(nc, nc))
        del rows, cols
        pat.sum_duplicates()
        pat.sort_indices()
        row_of = np.repeat(np.arange(nc), np.diff(pat.indptr))
        keys = row_of.astype(np.int64) * nc + pat.indices
        self.nnz = pat.nnz
        islot = np.int32 if pat.nnz < 2 ** 31 else np.int64  # half the bytes of searchsorted's int64
        self.slot_diag = np.searchsorted(keys, np.arange(nc, dtype=np.int64) * nc + np.arange(nc)).astype(islot)
        self.slot_PN = np.searchsorted(keys, P.astype(np.int64) * nc + N).astype(islot)
        self.slot_NP = np.searchsorted(keys, N.astype(np.int64) * nc + P).astype(islot)
        del keys, row_of
        self.indptr, self.indices = pat.indptr, pat.indices

        # boundary bookkeeping
        self.bzone = mesh.face_zone[boundary]
        self.Pb_h = Pb
        # wall-normal distance from the owner centre to each boundary face (y for y+, wall shear)
        self.yb_h = np.abs(np.einsum("fd,fd->f", db, Sb)) / np.sqrt(np.einsum("fd,fd->f", Sb, Sb))
        dev = lambda a, dt=self.dtype: xp.asarray(a, dtype=dt)  # noqa: E731
        self.P, self.N, self.Pb = xp.asarray(P), xp.asarray(N), xp.asarray(Pb)
        self.S, self.Sb = dev(S), dev(Sb)
        self.d, self.db = dev(d), dev(db)
        self.w = dev(w)
        self.alpha, self.alpha_b = dev(alpha), dev(alpha_b)
        self.Tvec = dev(Tvec)
        self.Sb_mag = dev(np.sqrt(SSb))
        nb_h = Sb / np.sqrt(SSb)[:, None]
        self.nb = dev(nb_h)
        # tangential part of (boundary face centre - owner centre): zero-normal-gradient
        # boundary values are phi_P + grad(phi)_P . db_t (exact for linear phi)
        self.db_t = dev(db - np.einsum("fd,fd->f", db, nb_h)[:, None] * nb_h)
        self.vol = dev(geo.cell_volume)
        self.xc = dev(xc)
        self.xfb = dev(xfb)
        self.yb = dev(self.yb_h)
        self.wl, self.wlb = dev(wl), dev(wlb)
        self.Ginv = dev(Ginv)
        self.dP_f = dev(xf - xc[P])  # owner centre -> face
        self.dN_f = dev(xf - xN)  # neighbour centre (through the periodic shift) -> face
        # face centre minus its point on the P-N line (skewness correction of face values)
        dxm = xf - (w[:, None] * xc[P] + (1.0 - w)[:, None] * xN)
        del xf, xN, Tvec, d, S
        self.geometry = geo = _LeanGeometry(mesh, geo)
        self.dxm = dev(dxm)
        self.slot_diag_d = xp.asarray(self.slot_diag)
        self.slot_PN_d = xp.asarray(self.slot_PN)
        self.slot_NP_d = xp.asarray(self.slot_NP)
        if backend.is_gpu:
            import cupyx.scipy.sparse as csp  # type: ignore

            self._sparse = csp
            self.Cown_d = csp.csr_matrix(self.Cown.astype(self.dtype))
            self.Cnei_d = csp.csr_matrix(self.Cnei.astype(self.dtype))
            self.Cb_d = csp.csr_matrix(self.Cb.astype(self.dtype))
            self.Cdiff_d = csp.csr_matrix(self.Cdiff.astype(self.dtype))
            self.indptr_d = xp.asarray(self.indptr)
            self.indices_d = xp.asarray(self.indices)
        self.nat = None  # C++/OpenMP kernels (CPU, FP64), see flowtrace.native
        self.cu = None  # CUDA kernels (GPU, FP64)
        native = None
        if not backend.is_gpu and self.dtype == np.float64:
            from .. import native

            native = native if native.load() is not None else None
        if native is not None:
            self._sparse = sp
            self.indptr_d, self.indices_d = self.indptr, self.indices
            self._init_native(native, self.dxm)
        elif not backend.is_gpu:
            self._sparse = sp
            self.Cown_d = self.Cown.astype(self.dtype)
            self.Cnei_d = self.Cnei.astype(self.dtype)
            self.Cb_d = self.Cb.astype(self.dtype)
            self.Cdiff_d = self.Cdiff.astype(self.dtype)
            self.indptr_d, self.indices_d = self.indptr, self.indices
        elif backend.is_gpu and self.dtype == np.float64:
            from ..native import cuda as native_cuda

            if native_cuda.available():
                self.cu = native_cuda.FVKernels(self, self.dxm)

    def _init_native(self, native, dxm):
        self.nat, self._lib = native, native.load()
        self.Cown_d, self.Cnei_d = native.NativeCSR(self.Cown), native.NativeCSR(self.Cnei)
        self.Cb_d, self.Cdiff_d = native.NativeCSR(self.Cb), native.NativeCSR(self.Cdiff)
        # the native wrappers forward every SciPy attribute: one copy of each incidence matrix
        self.Cown, self.Cnei, self.Cb, self.Cdiff = self.Cown_d, self.Cnei_d, self.Cb_d, self.Cdiff_d
        f64, i32, i64 = native.f64, native.i32, native.i64
        cd, cb = self.Cdiff_d, self.Cb_d
        self._n = {
            "iptr": cd.indptr, "iidx": cd.indices, "isgn": cd.data, "bptr": cb.indptr, "bidx": cb.indices,
            "P": i64(self.P), "N": i64(self.N), "w": f64(self.w), "wl": f64(self.wl), "d": f64(self.d),
            "wlb": f64(self.wlb), "db": f64(self.db), "Ginv": f64(self.Ginv), "S": f64(self.S), "Sb": f64(self.Sb),
            "vol": f64(self.vol), "dxm": f64(dxm), "slot_diag": i32(self.slot_diag), "slot_PN": i32(self.slot_PN),
            "slot_NP": i32(self.slot_NP), "indptr": i32(self.indptr), "indices": i32(self.indices),
        }

    @property
    def xf(self):
        """Interior face centres, rebuilt on demand (owner centre + owner->face vector)."""
        return self.xc[self.P] + self.dP_f

    # ------------------------------------------------------------------
    def scatter(self, f_int=None, f_nei=None, f_bnd=None):
        """Sum face values into cells: owner gets f_int, neighbour gets f_nei, boundary owner f_bnd."""
        out = None
        for C, v in ((self.Cown_d, f_int), (self.Cnei_d, f_nei), (self.Cb_d, f_bnd)):
            if v is None:
                continue
            t = C @ v
            out = t if out is None else out + t
        return out

    def net_outflow(self, F, Fb):
        """Sum over a cell's faces of outward fluxes (interior F is owner->neighbour)."""
        return self.Cdiff_d @ F + self.Cb_d @ Fb

    def gradient(self, phi, phi_b):
        """Least-squares cell gradient (nc, dim) of a scalar with boundary face values."""
        if self.nat is not None:
            n, p = self._n, self.nat._p
            phi, phi_b = self.nat.f64(phi), self.nat.f64(phi_b)
            out = np.empty((self.nc, self.dim))
            self._lib.ft_ls_gradient(self.nc, self.dim, p(n["iptr"]), p(n["iidx"]), p(n["bptr"]), p(n["bidx"]),
                                         p(n["P"]), p(n["N"]), p(n["wl"]), p(n["d"]), p(n["wlb"]), p(n["db"]),
                                         p(n["Ginv"]), p(phi), p(phi_b), p(out))
            return out
        if self.cu is not None:
            return self.cu.ls_gradient(phi, phi_b)
        xp = self.xp
        dphi = phi[self.N] - phi[self.P]
        dphib = phi_b - phi[self.Pb]
        r = (self.Cown_d @ ((self.wl * dphi)[:, None] * self.d)
             + self.Cnei_d @ ((self.wl * dphi)[:, None] * self.d)
             + self.Cb_d @ ((self.wlb * dphib)[:, None] * self.db))
        return xp.einsum("cab,cb->ca", self.Ginv, r)

    def gradient_gg(self, phi, phi_b, grad=None):
        """Green-Gauss cell gradient (1/V) sum_f phi_f S_f with linearly interpolated
        face values (plus a skewness correction from ``grad`` when given)."""
        if self.nat is not None:
            n, p, f64 = self._n, self.nat._p, self.nat.f64
            phi, phi_b = f64(phi), f64(phi_b)
            g = f64(grad) if grad is not None else None
            out = np.empty((self.nc, self.dim))
            self._lib.ft_gg_gradient(self.nc, self.dim, p(n["iptr"]), p(n["iidx"]), p(n["isgn"]), p(n["bptr"]),
                                         p(n["bidx"]), p(n["P"]), p(n["N"]), p(n["w"]), p(n["S"]), p(n["Sb"]),
                                         p(n["vol"]), p(n["dxm"]), p(phi), p(phi_b), p(g) if g is not None else None,
                                         p(out))
            return out
        if self.cu is not None:
            return self.cu.gg_gradient(phi, phi_b, grad)
        xp = self.xp
        pf = self.face_interp(phi)
        if grad is not None:
            # correct the interpolated value from the point on the P-N line to the face centre
            pf = pf + xp.sum(self.face_interp_vec(grad) * self.dxm, axis=1)
        s = self.Cdiff_d @ (pf[:, None] * self.S) + self.Cb_d @ (phi_b[:, None] * self.Sb)
        return s / self.vol[:, None]

    def _interp_native(self, a, k):
        n, p = self._n, self.nat._p
        a = self.nat.f64(a)
        out = np.empty((self.n_if,) if k == 1 else (self.n_if, k))
        self._lib.ft_face_interp(self.n_if, k, p(n["w"]), p(n["P"]), p(n["N"]), p(a), p(out))
        return out

    def face_interp(self, phi):
        if self.nat is not None and phi.dtype == np.float64:
            return self._interp_native(phi, 1)
        if self.cu is not None:
            return self.cu.face_interp(phi, 1)
        return self.w * phi[self.P] + (1.0 - self.w) * phi[self.N]

    def face_interp_vec(self, g):
        if self.nat is not None and g.dtype == np.float64:
            return self._interp_native(g, g.shape[1])
        if self.cu is not None:
            return self.cu.face_interp(g, g.shape[1])
        return self.w[:, None] * g[self.P] + (1.0 - self.w)[:, None] * g[self.N]

    def matrix(self, diag, a_PN, a_NP):
        """CSR matrix with the given diagonal and off-diagonal face coefficients."""
        if self.nat is not None:
            n, p, f64 = self._n, self.nat._p, self.nat.f64
            diag, a_PN, a_NP = f64(diag), f64(a_PN), f64(a_NP)
            data = np.empty(self.nnz)
            self._lib.ft_assemble(self.nnz, self.nc, self.n_if, p(n["slot_diag"]), p(n["slot_PN"]),
                                      p(n["slot_NP"]), p(diag), p(a_PN), p(a_NP), p(data))
            return self.nat.csr_from_parts(data, n["indices"], n["indptr"], (self.nc, self.nc))
        if self.cu is not None:
            return self.cu.matrix(diag, a_PN, a_NP)
        xp = self.xp
        data = xp.zeros(self.nnz, dtype=self.dtype)
        data[self.slot_diag_d] = diag
        data += xp.bincount(self.slot_PN_d, weights=a_PN, minlength=self.nnz).astype(self.dtype)
        data += xp.bincount(self.slot_NP_d, weights=a_NP, minlength=self.nnz).astype(self.dtype)
        return self._sparse.csr_matrix((data, self.indices_d, self.indptr_d), shape=(self.nc, self.nc))

    def matrix_like(self, data):
        """Cell-cell operator with the fixed sparsity pattern of :meth:`matrix` and the given
        CSR data array (for instance ``|A.data|`` of a matrix that :meth:`matrix` returned)."""
        if self.nat is not None:
            n = self._n
            return self.nat.csr_from_parts(data, n["indices"], n["indptr"], (self.nc, self.nc))
        return self._sparse.csr_matrix((data, self.indices_d, self.indptr_d), shape=(self.nc, self.nc))

    def wall_distance(self, wall_bfaces_mask) -> np.ndarray:
        """Cell-centre distance to the nearest wall (host array); mask over boundary faces."""
        from ..mesh.unstructured import wall_distance

        faces = self.bf_idx[np.asarray(wall_bfaces_mask, dtype=bool)]
        return wall_distance(self.geometry.cell_centre, self.mesh, faces)
