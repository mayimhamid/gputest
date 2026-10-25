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
        P = np.concatenate([mesh.owner[interior], mesh.owner[pa]])
        N = np.concatenate([mesh.neighbour[interior], mesh.owner[pb]])
        Pb = mesh.owner[boundary]
        self.n_if, self.n_bf = P.size, Pb.size
        xc = geo.cell_centre
        S = geo.face_area_vec[self.if_idx]
        Sb = geo.face_area_vec[boundary]
        xf = geo.face_centre[self.if_idx]
        xfb = geo.face_centre[boundary]
        offset = np.concatenate([np.zeros((int(interior.sum()), dim))] + shift) if shift else np.zeros((P.size, dim))
        xN = xc[N] - offset  # neighbour centre as seen from the owner
        if np.any(P == N):
            raise ValueError("a periodic face connects a cell to itself: use at least 2 cells across the period")
        self.offset = offset
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
        Gm = np.zeros((nc, dim, dim))
        outer = wl[:, None, None] * d[:, :, None] * d[:, None, :]
        outerb = wlb[:, None, None] * db[:, :, None] * db[:, None, :]
        for a in range(dim):
            for b in range(dim):
                Gm[:, a, b] = (np.bincount(P, outer[:, a, b], minlength=nc)
                               + np.bincount(N, outer[:, a, b], minlength=nc)
                               + np.bincount(Pb, outerb[:, a, b], minlength=nc))
        Ginv = np.linalg.pinv(Gm)

        # incidence matrices (cells x faces)
        ones_i = np.ones(P.size)
        self.Cown = sp.csr_matrix((ones_i, (P, np.arange(P.size))), shape=(nc, P.size))
        self.Cnei = sp.csr_matrix((ones_i, (N, np.arange(N.size))), shape=(nc, N.size))
        self.Cb = sp.csr_matrix((np.ones(Pb.size), (Pb, np.arange(Pb.size))), shape=(nc, Pb.size))
        self.Cdiff = (self.Cown - self.Cnei).tocsr()  # net outflow of face quantities

        # CSR sparsity pattern of cell-cell operators
        rows = np.concatenate([np.arange(nc), P, N])
        cols = np.concatenate([np.arange(nc), N, P])
        pat = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(nc, nc))
        pat.sum_duplicates()
        pat.sort_indices()
        row_of = np.repeat(np.arange(nc), np.diff(pat.indptr))
        keys = row_of.astype(np.int64) * nc + pat.indices
        self.nnz = pat.nnz
        self.slot_diag = np.searchsorted(keys, np.arange(nc, dtype=np.int64) * nc + np.arange(nc))
        self.slot_PN = np.searchsorted(keys, P.astype(np.int64) * nc + N)
        self.slot_NP = np.searchsorted(keys, N.astype(np.int64) * nc + P)
        self.indptr, self.indices = pat.indptr, pat.indices

        # boundary bookkeeping
        self.bzone = mesh.face_zone[boundary]
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
        self.xf = dev(xf)
        self.xfb = dev(xfb)
        self.wl, self.wlb = dev(wl), dev(wlb)
        self.Ginv = dev(Ginv)
        self.xN = dev(xN)
        self.dP_f = dev(xf - xc[P])  # owner centre -> face
        self.dN_f = dev(xf - xN)  # neighbour centre (through the periodic shift) -> face
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
        else:
            self._sparse = sp
            self.Cown_d = self.Cown.astype(self.dtype)
            self.Cnei_d = self.Cnei.astype(self.dtype)
            self.Cb_d = self.Cb.astype(self.dtype)
            self.Cdiff_d = self.Cdiff.astype(self.dtype)
            self.indptr_d, self.indices_d = self.indptr, self.indices

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
        xp = self.xp
        pf = self.face_interp(phi)
        if grad is not None:
            # correct the interpolated value from the point on the P-N line to the face centre
            xm = self.w[:, None] * self.xc[self.P] + (1.0 - self.w)[:, None] * self.xN
            pf = pf + xp.sum(self.face_interp_vec(grad) * (self.xf - xm), axis=1)
        s = self.Cdiff_d @ (pf[:, None] * self.S) + self.Cb_d @ (phi_b[:, None] * self.Sb)
        return s / self.vol[:, None]

    def face_interp(self, phi):
        return self.w * phi[self.P] + (1.0 - self.w) * phi[self.N]

    def face_interp_vec(self, g):
        return self.w[:, None] * g[self.P] + (1.0 - self.w)[:, None] * g[self.N]

    def matrix(self, diag, a_PN, a_NP):
        """CSR matrix with the given diagonal and off-diagonal face coefficients."""
        xp = self.xp
        data = xp.zeros(self.nnz, dtype=self.dtype)
        data[self.slot_diag_d] = diag
        data += xp.bincount(self.slot_PN_d, weights=a_PN, minlength=self.nnz).astype(self.dtype)
        data += xp.bincount(self.slot_NP_d, weights=a_NP, minlength=self.nnz).astype(self.dtype)
        return self._sparse.csr_matrix((data, self.indices_d, self.indptr_d), shape=(self.nc, self.nc))
