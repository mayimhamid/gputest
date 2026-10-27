r"""Algebraic, conservation and health diagnostics of one SIMPLE/SIMPLEC outer iteration.

:class:`~flowtrace.unstructured.simple.SimpleSolver` keeps references to the linear
systems it assembled and solved in its most recent iteration (no copies: the solver
rebinds, and never mutates, its field and flux arrays). :func:`evaluate` computes the
metrics below from those systems on demand, so the iteration itself does no extra work.
Every metric names the exact system and state it is evaluated on.

Evaluation points within iteration n (``SimpleSolver.step``)
-------------------------------------------------------------
1. Momentum is assembled at the iteration-start state ``(U^n, p^n, F^n)``:
   ``A`` (implicit upwind convection + orthogonal diffusion + boundary and transient
   diagonal terms) and one right-hand side ``b_k`` per velocity component. ``b_k``
   already contains the pressure force ``-(grad p)_P V_P``, body forces, sources,
   boundary terms and the explicit non-orthogonal and deferred-correction terms, so
   no force is counted twice below.
2. The system passed to the linear solver is the under-relaxed one,
   ``A_r = A`` with diagonal ``a_P / alpha_u`` and ``b_r,k = b_k + (1 - alpha_u)/alpha_u a_P u_k^n``.
   Algebraically ``b_r - A_r u^n = b - A u^n``: both systems have the same residual at
   the old velocity, but not the same row scales.
3. Jacobi-BiCGSTAB returns ``u*_k``. Its reported residual is the *recurrence*
   ``||r_k||_2 / ||r_0||_2`` (right preconditioning, so unpreconditioned in exact
   arithmetic), relative to the initial residual at ``u^n``.
4. Rhie-Chow predictor fluxes ``F*`` (interior, owner -> neighbour) and ``Fb*``
   (boundary, outward) give the predictor-state cell imbalance ``m*``.
5. The pressure correction solves ``A_p p' = -m*`` (AMG-PCG, relative tolerance
   ``case.pressure_tol``, residual relative to ``||m*||_2``).
6. Corrected fluxes ``F^{n+1} = F* + c_f (p'_P - p'_N)``, ``Fb^{n+1} = Fb* + c_b p'_P``.
   Their cell imbalance is exactly ``m* + A_p p'``, i.e. minus the pressure-correction
   residual: it is reduced by the pressure tolerance per iteration, not to zero.
7. ``U^{n+1} = U* - d grad p'`` and ``p^{n+1} = p^n + alpha_p p'``.
8. Energy, when this solver owns it: the same pattern as momentum with ``T``.

Componentwise relative residual (row-scaled)
--------------------------------------------
For a system ``A x = b`` evaluated at ``x``::

    r_i = b_i - sum_j A_ij x_j          s_i = |b_i| + sum_j |A_ij| |x_j|
    R_row = max over active rows i of |r_i| / s_i

``R_row`` is the Oettli-Prager componentwise backward error: the smallest ``w`` such
that ``x`` solves ``(A + dA) x = b + db`` exactly with ``|dA| <= w |A|`` and
``|db| <= w |b|`` entrywise (Higham 2002, Thm. 7.3). It is dimensionless and lies in
[0, 1]. It measures how well the *discrete system* is satisfied, not the error of the
solution against the continuous problem, and it is not mesh-independent.

``|A|`` is formed from the assembled CSR entries, so faces that share a cell pair
are summed before the absolute value is taken, as in the solved matrix. Every cell row
of these systems is a finite-volume balance: boundary conditions enter through the
diagonal and the right-hand side, so there are no identity rows to skip.

Small-scale rows. A row is excluded when ``s_i <= gamma_K * S_ref``, where ``S_ref`` is
the largest finite ``s_i`` over the whole equation set (all velocity components share
one reference, so a component that is identically zero, such as ``w`` in a 2-D flow on a
3-D mesh, is not measured against its own round-off), ``u`` is the unit round-off of the
working precision, ``K`` the largest number of terms in one row (nonzeros + 1) and
``gamma_K = K u / (1 - K u)`` the standard bound on the rounding error of a K-term sum
(Higham 2002, Sec. 3.1). Such a row's whole magnitude is below the rounding error of
evaluating the system's largest rows, so its ratio is not resolvable in this precision.
No dimensional constant is involved. Excluded rows are counted, and their largest
absolute residual is reported separately. With ``S_ref = 0`` every term is exactly zero,
the residual is exactly zero and ``R_row = 0``.

Non-finite values. Rows with a non-finite ``r_i`` or ``s_i`` are counted
(``nonfinite_rows``); ``R_row`` is then ``inf`` so that any threshold test fails.

Local continuity
----------------
Cell imbalance ``m_i = sum_f F_f`` over the cell's faces with outward sign (kg/s, per
unit depth in 2-D; positive = net outflow). Its scale is the cell throughflow
``t_i = sum_f |F_f|``; ``|m_i| / t_i`` is the componentwise backward error of the cell's
mass balance (the smallest relative change of its face fluxes that balances it). The
same exclusion rule applies with ``K`` = the largest number of faces of a cell.
Translational periodic faces are interior faces here, so they enter the local balance
and cancel from the global sum.

Signed global boundary imbalance ``G = sum_b Fb`` (kg/s, positive = net outflow) with
scale ``sum_b |Fb|``. ``G = sum_i m_i`` exactly, so ``G = 0`` is necessary but *not*
sufficient for local conservation. With no boundary throughflow (closed domain) the
relative value is 0 when ``G = 0`` and ``inf`` otherwise.

Outer field change
------------------
``dU = max_i |U^{n+1}_i - U^n_i| / U_ref`` with ``U_ref`` the largest velocity magnitude
in ``U^n``, ``U^{n+1}`` or on velocity boundaries; ``dp = max_i |p^{n+1}_i - p^n_i| / p_ref``
with ``p_ref = max(range(p^{n+1}), rho U_ref^2)``; ``dT`` relative to the range of ``T^{n+1}``
and the fixed boundary temperatures; ``dk``, ``domega/depsilon`` relative to the
field's maximum. ``0/0`` is reported as 0 and ``x/0`` (x > 0) as ``inf`` (Higham's
convention). These measure the change per outer iteration, which under-relaxation
keeps small even far from convergence; they are not error estimates.

All reductions are collected on the device and moved to the host in one transfer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DIAGNOSTICS_VERSION = 1

COMPONENTS = ("x-velocity", "y-velocity", "z-velocity")


def unit_roundoff(dtype) -> float:
    """u = eps / 2 of the working precision (2^-53 in FP64, 2^-24 in FP32)."""
    return float(np.finfo(np.dtype(dtype)).eps) / 2.0


def gamma(k: int, u: float) -> float:
    """Higham's gamma_k = k u / (1 - k u): bound on the relative rounding error of a k-term sum."""
    ku = k * u
    return ku / (1.0 - ku) if ku < 1.0 else math.inf


def ratio(num: float, den: float) -> float:
    """num / den with 0/0 = 0 and x/0 = inf (x > 0); non-finite inputs give inf."""
    if not (math.isfinite(num) and math.isfinite(den)):
        return math.inf
    if den > 0.0:
        return num / den
    return 0.0 if num == 0.0 else math.inf


class _Scalars:
    """Device scalars collected under names and moved to the host in one transfer."""

    def __init__(self, xp, to_numpy) -> None:
        self.xp, self.to_numpy = xp, to_numpy
        self.names: List[str] = []
        self.vals: list = []

    def add(self, name: str, v) -> None:
        self.names.append(name)
        self.vals.append(self.xp.asarray(v, dtype=self.xp.float64).reshape(()))

    def fetch(self) -> Dict[str, float]:
        if not self.vals:
            return {}
        host = np.asarray(self.to_numpy(self.xp.stack(self.vals)), dtype=np.float64)
        return {n: float(v) for n, v in zip(self.names, host)}


def _rows_of_system(fv, A, b, x):
    """(r, s) per row: r = b - A x, s = |b| + |A| |x| with |A| from the assembled entries."""
    xp = fv.xp
    r = b - A @ x
    s = xp.abs(b) + fv.matrix_like(xp.abs(A.data)) @ xp.abs(x)
    return r, s


def _row_set(sc: _Scalars, xp, prefix: str, pairs: Sequence[Tuple[str, object, object]], K: int, u: float) -> None:
    """Componentwise metrics of several (r, s) row sets that share one reference scale."""
    finite_s = [xp.where(xp.isfinite(s), s, 0.0) for _, _, s in pairs]
    s_ref = finite_s[0].max()
    for fs in finite_s[1:]:
        s_ref = xp.maximum(s_ref, fs.max())
    thr = gamma(K, u) * s_ref
    sc.add(f"{prefix}|s_ref", s_ref)
    for name, r, s in pairs:
        ar = xp.abs(r)
        finite = xp.isfinite(r) & xp.isfinite(s)
        active = finite & (s > thr)
        excluded = finite & ~active
        q = xp.where(active, ar / xp.where(active, s, 1.0), 0.0)
        key = f"{prefix}|{name}"
        sc.add(key + "|row_scaled", q.max())
        sc.add(key + "|worst_row", xp.argmax(q))
        sc.add(key + "|active_rows", active.sum())
        sc.add(key + "|excluded_rows", excluded.sum())
        sc.add(key + "|excluded_abs_max", xp.where(excluded, ar, 0.0).max())
        sc.add(key + "|nonfinite_rows", (~finite).sum())
        fin_r = xp.where(finite, ar, 0.0)
        sc.add(key + "|abs_l1", fin_r.sum())
        sc.add(key + "|abs_max", fin_r.max())
        sc.add(key + "|l2", xp.sqrt((fin_r * fin_r).sum()))


def _row_report(vals: Dict[str, float], prefix: str, name: str, centres: np.ndarray) -> dict:
    key = f"{prefix}|{name}"
    nonfinite = int(vals[key + "|nonfinite_rows"])
    worst = int(vals[key + "|worst_row"])
    out: dict = {
        "row_scaled": math.inf if nonfinite else vals[key + "|row_scaled"],
        "abs_l1": vals[key + "|abs_l1"], "abs_max": vals[key + "|abs_max"], "abs_l2": vals[key + "|l2"],
        "active_rows": int(vals[key + "|active_rows"]), "excluded_rows": int(vals[key + "|excluded_rows"]),
        "excluded_abs_max": vals[key + "|excluded_abs_max"], "nonfinite_rows": nonfinite,
    }
    if out["active_rows"] and not nonfinite:
        out["worst_cell"] = worst
        out["worst_cell_centre"] = [float(v) for v in centres[worst]]
    return out


def _linear_info(info) -> Optional[dict]:
    if info is None:
        return None
    res = float(info.residual)
    return {"iterations": int(info.iterations), "reported_residual": res if math.isfinite(res) else None,
            "converged": bool(info.converged)}


def _max_faces_per_cell(fv) -> int:
    k = getattr(fv, "_diag_max_faces", None)
    if k is None:
        per_cell = sum(np.diff(C.tocsr().indptr) for C in (fv.Cown, fv.Cnei, fv.Cb))
        k = int(np.max(per_cell)) if fv.nc else 1
        fv._diag_max_faces = k
    return k


def evaluate(solver) -> dict:
    """Diagnostics of the solver's most recent outer iteration (see the module docstring)."""
    cap = getattr(solver, "_capture", None)
    if cap is None:
        return {"available": False, "version": DIAGNOSTICS_VERSION,
                "reason": "no outer iteration has been completed since the solver was created or restored"}
    fv, xp, be = solver.fv, solver.xp, solver.be
    u = unit_roundoff(solver.dtype)
    K_rows = int(np.max(np.diff(fv.indptr))) + 1 if fv.nc else 1
    K_cont = _max_faces_per_cell(fv)
    sc = _Scalars(xp, be.to_numpy)
    dim = solver.dim
    names = COMPONENTS[:dim]

    # ---- momentum: pre-solve (unrelaxed system at U^n) and solved (relaxed system at U*) ----
    U_old, U_star = cap["U_old"], cap["U_star"]
    pre = [(names[k],) + _rows_of_system(fv, cap["A"], cap["b"][k], U_old[:, k]) for k in range(dim)]
    sol = [(names[k],) + _rows_of_system(fv, cap["Ar"], cap["br"][k], U_star[:, k]) for k in range(dim)]
    _row_set(sc, xp, "mom_pre", pre, K_rows, u)
    _row_set(sc, xp, "mom_sol", sol, K_rows, u)
    for k in range(dim):
        # true residual reduction of the linear solve: ||b_r - A_r u*||_2 / ||b_r - A_r u^n||_2
        r0 = cap["br"][k] - cap["Ar"] @ U_old[:, k]
        sc.add(f"mom_true_r0|{names[k]}", xp.sqrt(xp.sum(r0 * r0)))

    # ---- energy (only when this solver owns the equation) ----
    ecap = cap.get("energy")
    if ecap is not None:
        _row_set(sc, xp, "T_pre", [("energy",) + _rows_of_system(fv, ecap["A"], ecap["b"], ecap["T_old"])], K_rows, u)
        _row_set(sc, xp, "T_sol", [("energy",) + _rows_of_system(fv, ecap["Ar"], ecap["br"], ecap["T_new"])],
                 K_rows, u)
        r0 = ecap["br"] - ecap["Ar"] @ ecap["T_old"]
        sc.add("T_true_r0", xp.sqrt(xp.sum(r0 * r0)))

    # ---- pressure correction A_p p' = -m* ----
    _row_set(sc, xp, "pcorr", [("p-correction",) + _rows_of_system(fv, cap["Ap"], -cap["mass"], cap["pp"])],
             K_rows, u)

    # ---- local continuity: predictor (F*, Fb*) and corrected (F^{n+1}, Fb^{n+1}) ----
    for tag, F, Fb in (("pred", cap["Fst"], cap["Fbst"]), ("corr", cap["F"], cap["Fb"])):
        m = fv.net_outflow(F, Fb)
        aF, aFb = xp.abs(F), xp.abs(Fb)
        t = fv.Cown_d @ aF + fv.Cnei_d @ aF + fv.Cb_d @ aFb
        _row_set(sc, xp, "cont_" + tag, [("cells", m, t)], K_cont, u)
        sc.add(f"cont_{tag}|face_flux_l1", aF.sum() + aFb.sum())
        sc.add(f"cont_{tag}|signed_sum", m.sum())
    Fb = cap["Fb"]
    sc.add("global|signed", Fb.sum())
    sc.add("global|throughflow", xp.abs(Fb).sum())

    # ---- outer field change and numerical health ----
    U_new, p_old, p_new = cap["U_new"], cap["p_old"], cap["p_new"]
    sc.add("dU|num", xp.sqrt(xp.sum((U_new - U_old) ** 2, axis=1)).max())
    vmag = xp.maximum(xp.sqrt(xp.sum(U_new * U_new, axis=1)).max(), xp.sqrt(xp.sum(U_old * U_old, axis=1)).max())
    if solver.fv.n_bf:
        ub = solver.ub_fixed
        vmag = xp.maximum(vmag, xp.sqrt(xp.sum(ub * ub, axis=1)).max())
    sc.add("dU|ref", vmag)
    sc.add("dp|num", xp.abs(p_new - p_old).max())
    sc.add("dp|range", p_new.max() - p_new.min())
    health = {"velocity": U_new, "pressure": p_new, "face_mass_flux": cap["F"], "boundary_mass_flux": cap["Fb"]}
    if ecap is not None:
        from .simple import T_FIXED

        T_old, T_new = ecap["T_old"], ecap["T_new"]
        sc.add("dT|num", xp.abs(T_new - T_old).max())
        hi, lo = T_new.max(), T_new.min()
        if fv.n_bf:  # fixed boundary temperatures widen the range (no host round trip)
            fixed = solver.tkind == T_FIXED
            hi = xp.maximum(hi, xp.where(fixed, solver.tval, -xp.inf).max())
            lo = xp.minimum(lo, xp.where(fixed, solver.tval, xp.inf).min())
        sc.add("dT|ref", hi - lo)
        health["temperature"] = T_new
    for tname, (old, new) in (cap.get("turbulence") or {}).items():
        sc.add(f"d{tname}|num", xp.abs(new - old).max())
        sc.add(f"d{tname}|ref", xp.abs(new).max())
        health[tname] = new
    for hname, arr in health.items():
        sc.add(f"health|{hname}", (~xp.isfinite(arr)).sum())

    vals = sc.fetch()
    centres = fv.geometry.cell_centre
    out: dict = {"available": True, "version": DIAGNOSTICS_VERSION, "iteration": int(cap["iteration"]),
                 "rolled_back": bool(cap.get("rolled_back", False)), "precision": str(solver.dtype),
                 "unit_roundoff": u, "row_terms_max": K_rows, "cell_faces_max": K_cont}

    mom: dict = {"definition": "pre_solve: b - A U^n (assembled, unrelaxed); solve: b_r - A_r U* (system passed "
                               "to the linear solver)",
                 "relax_u": float(cap["alpha_u"]),
                 "row_scale_reference": {"pre_solve": vals["mom_pre|s_ref"], "solve": vals["mom_sol|s_ref"]},
                 "excluded_row_threshold": {"pre_solve": gamma(K_rows, u) * vals["mom_pre|s_ref"],
                                            "solve": gamma(K_rows, u) * vals["mom_sol|s_ref"]}}
    for k, nm in enumerate(names):
        p_rep = _row_report(vals, "mom_pre", nm, centres)
        s_rep = _row_report(vals, "mom_sol", nm, centres)
        s_rep["linear_solver"] = _linear_info(cap["mom_info"][k])
        s_rep["true_reduction_l2"] = ratio(s_rep["abs_l2"], vals[f"mom_true_r0|{nm}"])
        mom[nm] = {"pre_solve": p_rep, "solve": s_rep}
    out["momentum"] = mom

    if ecap is not None:
        p_rep = _row_report(vals, "T_pre", "energy", centres)
        s_rep = _row_report(vals, "T_sol", "energy", centres)
        s_rep["linear_solver"] = _linear_info(ecap.get("info"))
        s_rep["true_reduction_l2"] = ratio(s_rep["abs_l2"], vals["T_true_r0"])
        out["energy"] = {"pre_solve": p_rep, "solve": s_rep, "relax_T": float(ecap["alpha"])}

    pc = _row_report(vals, "pcorr", "p-correction", centres)
    pc["linear_solver"] = _linear_info(cap.get("p_info"))
    pc["relative_l2"] = ratio(pc["abs_l2"], vals["cont_pred|cells|l2"])
    pc["definition"] = "A_p p' = -m* (rhs = minus the predictor imbalance); residual = -(corrected imbalance)"
    out["pressure_correction"] = pc

    cont: dict = {"units": "kg/s (per unit depth in 2-D), positive = net outflow"}
    for tag, label in (("pred", "predictor"), ("corr", "corrected")):
        rep = _row_report(vals, "cont_" + tag, "cells", centres)
        cont[label] = {"local_max_relative": rep["row_scaled"], "abs_l1": rep["abs_l1"], "abs_max": rep["abs_max"],
                       "abs_l2": rep["abs_l2"],
                       "l1_over_face_flux": ratio(rep["abs_l1"], vals[f"cont_{tag}|face_flux_l1"]),
                       "active_cells": rep["active_rows"], "excluded_cells": rep["excluded_rows"],
                       "excluded_abs_max": rep["excluded_abs_max"], "nonfinite_cells": rep["nonfinite_rows"],
                       "sum_of_cells": vals[f"cont_{tag}|signed_sum"]}
        if "worst_cell" in rep:
            cont[label]["worst_cell"] = rep["worst_cell"]
            cont[label]["worst_cell_centre"] = rep["worst_cell_centre"]
    cont["global_boundary"] = {"signed": vals["global|signed"], "throughflow": vals["global|throughflow"],
                               "relative": ratio(abs(vals["global|signed"]), vals["global|throughflow"])}
    out["continuity"] = cont

    rolled = out["rolled_back"]
    U_ref = vals["dU|ref"]
    change: dict = {"velocity": None if rolled else ratio(vals["dU|num"], U_ref), "velocity_reference": U_ref}
    rho = float(solver.case.density)
    p_ref = max(vals["dp|range"], rho * U_ref * U_ref)
    change["pressure"] = None if rolled else ratio(vals["dp|num"], p_ref)
    change["pressure_reference"] = p_ref
    if ecap is not None:
        change["temperature"] = None if rolled else ratio(vals["dT|num"], vals["dT|ref"])
        change["temperature_reference"] = vals["dT|ref"]
    for tname in cap.get("turbulence") or {}:
        change[tname] = None if rolled else ratio(vals[f"d{tname}|num"], vals[f"d{tname}|ref"])
    out["field_change"] = change

    nonfinite = {hname: int(vals[f"health|{hname}"]) for hname in health}
    out["health"] = {"finite": not any(nonfinite.values()), "nonfinite_values": nonfinite}
    return out


def summary(d: dict) -> dict:
    """Flat headline numbers of :func:`evaluate` for residual events and logs."""
    if not d.get("available"):
        return {}
    mom = d["momentum"]
    comps = [k for k in COMPONENTS if k in mom]
    out: dict = {
        "iteration": d["iteration"],
        "momentum_pre_row_scaled": max(mom[k]["pre_solve"]["row_scaled"] for k in comps),
        "momentum_solve_row_scaled": max(mom[k]["solve"]["row_scaled"] for k in comps),
        "continuity_predictor_local_max": d["continuity"]["predictor"]["local_max_relative"],
        "continuity_corrected_local_max": d["continuity"]["corrected"]["local_max_relative"],
        "continuity_global_relative": d["continuity"]["global_boundary"]["relative"],
        "velocity_change": d["field_change"]["velocity"],
        "pressure_change": d["field_change"]["pressure"],
        "finite": d["health"]["finite"],
    }
    if "energy" in d:
        out["energy_pre_row_scaled"] = d["energy"]["pre_solve"]["row_scaled"]
    return out


def pressure_matrix_properties(solver) -> Optional[dict]:
    """Verify, on the assembled pressure-correction matrix of the latest iteration, the
    conditions under which preconditioned CG applies: symmetry, non-positive off-diagonal
    entries and weak diagonal dominance (a symmetric M-matrix, hence positive semi-definite;
    singular, up to a constant, when no boundary fixes the pressure). None before the first
    iteration. Runs on the host (one transfer of the matrix): call it once, not per iteration."""
    cap = getattr(solver, "_capture", None)
    if cap is None:
        return None
    import scipy.sparse as sp

    Ap = cap["Ap"]
    A = getattr(Ap, "csr", None)
    if A is None:
        A = Ap.get() if hasattr(Ap, "get") else Ap
    A = sp.csr_matrix(A, dtype=np.float64)
    scale = float(abs(A).max()) or 1.0
    asym = float(abs(A - A.T).max()) / scale if A.nnz else 0.0
    diag = A.diagonal()
    off = A - sp.diags(diag)
    off_row_abs = np.asarray(abs(off).sum(axis=1)).ravel()
    margin = (diag - off_row_abs) / np.maximum(np.abs(diag), 1e-300)
    tol = 64 * unit_roundoff(np.float64)
    out = {"symmetric": asym <= tol, "max_relative_asymmetry": asym,
           "offdiagonal_nonpositive": bool(off.data.max() <= tol * scale) if off.nnz else True,
           "weakly_diagonally_dominant": bool(margin.min() >= -tol) and bool((diag > 0).all()),
           "min_dominance_margin": float(margin.min()), "singular": bool(getattr(solver, "singular_p", False))}
    out["cg_applicable"] = out["symmetric"] and out["offdiagonal_nonpositive"] and out["weakly_diagonally_dominant"]
    return out
