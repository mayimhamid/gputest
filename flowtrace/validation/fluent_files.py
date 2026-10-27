"""Check FlowTrace's Fluent readers and writers against files written by Ansys Fluent itself.

Reference files: the PyFluent "mixing elbow" example (a 3-D polyhedral mesh in inches, the case
scaled to metres, and a converged solution), published by Ansys in the ``ansys/example-data``
repository under the MIT licence (Copyright Synopsys, Inc. and ANSYS, Inc.). They are not shipped
with FlowTrace: :func:`download` fetches them and verifies their SHA-256 checksums, and
:func:`check` reads them and tests the properties below. ``benchmarks/fluent_files_check.py``
runs both and writes ``benchmarks/results/fluent_files_check.md``; ``tests/test_fluent_files.py``
runs :func:`check` when the files are present.

What is verified (and what is not):

* the CFF reader understands Fluent's layout: counts, zone names and types, mixed polygonal faces;
* orientation: Fluent stores 3-D faces with the right-hand normal toward c0; read correctly, no face
  needs re-orientation and every cell closes (sum of outward area vectors = 0 to round-off);
* the ``.msh.h5`` (inches) and ``.cas.h5`` (metres) describe the same mesh: identical topology and
  volumes in the ratio 0.0254^3;
* writing the mesh back with FlowTrace reproduces Fluent's node order and c0/c1 for every face;
* the ``.dat.h5`` cell data maps onto the case's cells: the mass flow through the outlet, from
  the cell values next to it, matches the two inlets within 10 % (a wrong cell mapping gives
  values far outside that band).

Not verified here: Fluent reading files that FlowTrace wrote (needs a licensed Fluent).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from typing import Dict, Optional

import numpy as np

BASE_URL = "https://raw.githubusercontent.com/ansys/example-data/master/pyfluent/mixing_elbow/"
FILES = {
    "mixing_elbow.msh.h5": "0e449d741dd7cdc48aa3f75e19f1f99281972b9f8d4637da2d1a6be25f9e2cb0",
    "mixing_elbow.cas.h5": "8145813f50f989a521ed8d1b02c891c7ca22ccc5fafa35a4eeba011949f19cbc",
    "mixing_elbow.dat.h5": "863a813227cf4274a9ca24fdad43e6c1f9f132dd62fcabc39a3be8cf493cff42",
}
LICENSE_NOTE = ("PyFluent mixing-elbow example, ansys/example-data (MIT licence, "
                "Copyright Synopsys, Inc. and ANSYS, Inc.)")
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".flowtrace", "fluent_examples")
INCH = 0.0254


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find(directory: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Paths of the three reference files if all are present with the expected checksums."""
    d = directory or os.environ.get("FLOWTRACE_FLUENT_EXAMPLES") or DEFAULT_DIR
    paths = {name: os.path.join(d, name) for name in FILES}
    if not all(os.path.isfile(p) for p in paths.values()):
        return None
    if any(sha256(p) != FILES[n] for n, p in paths.items()):
        return None
    return paths


def download(directory: str = DEFAULT_DIR) -> Dict[str, str]:
    """Fetch the reference files (about 9 MB) and verify their SHA-256; return their paths."""
    os.makedirs(directory, exist_ok=True)
    out = {}
    for name, digest in FILES.items():
        path = os.path.join(directory, name)
        if not (os.path.isfile(path) and sha256(path) == digest):
            tmp = path + ".part"
            urllib.request.urlretrieve(BASE_URL + name, tmp)  # noqa: S310 (fixed https URL)
            if sha256(tmp) != digest:
                os.remove(tmp)
                raise RuntimeError(f"{name}: checksum mismatch; the published file changed")
            os.replace(tmp, path)
        out[name] = path
    return out


def _closure(mesh, geo) -> float:
    """max over cells of |sum of outward area vectors| / sum of face areas."""
    nc = mesh.n_cells
    inter = mesh.neighbour >= 0
    S = geo.face_area_vec
    acc = np.zeros((nc, mesh.dim))
    for d in range(mesh.dim):
        acc[:, d] = (np.bincount(mesh.owner, S[:, d], minlength=nc)
                     - np.bincount(mesh.neighbour[inter], S[inter, d], minlength=nc))
    area = np.bincount(mesh.owner, geo.face_area, minlength=nc) + np.bincount(
        mesh.neighbour[inter], geo.face_area[inter], minlength=nc)
    return float((np.linalg.norm(acc, axis=1) / area).max())


def _stored_faces(path: str):
    """(node lists in file order, c0, c1) exactly as stored in a CFF case/mesh file; every section
    is placed by its own minId/maxId attributes, as Fluent may split the groups differently."""
    import h5py

    def ids(obj):
        return (int(np.asarray(obj.attrs["minId"]).ravel()[0]), int(np.asarray(obj.attrs["maxId"]).ravel()[0]))

    with h5py.File(path, "r") as f:
        g = f["meshes/1/faces"]
        nf = max(ids(sec)[1] for sec in g["nodes"].values())
        nodes = [()] * nf
        for sec in g["nodes"].values():
            lo, hi = ids(sec)
            cnt = np.asarray(sec["nnodes"][()], np.int64)
            nod = np.asarray(sec["nodes"][()], np.int64)
            ptr = np.concatenate([[0], np.cumsum(cnt)])
            for i in range(hi - lo + 1):
                nodes[lo - 1 + i] = tuple(int(v) for v in nod[ptr[i]:ptr[i + 1]])
        cells = {}
        for grp in ("c0", "c1"):
            arr = np.zeros(nf, np.int64)
            for sec in g[grp].values():
                lo, hi = ids(sec)
                arr[lo - 1:hi] = np.asarray(sec[()], np.int64)[:hi - lo + 1]
            cells[grp] = arr
    return nodes, cells["c0"], cells["c1"]


def check(paths: Dict[str, str], workdir: Optional[str] = None) -> dict:
    """Read the reference files and return the verified quantities (see the module docstring)."""
    import tempfile

    from ..mesh.fluent_h5 import read_any_mesh, read_data_h5, write_case_h5
    from ..mesh.unstructured import compute_geometry

    out: dict = {"source": LICENSE_NOTE}
    meshes = {}
    for key in ("mixing_elbow.msh.h5", "mixing_elbow.cas.h5"):
        m, _ = read_any_mesh(paths[key])
        g = compute_geometry(m)
        meshes[key] = (m, g)
        out[key] = {"cells": m.n_cells, "faces": m.n_faces, "nodes": m.n_nodes, "dim": m.dim,
                    "nodes_per_face": sorted(int(k) for k in np.unique(np.diff(m.face_ptr))),
                    "zones": {z.name: z.type for z in m.zones.values()},
                    "faces_reoriented": int(g.flipped_faces), "closure_max": _closure(m, g),
                    "volume": float(g.total_volume), "min_cell_volume": float(g.cell_volume.min())}
    (mm, gm), (mc, gc) = meshes["mixing_elbow.msh.h5"], meshes["mixing_elbow.cas.h5"]
    out["same_topology"] = bool(np.array_equal(mm.owner, mc.owner) and np.array_equal(mm.neighbour, mc.neighbour)
                                and np.array_equal(mm.face_nodes, mc.face_nodes))
    out["volume_ratio_cas_over_msh"] = float(gc.total_volume / gm.total_volume)
    out["volume_ratio_expected"] = INCH ** 3

    # write back with FlowTrace: the stored node order and c0/c1 must equal Fluent's
    tmpdir = workdir or tempfile.mkdtemp(prefix="flowtrace_fluent_")
    back = os.path.join(tmpdir, "roundtrip.cas.h5")
    write_case_h5(mc, back)
    n_ref, c0_ref, c1_ref = _stored_faces(paths["mixing_elbow.cas.h5"])
    n_new, c0_new, c1_new = _stored_faces(back)
    by_ref = {fs: (a, b) for fs, a, b in zip(map(frozenset, n_ref), c0_ref, c1_ref)}
    same_order = same_cells = 0
    ref_order = {frozenset(n): n for n in n_ref}
    for n, a, b in zip(n_new, c0_new, c1_new):
        key = frozenset(n)
        ra, rb = by_ref[key]
        same_cells += int((a, b) == (ra, rb))
        r = ref_order[key]
        k = r.index(n[0])
        same_order += int(tuple(r[k:] + r[:k]) == tuple(n))  # same cyclic order (start node may differ)
    out["roundtrip_faces"] = len(n_new)
    out["roundtrip_same_cyclic_node_order"] = same_order
    out["roundtrip_same_c0_c1"] = same_cells

    # data mapping: mass flow through the inlets and the outlet from the adjacent cell values
    data = read_data_h5(paths["mixing_elbow.dat.h5"])
    rho = np.asarray(data["density"], float)
    U = np.column_stack([np.asarray(data[k], float) for k in ("x-velocity", "y-velocity", "z-velocity")])
    S = gc.face_area_vec
    flows = {}
    for z in mc.boundary_zones():
        f = mc.faces_of_zone(z.id)
        cells = mc.owner[f]
        flows[z.name] = float(np.sum(rho[cells] * np.einsum("fd,fd->f", U[cells], S[f])))  # > 0 = outflow
    out["mass_flow_by_zone"] = flows
    m_in = -(flows.get("hot-inlet", 0.0) + flows.get("cold-inlet", 0.0))
    out["outlet_over_inlets"] = float(flows.get("outlet", 0.0) / m_in) if m_in > 0 else float("nan")
    out["data_cells"] = int(rho.size)
    return out


def verdicts(r: dict) -> Dict[str, bool]:
    """Pass/fail of each verified property (criteria fixed in the module docstring)."""
    msh, cas = r["mixing_elbow.msh.h5"], r["mixing_elbow.cas.h5"]
    return {
        "3-D polyhedral mesh read (17,822 cells, 91,581 faces)": (cas["cells"], cas["faces"], cas["dim"]) == (17822, 91581, 3),
        "zone types": cas["zones"].get("hot-inlet") == "velocity-inlet" and cas["zones"].get("outlet") == "pressure-outlet"
        and cas["zones"].get("wall-elbow") == "wall" and cas["zones"].get("elbow-fluid") == "fluid",
        "no face re-oriented on read": msh["faces_reoriented"] == 0 and cas["faces_reoriented"] == 0,
        "every cell closes (|sum S| / sum |S| < 1e-12)": max(msh["closure_max"], cas["closure_max"]) < 1e-12,
        "mesh and case: same topology, volumes in the ratio 0.0254^3":
            r["same_topology"] and abs(r["volume_ratio_cas_over_msh"] / r["volume_ratio_expected"] - 1) < 1e-9,
        "written back: Fluent's node order and c0/c1 for every face":
            r["roundtrip_same_cyclic_node_order"] == r["roundtrip_faces"] == r["roundtrip_same_c0_c1"],
        "data mapped to cells: outlet / inlets mass flow within 10 %":
            r["data_cells"] == cas["cells"] and 0.9 <= r["outlet_over_inlets"] <= 1.1,
    }
