"""Ansys Fluent CFF/HDF5 files: ``.cas.h5`` / ``.msh.h5`` (mesh) and ``.dat.h5`` (cell data).

Fluent 2020 R1 and later write case and data files in the Common Fluids
Format (CFF), an HDF5 layout. Ansys does not publish its specification; the
layout implemented here is the one read by the open-source VTK reader
``vtkFLUENTCFFReader`` (VTK 9.3+, ``IO/FLUENTCFF``):

* ``/meshes/1``                       attrs ``dimension``, ``nodeCount``, ``cellCount``, ``faceCount``
* ``/meshes/1/nodes/zoneTopology``    ``id, minId, maxId, dimension``; ``coords/<id>`` (n x dim, float64)
* ``/meshes/1/cells/zoneTopology``    ``id, minId, maxId, cellType, childZoneId, dimension``
* ``/meshes/1/cells/ctype/<k>``       attrs ``elementType, minId, maxId``; ``cell-types`` (int16)
* ``/meshes/1/faces/zoneTopology``    ``id, minId, maxId, zoneType, faceType, childZoneId, shadowZoneId,
  flags, dimension`` and ``name`` (zone names joined by ``;``)
* ``/meshes/1/faces/nodes/<k>``       ``nnodes`` (int16), ``nodes`` (uint32, 1-based)
* ``/meshes/1/faces/c0|c1/<k>``       owner / neighbour (uint32, 1-based, 0 = none)
* ``/results/1/phase-1/cells``        ``fields`` (``;``-joined names), ``<name>/<k>`` sections with
  attrs ``minId, maxId`` (Fluent solver-variable names such as ``SV_P``, ``SV_U``)

The reader is tolerant (sections in any number, names as fixed or variable
length strings, 32- or 64-bit data). The writer produces exactly this layout
plus a minimal ``/settings`` group. **Status:** round-trip tested within
FlowTrace and against the public reader's layout; *not yet verified against
files written by Fluent itself or read back into Fluent* (no licence was
available). The ``.msh`` + ``.ip`` route remains the verified hand-off.
Cell-zone types (fluid/solid) and periodic pairs, which the public layout
does not describe, are stored in a FlowTrace extension group ``/flowtrace``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .unstructured import BC_CODES, BC_NAMES, UnstructuredMesh, Zone, sort_faces_by_zone

# Fluent field names (as in .ip/.prof) <-> CFF solver-variable names
SV_NAMES = {
    "pressure": "SV_P", "x-velocity": "SV_U", "y-velocity": "SV_V", "z-velocity": "SV_W",
    "temperature": "SV_T", "turb-kinetic-energy": "SV_K", "turb-diss-rate": "SV_D",
    "specific-diss-rate": "SV_O", "density": "SV_DENSITY", "viscosity-turb": "SV_MU_T",
}
FIELD_NAMES = {v: k for k, v in SV_NAMES.items()}


def _h5():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise ImportError("reading/writing .cas.h5/.dat.h5 needs h5py: pip install h5py "
                          "(or pip install flowtrace[h5])") from exc
    return h5py


def _string(ds) -> str:
    v = ds[()]
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else b""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _put_string(group, name: str, text: str):
    h5py = _h5()
    group.create_dataset(name, data=np.bytes_(text.encode("utf-8")), dtype=h5py.string_dtype("ascii", len(text) or 1))


def _sections(group):
    n = int(group.attrs.get("nSections", 0)) or len([k for k in group.keys() if k.isdigit()])
    return [group[str(k)] for k in range(1, n + 1) if str(k) in group]


# --------------------------------------------------------------------------
# element types (as in the .msh writer)
# --------------------------------------------------------------------------

def _cell_types(mesh: UnstructuredMesh) -> np.ndarray:
    from .fluent_msh import _cell_types as ct

    return ct(mesh)


def write_case_h5(mesh: UnstructuredMesh, path: str, title: str = "FlowTrace") -> str:
    """Write the mesh as a CFF ``.cas.h5``/``.msh.h5`` file."""
    h5py = _h5()
    mesh = sort_faces_by_zone(mesh)
    dim, nn, nc, nf = mesh.dim, mesh.n_nodes, mesh.n_cells, mesh.n_faces
    with h5py.File(path, "w") as f:
        f.attrs["FlowTrace"] = np.bytes_(f"{title} CFF writer".encode())
        m = f.create_group("meshes/1")
        m.attrs["dimension"] = np.int32(dim)
        for k, v in (("nodeOffset", 0), ("nodeCount", nn), ("cellOffset", 0), ("cellCount", nc),
                     ("faceOffset", 0), ("faceCount", nf)):
            m.attrs[k] = np.uint64(v)
        # nodes: one zone
        nz = m.create_group("nodes/zoneTopology")
        nz.attrs["nZones"] = np.uint64(1)
        nz.create_dataset("id", data=np.array([1], np.int32))
        nz.create_dataset("minId", data=np.array([1], np.uint64))
        nz.create_dataset("maxId", data=np.array([nn], np.uint64))
        nz.create_dataset("dimension", data=np.array([dim], np.uint64))
        c = m.create_dataset("nodes/coords/1", data=np.ascontiguousarray(mesh.nodes[:, :dim], dtype=np.float64))
        c.attrs["minId"], c.attrs["maxId"] = np.uint64(1), np.uint64(nn)
        # cells
        czones = sorted(mesh.cell_zones(), key=lambda z: z.first)
        types = _cell_types(mesh).astype(np.int16)
        cz = m.create_group("cells/zoneTopology")
        cz.attrs["nZones"] = np.uint64(len(czones))
        ids, mins, maxs, ctypes = [], [], [], []
        for z in czones:
            cells = np.nonzero(mesh.cell_zone == z.id)[0]
            if cells.size and cells[-1] - cells[0] + 1 != cells.size:
                raise ValueError("cell zones must be contiguous to write a CFF case")
            t = types[cells]
            ids.append(z.id)
            mins.append(cells[0] + 1)
            maxs.append(cells[-1] + 1)
            ctypes.append(int(t[0]) if np.all(t == t[0]) else 0)
        cz.create_dataset("id", data=np.array(ids, np.int32))
        cz.create_dataset("minId", data=np.array(mins, np.uint64))
        cz.create_dataset("maxId", data=np.array(maxs, np.uint64))
        cz.create_dataset("cellType", data=np.array(ctypes, np.int32))
        cz.create_dataset("childZoneId", data=np.zeros(len(ids), np.int32))
        cz.create_dataset("dimension", data=np.full(len(ids), dim, np.uint64))
        _put_string(cz, "name", ";".join(z.name for z in czones))
        ct = m.create_group("cells/ctype")
        ct.attrs["nSections"] = np.uint64(1)
        s = ct.create_group("1")
        s.attrs["elementType"] = np.int16(0)
        s.attrs["minId"], s.attrs["maxId"] = np.uint64(1), np.uint64(nc)
        s.create_dataset("cell-types", data=types)
        # faces: one section per zone
        fzones = sorted(mesh.face_zones(), key=lambda z: z.first)
        counts = np.diff(mesh.face_ptr)
        fz = m.create_group("faces/zoneTopology")
        fz.attrs["nZones"] = np.uint64(len(fzones))
        fid, fmin, fmax, ztype, ftype = [], [], [], [], []
        fn = m.create_group("faces/nodes")
        fc0 = m.create_group("faces/c0")
        fc1 = m.create_group("faces/c1")
        for k, z in enumerate(fzones, start=1):
            fids = np.arange(z.first, z.last + 1)
            cnt = counts[fids]
            fid.append(z.id)
            fmin.append(z.first + 1)
            fmax.append(z.last + 1)
            ztype.append(BC_NAMES.get(z.type, 3))
            ftype.append(int(cnt[0]) if np.all(cnt == cnt[0]) and cnt[0] in (2, 3, 4) and not (dim == 3 and cnt[0] == 2) else 0)
            sec = fn.create_group(str(k))
            sec.attrs["minId"], sec.attrs["maxId"] = np.uint64(z.first + 1), np.uint64(z.last + 1)
            sec.create_dataset("nnodes", data=cnt.astype(np.int16))
            sl = slice(mesh.face_ptr[z.first], mesh.face_ptr[z.last + 1])
            sec.create_dataset("nodes", data=(mesh.face_nodes[sl] + 1).astype(np.uint32))
            c1 = np.where(mesh.neighbour[fids] >= 0, mesh.neighbour[fids] + 1, 0)
            for grp, arr in ((fc0, mesh.owner[fids] + 1), (fc1, c1)):
                d = grp.create_dataset(str(k), data=arr.astype(np.uint32))
                d.attrs["minId"], d.attrs["maxId"] = np.uint64(z.first + 1), np.uint64(z.last + 1)
        for grp in (fn, fc0, fc1):
            grp.attrs["nSections"] = np.uint64(len(fzones))
        fz.create_dataset("id", data=np.array(fid, np.int32))
        fz.create_dataset("minId", data=np.array(fmin, np.uint64))
        fz.create_dataset("maxId", data=np.array(fmax, np.uint64))
        fz.create_dataset("zoneType", data=np.array(ztype, np.int32))
        fz.create_dataset("faceType", data=np.array(ftype, np.int32))
        n = len(fzones)
        fz.create_dataset("childZoneId", data=np.zeros(n, np.int32))
        shadow = np.zeros(n, np.int32)
        for p in mesh.periodic:
            shadow[[i for i, z in enumerate(fzones) if z.id == p["zone_a"]]] = p["zone_b"]
        fz.create_dataset("shadowZoneId", data=shadow)
        fz.create_dataset("flags", data=np.zeros(n, np.int32))
        fz.create_dataset("dimension", data=np.full(n, dim, np.uint64))
        _put_string(fz, "name", ";".join(z.name for z in fzones))
        st = f.create_group("settings")
        _put_string(st, "Origin", f"{title}: mesh only, no solver settings")
        ext = f.create_group("flowtrace")
        _put_string(ext, "cellZoneTypes", ";".join(z.type for z in czones))
        for i, p in enumerate(mesh.periodic):
            g = ext.create_group(f"periodic/{i}")
            g.attrs["zone_a"], g.attrs["zone_b"] = int(p["zone_a"]), int(p["zone_b"])
            g.attrs["translation"] = np.asarray(p["translation"], dtype=float)
            g.create_dataset("faces_a", data=np.asarray(p["faces_a"], np.int64))
            g.create_dataset("faces_b", data=np.asarray(p["faces_b"], np.int64))
    return path


def read_case_h5(path: str, name: Optional[str] = None) -> UnstructuredMesh:
    """Read the mesh of a CFF ``.cas.h5``/``.msh.h5`` file."""
    import os

    h5py = _h5()
    with h5py.File(path, "r") as f:
        if "meshes/1" not in f:
            raise ValueError(f"{path}: no /meshes/1 group (not a Fluent CFF case/mesh file)")
        m = f["meshes/1"]
        dim = int(m.attrs["dimension"])
        nn = int(m.attrs.get("nodeCount", 0))
        nz = m["nodes/zoneTopology"]
        node_blocks = []
        for zid, lo, hi in zip(nz["id"][()], nz["minId"][()], nz["maxId"][()]):
            ds = m[f"nodes/coords/{int(zid)}"]
            xyz = np.asarray(ds[()], dtype=np.float64).reshape(int(hi) - int(lo) + 1, -1)
            node_blocks.append((int(lo), xyz))
        nn = max(nn, max(lo + len(x) - 1 for lo, x in node_blocks))
        nodes = np.zeros((nn, dim))
        for lo, xyz in node_blocks:
            nodes[lo - 1:lo - 1 + len(xyz)] = xyz[:, :dim]
        fz = m["faces/zoneTopology"]
        f_ids = [int(x) for x in fz["id"][()]]
        f_min = [int(x) for x in fz["minId"][()]]
        f_max = [int(x) for x in fz["maxId"][()]]
        f_type = [int(x) for x in fz["zoneType"][()]] if "zoneType" in fz else [3] * len(f_ids)
        f_names = _string(fz["name"]).split(";") if "name" in fz else []
        nf = int(m.attrs.get("faceCount", max(f_max)))
        counts = np.zeros(nf, np.int64)
        covered = np.zeros(nf, bool)
        secs = []
        for sec in _sections(m["faces/nodes"]):
            lo, hi = int(sec.attrs["minId"]), int(sec.attrs["maxId"])
            nnod = np.asarray(sec["nnodes"][()], np.int64)
            counts[lo - 1:hi] = nnod
            covered[lo - 1:hi] = True
            secs.append((lo, np.asarray(sec["nodes"][()], np.int64)[:int(nnod.sum())] - 1))
        secs.sort(key=lambda t: t[0])
        face_nodes = np.concatenate([s[1] for s in secs]) if secs else np.zeros(0, np.int64)
        c0 = np.zeros(nf, np.int64)
        c1 = np.zeros(nf, np.int64)
        for grp, arr in (("faces/c0", c0), ("faces/c1", c1)):
            for sec in _sections(m[grp]):
                lo, hi = int(sec.attrs["minId"]), int(sec.attrs["maxId"])
                arr[lo - 1:hi] = np.asarray(sec[()], np.int64)
        cz = m["cells/zoneTopology"]
        c_ids = [int(x) for x in cz["id"][()]]
        c_min = [int(x) for x in cz["minId"][()]]
        c_max = [int(x) for x in cz["maxId"][()]]
        c_names = _string(cz["name"]).split(";") if "name" in cz else []
        c_types = []
        if "flowtrace/cellZoneTypes" in f:
            c_types = _string(f["flowtrace/cellZoneTypes"]).split(";")
        periodic = []
        if "flowtrace/periodic" in f:
            for g in f["flowtrace/periodic"].values():
                periodic.append({"zone_a": int(g.attrs["zone_a"]), "zone_b": int(g.attrs["zone_b"]),
                                 "translation": np.asarray(g.attrs["translation"], float),
                                 "faces_a": np.asarray(g["faces_a"][()]), "faces_b": np.asarray(g["faces_b"][()])})
    if not covered.all():
        raise ValueError(f"{path}: face/node sections do not cover all {nf} faces")
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    # boundary faces stored with c0 == 0: flip so that the owner exists
    swap = (c0 == 0) & (c1 != 0)
    for fidx in np.nonzero(swap)[0]:
        face_nodes[ptr[fidx]:ptr[fidx + 1]] = face_nodes[ptr[fidx]:ptr[fidx + 1]][::-1].copy()
    c0, c1 = np.where(swap, c1, c0), np.where(swap, 0, c1)
    n_cells = int(max(c0.max(), c1.max(), max(c_max) if c_max else 0))
    face_zone = np.zeros(nf, np.int64)
    zones: Dict[int, Zone] = {}
    for i, (zid, lo, hi) in enumerate(zip(f_ids, f_min, f_max)):
        face_zone[lo - 1:hi] = zid
        nm = f_names[i] if i < len(f_names) and f_names[i] else f"zone-{zid}"
        zones[zid] = Zone(zid, nm, BC_CODES.get(f_type[i], "wall"), "face", lo - 1, hi - 1)
    cell_zone = np.zeros(n_cells, np.int64)
    for i, (zid, lo, hi) in enumerate(zip(c_ids, c_min, c_max)):
        cell_zone[lo - 1:hi] = zid
        nm = c_names[i] if i < len(c_names) and c_names[i] else f"zone-{zid}"
        zt = c_types[i] if i < len(c_types) and c_types[i] else "fluid"
        zones[zid] = Zone(zid, nm, zt, "cell", lo - 1, hi - 1)
    mesh = UnstructuredMesh(dim=dim, nodes=nodes, face_ptr=ptr, face_nodes=face_nodes.astype(np.int64),
                            owner=c0 - 1, neighbour=np.where(c1 > 0, c1 - 1, -1), face_zone=face_zone,
                            n_cells=n_cells, cell_zone=cell_zone, zones=zones,
                            name=name or os.path.basename(str(path)), periodic=periodic)
    return mesh


def write_data_h5(path: str, mesh: UnstructuredMesh, fields: Dict[str, np.ndarray], iteration: int = 0) -> str:
    """Cell data (Fluent field names) as a CFF ``.dat.h5``: one section per cell zone."""
    h5py = _h5()
    czones = sorted(mesh.cell_zones(), key=lambda z: z.first)
    names = []
    with h5py.File(path, "w") as f:
        f.attrs["FlowTrace"] = np.bytes_(b"FlowTrace CFF data writer")
        cells = f.create_group("results/1/phase-1/cells")
        for fname, vals in fields.items():
            sv = SV_NAMES.get(fname, fname)
            v = np.asarray(vals, dtype=np.float64).ravel()
            if v.size != mesh.n_cells:
                raise ValueError(f"{fname}: expected {mesh.n_cells} values, got {v.size}")
            g = cells.create_group(sv)
            g.attrs["nSections"] = np.uint64(len(czones))
            for k, z in enumerate(czones, start=1):
                idx = np.nonzero(mesh.cell_zone == z.id)[0]
                d = g.create_dataset(str(k), data=v[idx[0]:idx[-1] + 1])
                d.attrs["minId"], d.attrs["maxId"] = np.uint64(idx[0] + 1), np.uint64(idx[-1] + 1)
            names.append(sv)
        _put_string(cells, "fields", ";".join(names))
        f["results/1"].attrs["iteration"] = np.int64(iteration)
    return path


def read_data_h5(path: str, n_cells: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Cell fields of a CFF ``.dat.h5`` keyed by Fluent field name (unknown SV names kept)."""
    h5py = _h5()
    out = {}
    with h5py.File(path, "r") as f:
        cells = f["results/1/phase-1/cells"]
        names = [s for s in _string(cells["fields"]).split(";") if s]
        for sv in names:
            if sv not in cells:
                continue
            secs = _sections(cells[sv])
            hi = max(int(s.attrs["maxId"]) for s in secs)
            n = max(hi, n_cells or 0)
            first = np.asarray(secs[0][()])
            ncomp = first.shape[1] if first.ndim == 2 else 1
            arr = np.full((n, ncomp), np.nan)
            for s in secs:
                lo, hi_ = int(s.attrs["minId"]), int(s.attrs["maxId"])
                d = np.asarray(s[()], dtype=np.float64).reshape(hi_ - lo + 1, -1)
                arr[lo - 1:hi_] = d
            out[FIELD_NAMES.get(sv, sv)] = arr[:, 0] if ncomp == 1 else arr
    return out


def describe_h5(path: str, max_items: int = 400) -> str:
    """Tree of groups/datasets/attributes (to debug files from other tools)."""
    h5py = _h5()
    lines = []

    def visit(name, obj):
        if len(lines) >= max_items:
            return
        if isinstance(obj, h5py.Dataset):
            lines.append(f"{name}  dataset {obj.shape} {obj.dtype}")
        else:
            lines.append(f"{name}/")
        for k, v in obj.attrs.items():
            lines.append(f"{name}  @{k} = {v!r}"[:200])

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return "\n".join(lines)


def is_h5(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89HDF\r\n\x1a\n"


def read_any_mesh(path: str) -> Tuple[UnstructuredMesh, str]:
    """Read .msh/.cas (text/binary, optionally gzipped) or .cas.h5/.msh.h5 -> (mesh, format)."""
    from .fluent_msh import read_fluent_mesh

    if is_h5(path):
        return read_case_h5(path), "cff-hdf5"
    return read_fluent_mesh(path), "fluent-msh"
