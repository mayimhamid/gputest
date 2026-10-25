"""Reader and writer for Ansys Fluent mesh files (.msh, legacy .cas, optionally gzipped).

Supported sections
------------------
0  comment            2  dimension
10 / 2010 / 3010      nodes (ASCII, binary single, binary double)
12 / 2012 / 3012      cells (declarations, element types)
13 / 2013 / 3013      faces (linear, tri, quad, polygonal and mixed)
18 / 2018             periodic shadow pairs (parsed and kept)
39 / 45               zone names and types
Everything else (case settings in a legacy .cas, etc.) is skipped safely.

Header numbers in grid sections are hexadecimal; zone ids in 39/45 are
decimal, exactly as in the Fluent file-format specification. Face records are
``n0 n1 ... c0 c1`` with c0/c1 1-based and 0 meaning "no cell"; the area
vector from the right-hand rule on the node order points from c0 to c1.
"""

from __future__ import annotations

import gzip
import io
import os
import re
from typing import Dict, List, Tuple

import numpy as np

from .unstructured import BC_CODES, BC_NAMES, UnstructuredMesh, Zone, sort_faces_by_zone

_WS = b" \t\r\n"


class FluentMeshError(ValueError):
    pass


def _open_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        head = fh.read(2)
    opener = gzip.open if head == b"\x1f\x8b" else open
    with opener(path, "rb") as fh:
        return fh.read()


class _Scanner:
    def __init__(self, buf: bytes):
        self.b = buf
        self.n = len(buf)
        self.p = 0

    def skip_ws(self):
        b, n, p = self.b, self.n, self.p
        while p < n and b[p] in _WS:
            p += 1
        self.p = p

    def expect(self, ch: bytes):
        self.skip_ws()
        if self.b[self.p:self.p + 1] != ch:
            ctx = self.b[self.p:self.p + 40]
            raise FluentMeshError(f"expected {ch!r} at byte {self.p}, found {ctx!r}")
        self.p += 1

    def peek(self) -> bytes:
        self.skip_ws()
        return self.b[self.p:self.p + 1]

    def read_token(self) -> bytes:
        self.skip_ws()
        b, p, n = self.b, self.p, self.n
        s = p
        while p < n and b[p] not in _WS and b[p] not in b"()":
            p += 1
        self.p = p
        return b[s:p]

    def skip_string(self):
        # at an opening quote
        p = self.p + 1
        b = self.b
        while True:
            q = b.find(b'"', p)
            if q < 0:
                raise FluentMeshError("unterminated string")
            if b[q - 1:q] != b"\\":
                self.p = q + 1
                return
            p = q + 1

    def skip_balanced(self):
        """Skip a parenthesised ASCII expression starting at '('."""
        self.expect(b"(")
        depth = 1
        b, n = self.b, self.n
        p = self.p
        while depth and p < n:
            c = b[p]
            if c == 0x22:  # '"'
                self.p = p
                self.skip_string()
                p = self.p
                continue
            if c == 0x28:
                depth += 1
            elif c == 0x29:
                depth -= 1
            p += 1
        self.p = p

    def read_header(self) -> List[bytes]:
        self.expect(b"(")
        q = self.b.find(b")", self.p)
        toks = self.b[self.p:q].split()
        self.p = q + 1
        return toks

    def ascii_block(self) -> bytes:
        """Return the raw text of '( ... )' (no nested parens in grid data)."""
        self.expect(b"(")
        q = self.b.find(b")", self.p)
        if q < 0:
            raise FluentMeshError("unterminated data block")
        data = self.b[self.p:q]
        self.p = q + 1
        return data

    def end_binary(self, index: int):
        marker = b"End of Binary Section"
        q = self.b.find(marker, self.p)
        if q < 0:
            raise FluentMeshError(f"missing 'End of Binary Section {index}'")
        self.p = q + len(marker)
        self.read_token()  # the index
        self.expect(b")")


def _hex(tok: bytes) -> int:
    return int(tok, 16)


def _parse_hex_ints(data: bytes) -> np.ndarray:
    toks = data.split()
    if not toks:
        return np.zeros(0, dtype=np.int64)
    # numpy cannot parse hex directly; this is the bulk cost for large ASCII meshes
    return np.fromiter((int(t, 16) for t in toks), dtype=np.int64, count=len(toks))


def _faces_from_ints(vals: np.ndarray, nfaces: int, face_type: int):
    """Split a flat face record list into (counts, nodes, c0, c1)."""
    if face_type in (2, 3, 4):
        k = face_type
        rec = vals[: nfaces * (k + 2)].reshape(nfaces, k + 2)
        counts = np.full(nfaces, k, dtype=np.int64)
        return counts, rec[:, :k].ravel(), rec[:, k], rec[:, k + 1], nfaces * (k + 2)
    counts = np.empty(nfaces, dtype=np.int64)
    c0 = np.empty(nfaces, dtype=np.int64)
    c1 = np.empty(nfaces, dtype=np.int64)
    nodes = []
    p = 0
    for i in range(nfaces):
        k = int(vals[p])
        counts[i] = k
        nodes.append(vals[p + 1:p + 1 + k])
        c0[i] = vals[p + 1 + k]
        c1[i] = vals[p + 2 + k]
        p += k + 3
    return counts, (np.concatenate(nodes) if nodes else np.zeros(0, np.int64)), c0, c1, p


def read_fluent_mesh(path_or_bytes, name: str = None) -> UnstructuredMesh:
    """Read a Fluent .msh/.cas (ASCII or binary, optionally .gz) into an UnstructuredMesh."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf = bytes(path_or_bytes)
        if buf[:2] == b"\x1f\x8b":
            buf = gzip.decompress(buf)
        name = name or "mesh"
    else:
        buf = _open_bytes(path_or_bytes)
        name = name or os.path.basename(str(path_or_bytes))
    sc = _Scanner(buf)
    dim = None
    node_blocks: List[Tuple[int, np.ndarray]] = []
    face_blocks = []
    cell_blocks = []
    zone_info: Dict[int, Tuple[str, str]] = {}
    total_nodes = total_cells = total_faces = 0
    periodic = []
    while True:
        sc.skip_ws()
        if sc.p >= sc.n:
            break
        if sc.b[sc.p:sc.p + 1] != b"(":
            # stray bytes between sections (should not happen); skip a line
            q = sc.b.find(b"\n", sc.p)
            sc.p = sc.n if q < 0 else q + 1
            continue
        start = sc.p
        sc.p += 1
        tok = sc.read_token()
        try:
            index = int(tok)
        except ValueError:
            sc.p = start
            sc.skip_balanced()
            continue
        if index == 0 or index == 1:
            sc.skip_ws()
            if sc.peek() == b'"':
                sc.skip_string()
            sc.p = start
            sc.skip_balanced()
        elif index == 2:
            dim = int(sc.read_token())
            sc.expect(b")")
        elif index in (10, 2010, 3010):
            h = sc.read_header()
            zid, first, last, ntype = (_hex(t) for t in h[:4])
            nd = _hex(h[4]) if len(h) > 4 else (dim or 3)
            if dim is None:
                dim = nd
            if zid == 0:
                total_nodes = last - first + 1 if last >= first else 0
                sc.expect(b")")
                continue
            count = last - first + 1
            if sc.peek() == b")":
                sc.expect(b")")
                continue
            if index == 10:
                data = sc.ascii_block()
                xyz = np.array(data.split(), dtype=np.float64)
                sc.expect(b")")
            else:
                sc.expect(b"(")
                dt = np.float32 if index == 2010 else np.float64
                nbytes = count * nd * np.dtype(dt).itemsize
                xyz = np.frombuffer(sc.b, dtype=dt, count=count * nd, offset=sc.p).astype(np.float64)
                sc.p += nbytes
                sc.end_binary(index)
            node_blocks.append((first, xyz.reshape(count, nd)))
        elif index in (12, 2012, 3012):
            h = sc.read_header()
            zid, first, last, ztype = (_hex(t) for t in h[:4])
            etype = _hex(h[4]) if len(h) > 4 else 0
            if zid == 0:
                total_cells = last - first + 1 if last >= first else 0
                sc.expect(b")")
                continue
            if sc.peek() == b"(":
                if index == 12:
                    sc.ascii_block()
                    sc.expect(b")")
                else:
                    sc.expect(b"(")
                    sc.p += 4 * (last - first + 1)
                    sc.end_binary(index)
            else:
                sc.expect(b")")
            cell_blocks.append((zid, first, last, ztype, etype))
        elif index in (13, 2013, 3013):
            h = sc.read_header()
            vals5 = [_hex(t) for t in h[:5]] + [0] * (5 - len(h[:5]))
            zid, first, last, bctype, ftype = vals5
            if zid == 0:
                total_faces = last - first + 1 if last >= first else 0
                sc.expect(b")")
                continue
            nf = last - first + 1
            if sc.peek() == b")":
                sc.expect(b")")
                continue
            if index == 13:
                vals = _parse_hex_ints(sc.ascii_block())
                sc.expect(b")")
            else:
                sc.expect(b"(")
                if ftype in (2, 3, 4):
                    n_int = nf * (ftype + 2)
                    vals = np.frombuffer(sc.b, dtype=np.int32, count=n_int, offset=sc.p).astype(np.int64)
                    sc.p += 4 * n_int
                else:
                    # mixed/polygonal: walk the records to find the length
                    p0 = sc.p
                    out = []
                    p = p0
                    for _ in range(nf):
                        k = int(np.frombuffer(sc.b, dtype=np.int32, count=1, offset=p)[0])
                        rec = np.frombuffer(sc.b, dtype=np.int32, count=k + 3, offset=p)
                        out.append(rec)
                        p += 4 * (k + 3)
                    vals = np.concatenate(out).astype(np.int64)
                    sc.p = p
                sc.end_binary(index)
            counts, nodes, c0, c1, used = _faces_from_ints(vals, nf, ftype)
            if used != vals.size:
                raise FluentMeshError(f"face zone {zid}: expected {used} integers, found {vals.size}")
            face_blocks.append((zid, first, bctype, counts, nodes, c0, c1))
        elif index in (18, 2018):
            h = sc.read_header()
            if sc.peek() == b"(":
                if index == 18:
                    vals = _parse_hex_ints(sc.ascii_block())
                    sc.expect(b")")
                else:
                    sc.expect(b"(")
                    cnt = 2 * (_hex(h[1]) - _hex(h[0]) + 1)
                    vals = np.frombuffer(sc.b, dtype=np.int32, count=cnt, offset=sc.p).astype(np.int64)
                    sc.p += 4 * cnt
                    sc.end_binary(index)
                periodic.append((h, vals.reshape(-1, 2) - 1))
            else:
                sc.expect(b")")
        elif index in (39, 45):
            sc.expect(b"(")
            q = sc.b.find(b")", sc.p)
            toks = sc.b[sc.p:q].split()
            sc.p = q + 1
            zid = int(toks[0])
            ztype = toks[1].decode(errors="replace")
            zname = toks[2].decode(errors="replace") if len(toks) > 2 else f"zone-{zid}"
            zone_info[zid] = (ztype, zname)
            # remaining (boundary-condition data or "()") then the closing paren
            while sc.peek() == b"(":
                sc.skip_balanced()
            sc.expect(b")")
        else:
            sc.p = start
            if index >= 2000:
                q = sc.b.find(b"End of Binary Section", sc.p)
                if q < 0:
                    raise FluentMeshError(f"cannot skip binary section {index}")
                sc.p = q
                sc.read_token(); sc.read_token(); sc.read_token(); sc.read_token()
                sc.read_token()
                sc.expect(b")")
            else:
                sc.skip_balanced()

    if dim is None or not node_blocks or not face_blocks:
        raise FluentMeshError("file does not contain a complete Fluent mesh (nodes/faces missing)")

    # --- assemble nodes (1-based Fluent indices) ---
    n_nodes = max(total_nodes, max(f + len(x) - 1 for f, x in node_blocks))
    nodes = np.zeros((n_nodes, dim))
    for first, xyz in node_blocks:
        nodes[first - 1:first - 1 + len(xyz)] = xyz[:, :dim]

    # --- assemble faces in file order of their indices ---
    face_blocks.sort(key=lambda fb: fb[1])
    counts = np.concatenate([fb[3] for fb in face_blocks])
    fnodes = np.concatenate([fb[4] for fb in face_blocks]) - 1
    c0 = np.concatenate([fb[5] for fb in face_blocks])
    c1 = np.concatenate([fb[6] for fb in face_blocks])
    fz = np.concatenate([np.full(len(fb[3]), fb[0]) for fb in face_blocks])
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    # boundary faces stored with c0 == 0: swap so that the owner always exists
    swap = (c0 == 0) & (c1 != 0)
    if swap.any():
        for f in np.nonzero(swap)[0]:
            s, e = ptr[f], ptr[f + 1]
            fnodes[s:e] = fnodes[s:e][::-1].copy()
        c0[swap], c1[swap] = c1[swap], 0
    owner = c0 - 1
    neighbour = np.where(c1 > 0, c1 - 1, -1)
    n_cells = max(total_cells, int(owner.max()) + 1, int(neighbour.max()) + 1)

    zones: Dict[int, Zone] = {}
    start = 0
    for zid, first, bctype, cnts, *_ in face_blocks:
        ztype, zname = zone_info.get(zid, (BC_CODES.get(bctype, "wall"), f"zone-{zid}"))
        zones[zid] = Zone(zid, zname, ztype, "face", start, start + len(cnts) - 1)
        start += len(cnts)
    cell_zone = np.zeros(n_cells, dtype=np.int64)
    for zid, first, last, ztype, etype in cell_blocks:
        cell_zone[first - 1:last] = zid
        zt, zn = zone_info.get(zid, ("fluid" if ztype == 1 else "solid", f"zone-{zid}"))
        zones[zid] = Zone(zid, zn, zt, "cell", first - 1, last - 1)
    if not cell_blocks:
        zones[1] = Zone(1, "fluid", "fluid", "cell", 0, n_cells - 1)
        cell_zone[:] = 1
    mesh = UnstructuredMesh(dim=dim, nodes=nodes, face_ptr=ptr, face_nodes=fnodes.astype(np.int64),
                            owner=owner.astype(np.int64), neighbour=neighbour.astype(np.int64),
                            face_zone=fz.astype(np.int64), n_cells=n_cells, cell_zone=cell_zone,
                            zones=zones, name=name)
    mesh.periodic_pairs = periodic
    return mesh


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

def _zone_sym(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-.:]", "-", name).strip("-")
    return s or "zone"


def _cell_types(mesh: UnstructuredMesh) -> np.ndarray:
    """Fluent element type per cell from its face signature (7 = polyhedral/polygonal)."""
    nc = mesh.n_cells
    fcnt = np.diff(mesh.face_ptr)
    inter = mesh.neighbour >= 0
    nfaces = np.bincount(mesh.owner, minlength=nc) + np.bincount(mesh.neighbour[inter], minlength=nc)
    if mesh.dim == 2:
        return np.where(nfaces == 3, 1, np.where(nfaces == 4, 3, 7))
    tri = (np.bincount(mesh.owner, fcnt == 3, minlength=nc)
           + np.bincount(mesh.neighbour[inter], (fcnt == 3)[inter], minlength=nc)).astype(int)
    quad = (np.bincount(mesh.owner, fcnt == 4, minlength=nc)
            + np.bincount(mesh.neighbour[inter], (fcnt == 4)[inter], minlength=nc)).astype(int)
    t = np.full(nc, 7)
    t[(nfaces == 4) & (tri == 4)] = 2
    t[(nfaces == 6) & (quad == 6)] = 4
    t[(nfaces == 5) & (tri == 4) & (quad == 1)] = 5
    t[(nfaces == 5) & (tri == 2) & (quad == 3)] = 6
    return t


def write_fluent_mesh(mesh: UnstructuredMesh, path: str, binary: bool = False, title: str = "FlowTrace") -> None:
    """Write an UnstructuredMesh as a Fluent .msh file (ASCII, or binary double precision)."""
    mesh = sort_faces_by_zone(mesh)
    dim = mesh.dim
    out = io.BytesIO()
    w = out.write

    def hx(v: int) -> str:
        return format(int(v), "x")

    w(f'(0 "{title} mesh: {mesh.n_cells} cells")\n'.encode())
    w(f"(2 {dim})\n".encode())
    nn, nc, nf = mesh.n_nodes, mesh.n_cells, mesh.n_faces
    w(f"(10 (0 1 {hx(nn)} 0 {dim}))\n".encode())
    node_zone = 1
    if binary:
        w(f"(3010 ({hx(node_zone)} 1 {hx(nn)} 1 {dim})(".encode())
        w(np.ascontiguousarray(mesh.nodes[:, :dim], dtype="<f8").tobytes())
        w(b")\nEnd of Binary Section 3010)\n")
    else:
        w(f"(10 ({hx(node_zone)} 1 {hx(nn)} 1 {dim})(\n".encode())
        np.savetxt(out, mesh.nodes[:, :dim], fmt="%.15g")
        w(b"))\n")
    w(f"(12 (0 1 {hx(nc)} 0))\n".encode())
    # cell zones: element type 0 (mixed) is always valid; declare per zone
    counts = np.diff(mesh.face_ptr)
    for z in mesh.cell_zones():
        cells = np.nonzero(mesh.cell_zone == z.id)[0]
        if cells.size == 0:
            continue
        if cells[-1] - cells[0] + 1 != cells.size:
            raise ValueError("cell zones must be contiguous to write a Fluent mesh")
        types = _cell_types(mesh)[cells]
        etype = int(types[0]) if np.all(types == types[0]) else 0
        w(f"(12 ({hx(z.id)} {hx(cells[0] + 1)} {hx(cells[-1] + 1)} 1 {hx(etype)})".encode())
        if etype == 0:
            w(b"(\n" + " ".join(hx(t) for t in types).encode() + b"\n))\n")
        else:
            w(b")\n")
    w(f"(13 (0 1 {hx(nf)} 0))\n".encode())
    for z in sorted(mesh.face_zones(), key=lambda z: z.first):
        fids = np.arange(z.first, z.last + 1)
        cnt = counts[fids]
        k = int(cnt[0]) if np.all(cnt == cnt[0]) and cnt[0] in (2, 3, 4) else 0
        if dim == 3 and k == 2:
            k = 0
        bc = BC_NAMES.get(z.type, 3)
        c0 = mesh.owner[fids] + 1
        c1 = np.where(mesh.neighbour[fids] >= 0, mesh.neighbour[fids] + 1, 0)
        header = f"({hx(z.id)} {hx(fids[0] + 1)} {hx(fids[-1] + 1)} {hx(bc)} {hx(k)})"
        if binary:
            w(f"(2013 {header}(".encode())
        else:
            w(f"(13 {header}(\n".encode())
        rows = []
        for f, kk, a, b in zip(fids, cnt, c0, c1):
            nodes = mesh.face_nodes[mesh.face_ptr[f]:mesh.face_ptr[f + 1]] + 1
            rec = ([int(kk)] if k == 0 else []) + nodes.tolist() + [int(a), int(b)]
            rows.append(rec)
        if binary:
            flat = np.concatenate([np.asarray(r, dtype="<i4") for r in rows]) if rows else np.zeros(0, "<i4")
            w(flat.tobytes())
            w(b")\nEnd of Binary Section 2013)\n")
        else:
            w(("\n".join(" ".join(hx(v) for v in r) for r in rows) + "\n))\n").encode())
    for z in sorted(mesh.zones.values(), key=lambda z: z.id):
        w(f"(45 ({z.id} {z.type} {_zone_sym(z.name)})())\n".encode())
    data = out.getvalue()
    if str(path).endswith(".gz"):
        with gzip.open(path, "wb") as fh:
            fh.write(data)
    else:
        with open(path, "wb") as fh:
            fh.write(data)
