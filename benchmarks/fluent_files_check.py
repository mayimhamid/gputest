"""Check FlowTrace's Fluent readers and writers against files written by Ansys Fluent itself.

Fetches the PyFluent mixing-elbow example (ansys/example-data, MIT licence; about 9 MB,
SHA-256 verified) into ~/.flowtrace/fluent_examples, unless --dir already holds it, and runs
flowtrace.validation.fluent_files.check. Each property's criterion is fixed in that module.

Run:  python benchmarks/fluent_files_check.py [--dir DIR]
Writes benchmarks/results/fluent_files_check.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    from flowtrace import __version__
    from flowtrace.validation.fluent_files import DEFAULT_DIR, FILES, LICENSE_NOTE, check, download, find, verdicts

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=None, help=f"folder with the reference files (default {DEFAULT_DIR})")
    args = ap.parse_args()
    paths = find(args.dir) or download(args.dir or DEFAULT_DIR)
    result = check(paths)
    ok = verdicts(result)
    meta = {"date": dt.datetime.now().isoformat(timespec="seconds"), "flowtrace": __version__,
            "python": platform.python_version(), "os": platform.platform(),
            "files": {n: {"sha256": FILES[n]} for n in FILES}, "source": LICENSE_NOTE}
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fluent_files_check.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "result": result, "verdicts": ok}, fh, indent=1, default=str)
    cas = result["mixing_elbow.cas.h5"]
    L = [f"Measured {meta['date']} with FlowTrace {__version__}, Python {meta['python']}, {meta['os']}.",
         f"Reference files: {LICENSE_NOTE}; SHA-256 in `flowtrace/validation/fluent_files.py`.", "",
         "| Property | Measured | Result |", "|---|---|---|"]
    measured = {
        "3-D polyhedral mesh read (17,822 cells, 91,581 faces)":
            f"{cas['cells']:,} cells, {cas['faces']:,} faces with {cas['nodes_per_face'][0]}-{cas['nodes_per_face'][-1]} nodes",
        "zone types": ", ".join(f"{k} ({v})" for k, v in cas["zones"].items()),
        "no face re-oriented on read": f"{result['mixing_elbow.msh.h5']['faces_reoriented']} / {cas['faces_reoriented']}",
        "every cell closes (|sum S| / sum |S| < 1e-12)":
            f"{max(result['mixing_elbow.msh.h5']['closure_max'], cas['closure_max']):.2e}",
        "mesh and case: same topology, volumes in the ratio 0.0254^3":
            f"same topology {result['same_topology']}, ratio {result['volume_ratio_cas_over_msh']:.9g}",
        "written back: Fluent's node order and c0/c1 for every face":
            f"{result['roundtrip_same_cyclic_node_order']:,} / {result['roundtrip_same_c0_c1']:,} "
            f"of {result['roundtrip_faces']:,}",
        "data mapped to cells: outlet / inlets mass flow within 10 %": f"{result['outlet_over_inlets']:.4f}",
    }
    for k, passed in ok.items():
        L.append(f"| {k} | {measured.get(k, '')} | {'pass' if passed else '**FAIL**'} |")
    L += ["", "Not covered: Fluent reading files that FlowTrace wrote (needs a licensed Fluent).", ""]
    with open(os.path.join(out_dir, "fluent_files_check.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
