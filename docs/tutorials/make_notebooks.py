"""Build (and optionally execute) the tutorial notebooks.

    python docs/tutorials/make_notebooks.py            # write the .ipynb files
    python docs/tutorials/make_notebooks.py --execute  # ... and run them (outputs stored for the docs)
"""

import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    return n


T1 = [
    """md:# Cavity flow, manufactured solutions and grid convergence

This tutorial runs the structured (MAC) solver on the lid-driven cavity, compares it with
Ghia, Ghia & Shin (1982), then checks the order of accuracy with a manufactured solution and
estimates the discretisation error of a quantity of interest with the Grid Convergence Index.""",
    """import numpy as np
import matplotlib.pyplot as plt
from flowtrace.hardware.backend import get_backend
from flowtrace.structured.case import lid_driven_cavity
from flowtrace.structured.solver import MACSolver
from flowtrace.validation.reference import ghia, compare_ghia

be = get_backend("numba")   # or "numpy" / "cupy"
s = MACSolver(lid_driven_cavity(re=100, n=48), backend=be)
while not s.finished:
    s.step()
    if s.step_count % 10 == 0:
        s.residuals()
print(f"{s.step_count} steps, converged: {s.converged}, max |div u| = {s.max_divergence():.1e}")""",
    """(y, u), (x, v) = s.centerline()
(gy, gu), (gx, gv) = ghia(100)
print(compare_ghia(100, y, u, x, v))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(u, y, label="FlowTrace 48x48"); ax[0].plot(gu, gy, "o", mfc="none", label="Ghia et al. 1982")
ax[0].set_xlabel("u(x=0.5)"); ax[0].set_ylabel("y"); ax[0].legend()
ax[1].plot(x, v); ax[1].plot(gx, gv, "o", mfc="none"); ax[1].set_xlabel("x"); ax[1].set_ylabel("v(y=0.5)")
fig.tight_layout()""",
    """md:## Order of accuracy with a manufactured solution

`run_mac` adds the residual of a known smooth field as a source term and measures the error on
successive grids. Second order means the error drops by 4x when the grid spacing halves.""",
    """from flowtrace.validation.mms import run_mac
r = run_mac((16, 32, 64), backend="numba")
for q in r.errors:
    print(f"{q}: L2 errors {['%.2e' % e for e in r.errors[q]]}, observed order {['%.2f' % p for p in r.orders[q]]}")""",
    """md:## Grid Convergence Index

The mean Nusselt number of the differentially heated cavity at Ra = 1e4 on three grids, with the
Richardson-extrapolated value and the GCI (Celik et al. 2008).""",
    """from flowtrace.structured.case import natural_convection
from flowtrace.validation.gci import gci
vals, grids = [], (24, 48, 96)
for n in grids:
    sn = MACSolver(natural_convection(ra=1e4, n=n), backend=be)
    while not sn.finished:
        sn.step()
        if sn.step_count % 10 == 0:
            sn.residuals()
    vals.append(sn.nusselt("left"))
g = gci(vals[2], vals[1], vals[0], 1 / 96, 1 / 48, 1 / 24)
print("Nu on 24/48/96:", [round(v, 4) for v in vals])
print(f"observed order {g.observed_order:.2f}, extrapolated Nu {g.extrapolated:.4f} "
      f"(de Vahl Davis 1983: 2.243), GCI_fine {100 * g.gci_fine:.2f} %")""",
]

T2 = [
    """md:# Turbulent channel flow: k-omega SST against DNS

A fully developed channel at Re_tau = 392 is modelled as a periodic slab (4 cells in x) driven by
a constant pressure gradient. With u_tau = 1 and the half-height 1, the wall shear stress must
be exactly 1 at equilibrium, and U is directly U+. The reference is the DNS of Moser, Kim &
Mansour (1999).""",
    """import numpy as np
import matplotlib.pyplot as plt
from flowtrace.validation.turbulent import run_channel, DNS_YPLUS, DNS_UPLUS, RE_TAU

sst = run_channel("k-omega-sst")
ke = run_channel("k-epsilon-standard")
for r in (sst, ke):
    print(f"{r['model']:<20} iterations {r['iterations']:>5}  wall shear {r['tau_w']:.4f}  "
          f"first-cell y+ {r['y_plus_first']:.2f}  max |U+ - DNS| {r['max_abs_err_pct']:.1f} % (y+ >= {r['compared_from_yplus']:.0f})")""",
    """yp = np.logspace(-0.5, np.log10(RE_TAU), 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(DNS_YPLUS, DNS_UPLUS, "ko", mfc="none", label="DNS (MKM 1999)")
for r, st in ((sst, "-"), (ke, "--")):
    ax.semilogx(np.array(r["profile_y"]) * RE_TAU, r["profile_u"], st, label=r["model"])
ax.semilogx(yp[yp < 12], yp[yp < 12], ":", color="gray", label="u+ = y+")
ax.semilogx(yp[yp > 20], np.log(yp[yp > 20]) / 0.41 + 5.2, ":", color="gray", label="log law")
ax.set_xlabel("y+"); ax.set_ylabel("U+"); ax.legend(); fig.tight_layout()""",
    """md:SST resolves the viscous sublayer (first cell at y+ ~ 0.25). The k-epsilon run uses scalable
wall functions with the first cell at y+ ~ 25, so it is only compared from y+ = 30 outwards.""",
]

T3 = [
    """md:# Fluent warm start from a Python script

The same steps as `flowtrace fluent-warmstart`, from Python: read a Fluent mesh, check its
quality, solve, and write the interpolation file plus the scripts that continue *your* case in
Fluent. A small mesh is generated here so the notebook runs anywhere. With a real case, pass the
path of your `.msh`/`.cas.h5` file instead.""",
    """import os, tempfile, json
import numpy as np
from flowtrace.mesh.unstructured import cartesian_mesh_2d
from flowtrace.mesh.fluent_msh import write_fluent_mesh
from flowtrace.mesh.quality import mesh_quality

work = tempfile.mkdtemp(prefix="flowtrace_ws_")
mesh = cartesian_mesh_2d(40, 10, 4.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                         types=("velocity-inlet", "pressure-outlet", "wall", "wall"), triangulate=True, warp=0.02)
case_file = os.path.join(work, "duct.msh")
write_fluent_mesh(mesh, case_file)
print(mesh_quality(mesh).summary())""",
    """from flowtrace.fluent.warmstart import run_warmstart
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase

case = UnstructuredCase(boundaries={"inlet": BoundarySpec("inlet", normal_speed=1.0)}, viscosity=0.02,
                        max_iter=2000, tol=1e-6, backend="numpy")
report = run_warmstart(case_file, case, os.path.join(work, "ws"), log=lambda *a: None)
print(f"FlowTrace: {report['iterations']} iterations, converged: {report['converged']}, {report['solve_seconds']:.1f} s")
for k, v in report["files"].items():
    print(f"  {k:<7} {os.path.basename(v)}")""",
    """print(open(report["files"]["jou"]).read())""",
    """md:## How much does the warm start save?

Fluent itself needs a licence, so the measurement below uses FlowTrace's own SIMPLEC solver as a
proxy. The `.ip` file is interpolated onto a *different* mesh. The cold run is converged tightly
and its final field is the reference answer; both runs are counted until the velocity field is
within 10⁻⁴ (relative, max norm) of that answer.""",
    """from flowtrace.fluent import warmstart_study as W
W.CASES["tutorial"] = {"title": "cavity Re=100", "preset": "cavity", "params": {"re": 100, "n": 48},
                       "target": {"kind": "tri", "n": 32}}
r = W.run_proxy_case("tutorial", os.path.join(work, "study"), backend="numpy")
print(f"to the converged field: cold start {r.cold_iterations} iterations, warm start {r.warm_iterations} "
      f"iterations (x{r.speedup:.1f}); warm field within {r.max_velocity_difference:.1e} of the cold solution")""",
]


def main():
    books = {"01_cavity_and_verification.ipynb": T1, "02_turbulent_channel.ipynb": T2,
             "03_fluent_warmstart.ipynb": T3}
    for name, cells in books.items():
        n = nb(cells)
        path = os.path.join(HERE, name)
        if "--execute" in sys.argv:
            from nbclient import NotebookClient

            NotebookClient(n, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": os.path.dirname(HERE)}}).execute()
        nbf.write(n, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
