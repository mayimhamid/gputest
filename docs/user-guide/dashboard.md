# Dashboard

`flowtrace` (or `start_flowtrace.bat` on Windows) opens a local web dashboard at
<http://127.0.0.1:8765>. It runs offline, loads no external scripts and binds to localhost only.

## Layout

* **Case** – built-in cases, or drag and drop a Fluent mesh (`.msh`, `.cas`, `.cas.h5`). Zones are
  listed with their Fluent types and mapped to editable boundary conditions. The mesh quality
  summary is shown on upload.
* **Solver** – transient projection or steady SIMPLEC, schemes, turbulence model,
  backend/precision (or *Auto*: measure and choose), checkpoints, memory estimate.
* **Hardware** – everything the probe found, plus one-click benchmarks.
* **Fluent** – download the hand-off package (`.msh`, `.ip`, `.jou`, PyFluent `.py`, `.csv`).
* **Viewport** – contours, streamlines, vectors, mesh edges and a hover read-out. The right column
  shows Fluent-style residuals, performance, monitors and the execution plan.

## Projects

*Save project* stores the current case, solver settings and, after a run, the result summary in
`<workdir>/projects/<name>.json`. *Open project* lists saved projects and restores one. Projects
are plain JSON, so they can be kept under version control next to the mesh.

## Run report

*Export report* downloads a self-contained HTML file for the finished run: the case, the
execution plan, residual history, validation checks, mass and heat balance, and a snapshot of
the field. Attach it to a design review or a ticket; it needs no FlowTrace installation to view.

## Language, theme, accessibility

* **فارسی / English** – the interface language switch at the top right. Persian switches the
  layout to right-to-left. The choice is remembered in the browser.
* **Theme** – light, dark, or follow the system setting.
* Every control has a label. Tabs and buttons work from the keyboard with visible focus, and
  status changes are announced to screen readers. Colour maps include perceptually uniform
  options (Viridis), so values do not depend on hue alone.
