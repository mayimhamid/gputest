# FlowTrace — heterogeneous, hardware-aware CFD solver (Ansys Fluent–ready)

> **فارسی** پایین‌تر آمده است · [English](#english)

FlowTrace profiles the machine it runs on, **measures** every available compute
backend on the actual case, places each stage of the solve (stencils, pressure
solve, reductions, I/O, visualisation) where it is fastest, and exchanges meshes
and solutions with **Ansys Fluent** (`.msh` in/out, `.ip` warm-start files,
`.prof` profiles, journals, PyFluent scripts). It ships with a professional local
web dashboard.

<a name="english"></a>

## What is in the box

| Layer | What it does |
|---|---|
| **Hardware layer** (`flowtrace/hardware`) | Probes CPU (model, cores, SIMD), RAM, disk, battery, NVIDIA GPUs (VRAM, compute capability, Tensor/RT cores, FP64:FP32 ratio), libraries. Micro-benchmarks bandwidth, GEMM, DCT, stencils, launch latency, PCIe. **Memory budget** before allocating. **Planner** that times real solver steps on every backend and explains every placement. |
| **Structured solver** (`flowtrace/structured`) | 2-D incompressible Navier–Stokes + Boussinesq energy on a staggered (MAC) grid. QUICK / van Leer TVD / upwind / central convection, Adams–Bashforth 2, Crank–Nicolson diffusion where it pays off, exact pressure projection. Walls (moving), inlets (uniform/parabolic), convective outlets, symmetry, periodic, immersed obstacles. FMG grid sequencing for steady cases (like Fluent's FMG initialisation). |
| **Linear algebra** (`flowtrace/linalg`) | Direct separable Poisson/Helmholtz solver: per axis a fast transform (DCT-II/IV, DST-I/II/IV, FFT) or dense generalised eigen-decomposition applied as GEMM (non-uniform grids, any BC mix). Geometric multigrid-PCG for masked domains. Smoothed-aggregation AMG. Backend-agnostic CG/BiCGSTAB. |
| **Unstructured solver** (`flowtrace/unstructured`) | Fluent's pressure-based segregated algorithm: SIMPLE / SIMPLEC, Rhie–Chow, bounded second-order upwind, non-orthogonal correction, least-squares gradients, AMG-PCG pressure correction, energy equation, Fluent-style scaled residuals. Any cell shape, 2-D and 3-D. **Steady or transient** (Euler/BDF2 with inner iterations and Choi's Rhie–Chow correction). |
| **Physical models** | **k-ω SST** (Menter 2003, Fluent's form) with a y+-insensitive wall treatment (Spalding wall shear, blended ω wall value, k wall-function switch at y+ = 11.25), inlet turbulence from intensity + viscosity ratio, turbulent Prandtl number for heat transfer. **P-1 radiation** with Marshak wall emissivity. **Translational periodic** zones driven by a mean pressure gradient. Boussinesq buoyancy. **Species transport** (passive, non-reacting; laminar + turbulent Schmidt diffusion). |
| **Reports** | Per wall zone: pressure/viscous forces, Cd/Cl, y+ (avg/max), skin friction, convective and radiative heat flow; periodic mass flow; boundary mass balance; Cf/Cp/y+ distributions along a wall. |
| **Fluent interop** (`flowtrace/mesh`, `flowtrace/fluent`) | Reads/writes Fluent `.msh` (ASCII, binary single/double, gzip, legacy ASCII `.cas`), repairs left-handed faces, keeps zone names/types. Writes interpolation (`.ip` v3), profile (`.prof`), TUI journal, PyFluent script; optional live PyFluent bridge. |
| **Runtime & UI** (`flowtrace/runtime`, `flowtrace/ui`) | Threaded jobs, event hub, throttled/downsampled frames, asynchronous checkpoints with back-pressure, dashboard over Server-Sent Events (no external JS, works offline). |

## Install

```bash
pip install -e .                 # numpy + scipy only
pip install numba psutil         # multi-core JIT kernels (recommended)
pip install cupy-cuda12x         # NVIDIA GPU backend (match your CUDA driver: cupy-cuda11x / cupy-cuda12x)
pip install pyamg                # faster AMG setup on the CPU (optional)
pip install ansys-fluent-core    # live link to a licensed Fluent (optional)
```

Windows laptops work the same way (Python ≥ 3.9). No compiler is needed.

## Use

```bash
python -m flowtrace                  # dashboard at http://127.0.0.1:8765
python -m flowtrace probe            # what FlowTrace sees on this machine
python -m flowtrace bench            # measured throughput of every backend
python -m flowtrace run cavity --re 1000 --n 256 --export out/     # + Fluent package
python -m flowtrace run cylinder --re 100 --n-per-d 24 --t-end 150
python -m flowtrace mesh my_case.msh                                # inspect a Fluent mesh
python -m flowtrace solve my_case.msh --bc inlet=inlet:1.0 --bc outlet=outlet --mu 1.8e-5 --rho 1.2 \
       --turbulence k-omega-sst --intensity 0.05 --export out/
python -m flowtrace solve duct.msh --periodic inflow,outflow --dpdx -1,0 --turbulence sst   # periodic flow
python -m flowtrace solve furnace.msh --energy --radiation p1 --absorption 0.5 --bc wall=wall::T1200,e0.8
python -m flowtrace solve mixer.msh --unsteady --dt 1e-3 --t-end 0.5 --inner 20 \
       --species tracer --bc jet=inlet:2.0:tracer=1
python -m flowtrace case turbulent_channel --param re_tau=395     # built-in Fluent-style cases:
python -m flowtrace case flat_plate                                #   flat_plate, radiation_slab,
python -m flowtrace run step --re 20000 --turbulence k-omega-sst  #   periodic_taylor_green
python -m flowtrace validate         # validation suite
```

### Dashboard

* **Case** – built-in cases (lid-driven cavity, natural convection, channel,
  cylinder wake, backward-facing step, Rayleigh–Bénard, Taylor–Green, and the
  Fluent-style SST channel, SST flat plate, P-1 slab, transient periodic
  Taylor–Green) or **drag-and-drop a Fluent mesh**: zones are listed with their
  Fluent types and mapped to boundary conditions you can edit (`T=…`, `q=…`,
  emissivity `e=…`, periodic pairs).
* **Solver** – transient projection or steady SIMPLEC, schemes, CFL, pressure
  solver, FMG, backend/precision (or *Auto*: measure and choose), checkpoints,
  memory estimate; **Models**: laminar / k-ω SST, inlet turbulence, P-1
  radiation, periodic pressure gradient, transient time stepping.
* **Hardware** – everything the probe found, plus one-click benchmarks.
* **Fluent** – download the hand-off package (`.msh`, `.ip`, `.jou`, PyFluent
  `.py`, `.csv`, README).
* Viewport: contours (Turbo/Viridis/Fluent rainbow/cool-warm), streamlines,
  vectors, mesh edges, hover read-out; fields include k, ω, μt/μ, wall
  distance and incident radiation. Right column: Fluent-style residuals,
  performance (steps/s, Mcell/s, time split per stage), monitors (Nu, probes,
  KE), and the **execution plan with the measured reason for every stage**.
  After a run: validation checks and the wall report (forces, Cd, y+, Cf, heat).

## Fluent workflow (warm start)

1. Solve in FlowTrace (fast, any backend), then **Fluent › Download package**.
2. In Fluent: *File › Read › Mesh* `flowtrace.msh` (or your own mesh of the
   same geometry), set models/BCs, *Initialize*, then
   *File › Interpolate › Read and Interpolate* `flowtrace.ip`.
3. Iterate: Fluent starts from a converged field instead of from rest.

Batch: `fluent 2ddp -g -t4 -i run_fluent.jou`, or `python run_pyfluent.py`.
Going the other way, any Fluent `.msh` can be solved in FlowTrace with the same
pressure-based algorithm Fluent uses.

## How the hardware is used (and why)

* The planner **measures**: it builds the real solver on every backend that fits
  in memory and times real steps. On small grids the CPU often wins (GPU launch
  latency); on large grids the GPU does. The UI shows the numbers.
* GPU (CuPy): every field stays resident in VRAM; only scalars (CFL, residuals)
  and ≤4 fps downsampled frames cross PCIe. Consumer GeForce/RTX cards have
  1/32–1/64 FP64 rate, so FlowTrace defaults to FP32 fields there and FP64 on
  data-centre GPUs.
* CPU: Numba-fused stencils and red-black Gauss–Seidel on all cores
  (OpenMP workers set to passive waiting so they do not starve NumPy/SciPy),
  multi-threaded FFTs.
* Pressure: exact O(N log N) transforms or GEMM-based diagonalisation, chosen by
  timing both; multigrid-PCG when obstacles make the operator non-separable.
* Tensor Cores: used only where they keep accuracy (FP64 MMA on data-centre
  GPUs for the dense transforms). TF32 would degrade the pressure solve, so it is
  not enabled. RT cores/OptiX are not used: the radiation model is P-1 (a
  diffusion equation, solved like the pressure); ray-traced models (Monte-Carlo
  / S2S view factors) are where RT cores would pay off — see roadmap.
* Disk: asynchronous checkpoints with a bounded queue (the solver never waits on
  I/O unless the disk is slower than the checkpoint rate).

## Validation (measured in this repository's test suite)

| Case | Result | Reference |
|---|---|---|
| Taylor–Green vortex, error vs exact solution, n = 16→128 | 1.1e-3, 2.8e-4, 7.2e-5, 1.8e-5 (2nd order) | exact |
| Lid-driven cavity Re=100, 64², MAC | max \|Δu\| = 0.004, max \|Δv\| = 0.0085 | Ghia et al. 1982 |
| Lid-driven cavity Re=100, 48², SIMPLEC on Fluent-format quads | max \|Δu\| = 0.0044 | Ghia et al. 1982 |
| Same, skewed triangles (60° non-orthogonality) | max \|Δu\| = 0.0067 | Ghia et al. 1982 |
| Lid-driven cavity **Re=1000**, 128², SIMPLEC | max \|Δu\| = 0.005, max \|Δv\| = 0.010 (3 546 iterations, 354 s on 4 CPU cores) | Ghia et al. 1982 |
| Natural convection Ra=1e3 via SIMPLEC + Boussinesq | Nu ≈ 1.120 | 1.118 |
| Natural convection Ra=1e4, 64² | Nu = 2.251, u_max = 16.19, v_max = 19.64 | 2.243, 16.18, 19.62 (de Vahl Davis 1983) |
| Natural convection Ra=1e5, 64² (FMG) | Nu = 4.569 | 4.519 |
| Channel Re=50 | outlet profile within 0.006 of Poiseuille, dp/dx = 0.238 (0.24) | analytic |
| Projection | max \|div u\| ≈ 1e-14 (fast solver), 1e-11 (multigrid) | 0 |
| SIMPLEC channel / exported step mesh | net mass imbalance ≈ 1e-12 | 0 |
| **k-ω SST** turbulent channel Re_τ = 395 (periodic, y+ ≈ 0.25) | U_b⁺ = 17.62; u⁺(y⁺=100) = 16.9; wall shear = driving force within 0.03% | DNS U_b⁺ = 17.54 (Moser, Kim & Mansour 1999); log law 16.4 |
| **k-ω SST** flat plate Re_L = 5e6 | Cf(x=0.97) = 0.00264 (fine and coarse grid); Cf(1.5) = 0.00248; C_D = 0.00279 | NASA TMR SST 0.00271; Schlichting 0.00250; NASA C_D ≈ 0.00286 |
| **k-ω SST** in the log-law regime (backward-facing step Re_h = 2e4, y+ 20–80) | stable, reattachment x/h ≈ 5.7 on 8 cells/h | — |
| **P-1** radiating slab, τ = 0.1 / 1 / 5 | wall flux error 2e-10 / 4e-8 / 9e-7 | exact P-1: σ(T₁⁴−T₂⁴)/(1+3τ/4) |
| Periodic Poiseuille (unstructured triangles) | profile error 0.08% | analytic |
| Transient Taylor–Green on a periodic unstructured mesh (BDF2) | error 1.2e-2 → 1.5e-3 (n = 16 → 32) | exact |
| Species: two-stream mixing layer | max \|ΔY\| = 0.0047 at x = 1; species out/in = 1 | erf solution |

Run `python -m pytest` (88 tests) or `python -m flowtrace validate`.

**Steady vs transient:** for steady problems use *Steady SIMPLEC* (works for the
built-in cases too): Re=1000 converges in a few thousand iterations, whereas
time-marching has to integrate tens of convective time units. Use the
transient MAC projection for unsteady physics (vortex shedding, convection
rolls) — it is exact in continuity and GPU-friendly.

**GPU code paths** are tested without a GPU through `tests/fake_gpu.py`, a
NumPy-backed stand-in for CuPy (including CuPy's DCT/DST type limits), which
checks that every GPU-specific branch gives the same results as the CPU path.

**Windows:** double-click `start_flowtrace.bat` (installs on first run, then opens
the dashboard). Sample Fluent meshes are in `examples/`.

## Honest limits / roadmap

* The structured (MAC) solver is 2-D and laminar; turbulence, radiation,
  periodicity and transient pressure-based runs live in the unstructured solver
  (which also runs every built-in case via *Steady SIMPLEC*).
* Heat transfer with SST uses laminar conduction at the wall: resolve the wall
  (y+ ≈ 1) for accurate heat flux; momentum is y+-insensitive.
* Only translational periodicity (no rotational periodic zones yet).
* Not yet: LES/DES, compressible (density-based) flow, multiphase (VOF),
  discrete-ordinates or Monte-Carlo radiation (where OptiX/RT cores apply),
  moving/deforming meshes, reacting flows (species are passive).
* The CuPy GPU path uses the same verified array code as the CPU path, but it
  was developed in a container without a GPU: run `python -m flowtrace bench`
  and the planner on your laptop — it only selects the GPU if it measures
  faster, and falls back automatically if anything fails.
* PyFluent API names change between releases; the bridge falls back to TUI
  commands, and the journal uses long-standing TUI commands.

---

<div dir="rtl">

## فارسی

**FlowTrace** یک حلگر CFD ناهمگن و آگاه از سخت‌افزار است که آماده‌ی اتصال به **Ansys Fluent** است.

### چه چیزی ساخته شده

- **شناخت سیستم هدف:** پردازنده (مدل، هسته‌ها، دستورهای SIMD)، RAM، دیسک، باتری، کارت گرافیک NVIDIA (VRAM، Compute Capability، Tensor/RT Core، نسبت FP64 به FP32) و کتابخانه‌ها را شناسایی می‌کند؛ پهنای باند حافظه، GEMM، DCT، استنسیل و تأخیر PCIe را اندازه می‌گیرد.
- **برنامه‌ریز اجرا:** برای هر بک‌اند موجود، خودِ حلگر را روی خودِ مسئله می‌سازد و چند گام واقعی را **زمان‌گیری** می‌کند؛ سریع‌ترین را انتخاب می‌کند و دلیل جای‌گذاری هر مرحله را نشان می‌دهد (فرض «GPU همیشه سریع‌تر است» وجود ندارد).
- **بودجه‌ی حافظه:** پیش از تخصیص، حافظه‌ی لازم را تخمین می‌زند و اگر جا نشود پیشنهاد می‌دهد (FP32، CPU، شبکه‌ی کوچک‌تر).
- **حلگر سازمان‌یافته (MAC):** ناویه–استوکس تراکم‌ناپذیر دوبعدی + انرژی بوزینسک؛ QUICK/TVD، آدامز–بشفورث ۲، کرانک–نیکلسون، تصویر فشار دقیق، دیواره‌ی متحرک، ورودی/خروجی، تقارن، تناوبی، مانع غوطه‌ور، و FMG برای حالت پایا.
- **ریاضیات سریع:** حلگر مستقیم پواسون/هلمهولتز با تبدیل‌های DCT/DST/FFT یا قطری‌سازی متراکم به‌صورت GEMM، چندشبکه‌ای هندسی + CG، AMG تجمیعی هموارشده.
- **حلگر بی‌سازمان (مانند Fluent):** الگوریتم فشار-مبنا SIMPLE/SIMPLEC با Rhie–Chow، بالادست مرتبه‌دوم محدودشده، تصحیح غیرعمودی، گرادیان کمترین مربعات، AMG برای تصحیح فشار، معادله‌ی انرژی و باقیمانده‌های مقیاس‌شده‌ی Fluent؛ دوبعدی و سه‌بعدی با هر نوع سلول؛ **پایا یا گذرا** (Euler/BDF2 با تکرار داخلی).
- **مدل آشفتگی k-ω SST** (منتر ۲۰۰۳، همان فرم Fluent) با تیمار دیواره‌ی غیرحساس به y⁺ (قانون دیواره‌ی اسپالدینگ)، شدت آشفتگی و نسبت ویسکوزیته در ورودی، و عدد پرانتل آشفته برای انتقال حرارت.
- **مدل تابش P-1** با ضریب صدور دیواره، **مرزهای تناوبی** با گرادیان فشار محرک، **انتقال گونه‌ها** (غیرواکنشی)، و **گزارش‌ها:** نیرو، Cd/Cl، y⁺، ضریب اصطکاک و شار حرارتی هر دیواره.
- **اتصال به Fluent:** خواندن و نوشتن `.msh` (ASCII، باینری، gzip، `.cas` قدیمی)، فایل درون‌یابی `.ip` برای شروع گرم، `.prof`، ژورنال TUI و اسکریپت PyFluent.
- **رابط کاربری حرفه‌ای:** داشبورد وب محلی (بدون اینترنت): کانتور، خطوط جریان، بردار، باقیمانده‌ها، کارایی، پایشگرها، طرح اجرا، اعتبارسنجی و خروجی Fluent.

### نصب و اجرا

</div>

```bash
pip install -e .
pip install numba psutil        # پیشنهادی: هسته‌های JIT چندهسته‌ای
pip install cupy-cuda12x        # برای کارت گرافیک NVIDIA
python -m flowtrace             # باز شدن داشبورد در http://127.0.0.1:8765
```

<div dir="rtl">

### گردش کار با Fluent

۱. در FlowTrace حل کنید و از تب **Fluent** بسته‌ی خروجی را بگیرید.
۲. در Fluent فایل `flowtrace.msh` (یا مش خودتان با همان هندسه) را باز کنید، مدل‌ها و شرایط مرزی را تنظیم و Initialize کنید، سپس با *File › Interpolate › Read and Interpolate* فایل `flowtrace.ip` را بخوانید.
۳. Fluent به‌جای شروع از سکون، از میدان همگراشده‌ی FlowTrace ادامه می‌دهد.

همچنین می‌توانید هر مش `.msh` فلوئنت را مستقیم در داشبورد بکشید و رها کنید و با همان الگوریتم فشار-مبنای Fluent حل کنید.

### محدودیت‌های فعلی (صادقانه)

- حلگر سازمان‌یافته (MAC) دوبعدی و آرام است؛ آشفتگی، تابش، تناوب و حل گذرای فشار-مبنا در حلگر بی‌سازمان هستند.
- اعتبارسنجی SST: کانال آشفته Re_τ=395 با خطای ۰٫۵٪ نسبت به DNS و صفحه‌ی تخت با خطای ۲٫۸٪ نسبت به نتایج SST ناسا.
- هنوز پیاده نشده: LES/DES، جریان تراکم‌پذیر، چندفازی (VOF)، تابش DO یا مونت‌کارلو (جایی که OptiX/RT Core معنا پیدا می‌کند)، مش متحرک، احتراق و واکنش شیمیایی.
- مسیر GPU با CuPy همان کد آرایه‌ای تأییدشده‌ی CPU است اما در محیط توسعه GPU وجود نداشت؛ روی لپ‌تاپ خودتان `python -m flowtrace bench` را اجرا کنید — برنامه‌ریز فقط وقتی GPU را انتخاب می‌کند که واقعاً سریع‌تر اندازه‌گیری شود.

</div>
