/* FlowTrace dashboard — dependency-free (works offline). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const fmtBytes = (b) => { if (!b && b !== 0) return "–"; const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; while (Math.abs(b) >= 1024 && i < 4) { b /= 1024; i++; } return b.toFixed(b < 10 ? 2 : 1) + " " + u[i]; };
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v).toPrecision(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function b64f32(s) { const bin = atob(s); const buf = new ArrayBuffer(bin.length); const u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(buf); }
function b64i32(s) { const bin = atob(s); const buf = new ArrayBuffer(bin.length); const u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Int32Array(buf); }

/* ------------------------------------------------------------------ colormaps */
const CMAPS = {
  turbo(x) {
    x = Math.min(1, Math.max(0, x));
    const x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
    const r = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5;
    const g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5;
    const b = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5;
    return [r, g, b];
  },
  viridis(t) {
    t = Math.min(1, Math.max(0, t));
    const c = [[0.2777273272234177, 0.005407344544966578, 0.3340998053353061], [0.1050930431085774, 1.404613529898575, 1.384590162594685],
      [-0.3308618287255563, 0.214847559468213, 0.09509516302823659], [-4.634230498983486, -5.799100973351585, -19.33244095627987],
      [6.228269936347081, 14.17993336680509, 56.69055260068105], [4.776384997670288, -13.74514537774601, -65.35303263337234],
      [-5.435455855934631, 4.645852612178535, 26.3124352495832]];
    return [0, 1, 2].map((k) => { let v = c[6][k]; for (let i = 5; i >= 0; i--) v = c[i][k] + t * v; return v; });
  },
  rainbow(t) { // Fluent-like blue -> cyan -> green -> yellow -> red
    t = Math.min(1, Math.max(0, t));
    const h = (1 - t) * 240 / 60, x = 1 - Math.abs((h % 2) - 1);
    const s = [[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1]][Math.min(4, Math.floor(h))];
    return s;
  },
  coolwarm(t) {
    t = Math.min(1, Math.max(0, t));
    const a = [59, 76, 192], m = [221, 221, 221], b = [180, 4, 38];
    const lerp = (p, q, u) => p.map((v, i) => (v + (q[i] - v) * u) / 255);
    return t < 0.5 ? lerp(a, m, t * 2) : lerp(m, b, (t - 0.5) * 2);
  },
  gray(t) { t = Math.min(1, Math.max(0, t)); return [t, t, t]; },
};
const LUT = {};
function lut(name) {
  if (!LUT[name]) {
    const a = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) { const c = CMAPS[name](i / 255); a[3 * i] = c[0] * 255; a[3 * i + 1] = c[1] * 255; a[3 * i + 2] = c[2] * 255; }
    LUT[name] = a;
  }
  return LUT[name];
}

/* ------------------------------------------------------------------ state */
const S = {
  source: "structured", presets: [], system: null, meshId: null, meshInfo: null, geometry: null, frame: null,
  res: { keys: [], series: {}, x: [], labels: [] }, mon: [], monName: "", resCount: 0, job: null, view: null,
  cellPaths: null, cellCentres: null,
};

/* ------------------------------------------------------------------ UI wiring */
function initTabs() {
  $$("#tabs button").forEach((b) => b.addEventListener("click", () => {
    $$("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
    $$(".tab-body > section").forEach((s) => s.classList.toggle("active", s.dataset.tab === b.dataset.tab));
  }));
  $$("#sourceSeg button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
}
function setSource(src) {
  S.source = src;
  $$("#sourceSeg button").forEach((x) => x.classList.toggle("active", x.dataset.src === src));
  $("#structuredCase").classList.toggle("hidden", src !== "structured");
  $("#unstructuredCase").classList.toggle("hidden", src !== "unstructured");
  $("#structSolverCard").classList.toggle("hidden", src !== "structured");
  $("#unstructSolverCard").classList.toggle("hidden", src !== "unstructured" && $("#mode").value !== "steady-simplec");
}

function renderPresetParams() {
  const p = S.presets.find((x) => x.id === $("#presetSelect").value);
  if (!p) return;
  $("#presetDesc").textContent = p.description;
  const box = $("#presetParams");
  box.innerHTML = "";
  const labels = { re: "Reynolds number", ra: "Rayleigh number", pr: "Prandtl number", n: "Cells per side", nx: "Cells in x", ny: "Cells in y",
    n_per_d: "Cells per diameter", n_per_h: "Cells per step height", length: "Channel length", aspect: "Aspect ratio" };
  for (const [k, v] of Object.entries(p.params)) {
    const l = document.createElement("label");
    l.className = "field";
    l.innerHTML = `${esc(labels[k] || k)}<input type="number" step="any" data-param="${k}" value="${v}">`;
    box.appendChild(l);
  }
  const d = p.defaults;
  $("#tEnd").value = d.t_end ?? "";
  $("#maxSteps").value = d.max_steps;
  $("#steadyTol").value = d.steady_tol ?? "";
  $("#scheme").value = d.scheme;
  $("#cfl").value = d.cfl;
  $("#caseTitle").textContent = p.title;
}

async function loadSystem() {
  const sys = await api("/api/system");
  S.system = sys;
  const p = sys.profile;
  const g = p.best_gpu;
  $("#hwChips").innerHTML = [
    `<span class="chip">CPU <b>${p.physical_cores}C/${p.logical_cores}T</b></span>`,
    `<span class="chip">RAM <b>${fmtBytes(p.ram_total)}</b></span>`,
    g ? `<span class="chip">GPU <b>${esc(g.name)}</b> ${fmtBytes(g.vram_total)}</span>` : `<span class="chip">GPU <b>none usable</b></span>`,
  ].join("");
  const rows = [
    ["OS", p.os], ["CPU", p.cpu_model], ["Cores", `${p.physical_cores} physical / ${p.logical_cores} logical`],
    ["SIMD", (p.simd || []).join(" ") || "–"], ["RAM", `${fmtBytes(p.ram_available)} free of ${fmtBytes(p.ram_total)}`],
    ["Disk (work dir)", `${fmtBytes(p.disk_free)} free`], ["Power", p.on_battery === null ? "n/a" : (p.on_battery ? "battery" : "AC")],
    ["Backends", sys.backends.join(", ")],
    ["Libraries", Object.entries(p.libraries).map(([k, v]) => `${k} ${v || "—"}`).join(" · ")],
  ];
  $("#sysTable").innerHTML = rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
  $("#gpuCount").textContent = `${p.gpus.length} found`;
  $("#gpuBox").innerHTML = p.gpus.length ? p.gpus.map((x) => `
    <table class="kv"><tr><td>Name</td><td>${esc(x.name)} ${x.usable ? '<span class="tag gpu">usable</span>' : '<span class="tag">driver only</span>'}</td></tr>
    <tr><td>VRAM</td><td>${fmtBytes(x.vram_free)} free / ${fmtBytes(x.vram_total)}</td></tr>
    <tr><td>Compute capability</td><td>${esc(x.compute_capability || "?")} · ${x.multiprocessors || "?"} SMs</td></tr>
    <tr><td>Tensor / RT cores</td><td>${x.tensor_cores ? "yes" : "no"} / ${x.rt_cores ? "yes" : "no"}</td></tr>
    <tr><td>FP64 : FP32</td><td>1 : ${Math.round(1 / x.fp64_rate)}</td></tr></table>`).join("")
    : "No NVIDIA GPU detected. FlowTrace runs on the CPU backends (NumPy, Numba).";
  $("#sysNotes").innerHTML = (p.notes || []).map((n) => `• ${esc(n)}`).join("<br>") || "–";
  const be = $("#backend");
  be.innerHTML = '<option value="auto">Auto (measure & choose)</option>' + sys.backends.map((b) => `<option value="${b}">${b}</option>`).join("");
  if (sys.bench) renderBench(sys.bench);
  const pf = sys.pyfluent;
  $("#pyfluentBox").innerHTML = pf.available
    ? `PyFluent ${esc(pf.version)} installed. The exported <span class="mono">run_pyfluent.py</span> launches Fluent, reads the mesh, interpolates the FlowTrace solution and iterates.`
    : `PyFluent not installed. <span class="mono">pip install ansys-fluent-core</span> (needs a licensed Fluent) to drive Fluent from FlowTrace; the .msh/.ip/.jou files work without it.`;
}

function renderBench(b) {
  const r = b.results || {};
  const keys = Object.keys(r);
  if (!keys.length) return;
  const rows = [["Memory bandwidth", "bandwidth_gbs", "GB/s"], ["GEMM FP64", "gemm_gflops_fp64", "GFLOP/s"], ["GEMM FP32", "gemm_gflops_fp32", "GFLOP/s"],
    ["DCT 512²", "dct512_ms", "ms"], ["5-pt stencil", "stencil_mlups", "Mcell/s"], ["Launch latency", "launch_us", "µs"], ["Host→GPU", "h2d_gbs", "GB/s"], ["GPU→Host", "d2h_gbs", "GB/s"]];
  $("#benchBox").innerHTML = `<table class="grid"><tr><th></th>${keys.map((k) => `<th>${k}</th>`).join("")}</tr>` +
    rows.map(([lab, key, unit]) => `<tr><td>${lab} <span class="muted">${unit}</span></td>${keys.map((k) => `<td class="num">${r[k][key] !== undefined ? fmt(r[k][key], 4) : (r[k].error ? "err" : "–")}</td>`).join("")}</tr>`).join("") + "</table>";
}

async function loadPresets() {
  const { presets } = await api("/api/presets");
  S.presets = presets;
  $("#presetSelect").innerHTML = presets.map((p) => `<option value="${p.id}">${esc(p.title)}</option>`).join("");
  $("#presetSelect").addEventListener("change", renderPresetParams);
  renderPresetParams();
}

/* ------------------------------------------------------------------ mesh upload */
function initMeshUpload() {
  const dz = $("#dropzone"), fi = $("#meshFile");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files[0]) uploadMesh(e.dataTransfer.files[0]); });
  fi.addEventListener("change", () => fi.files[0] && uploadMesh(fi.files[0]));
}
async function uploadMesh(file) {
  $("#meshSummary").textContent = `Reading ${file.name} …`;
  try {
    const buf = await file.arrayBuffer();
    const r = await fetch("/api/mesh", { method: "POST", headers: { "X-Filename": file.name }, body: buf });
    const info = await r.json();
    if (!r.ok) throw new Error(info.error);
    S.meshId = info.id; S.meshInfo = info;
    $("#meshSummary").innerHTML = `<b>${esc(info.name)}</b> · ${info.dim}D · ${info.cells.toLocaleString()} cells · ${info.faces.toLocaleString()} faces · volume ${fmt(info.volume, 5)} · read in ${info.read_seconds}s` +
      (info.repaired_faces ? ` · <span style="color:var(--warn)">${info.repaired_faces} left-handed faces repaired</span>` : "");
    renderZoneTable(info);
    $("#zoneCard").classList.remove("hidden");
    $("#propsCard").classList.remove("hidden");
    $("#caseTitle").textContent = info.name;
    if (info.dim === 2) {
      S.geometry = parseGeometry(await api(`/api/mesh/${info.id}`));
      S.frame = null; draw();
    }
  } catch (e) { $("#meshSummary").innerHTML = `<span style="color:var(--err)">${esc(e.message)}</span>`; }
}
function renderZoneTable(info) {
  const t = $("#zoneTable");
  const kinds = ["wall", "inlet", "outlet", "pressure", "symmetry"];
  const dim = info.dim;
  t.innerHTML = info.zones.filter((z) => z.kind === "face" && z.type !== "interior").map((z) => {
    const b = info.boundaries[z.name] || { kind: "wall" };
    return `<tr data-zone="${esc(z.name)}"><td>
      <div class="row" style="justify-content:space-between"><b>${esc(z.name)}</b><span class="muted">${esc(z.type)} · ${z.count} faces</span></div>
      <div class="row" style="margin-top:4px">
        <select data-k="kind" style="width:96px">${kinds.map((k) => `<option ${k === b.kind ? "selected" : ""}>${k}</option>`).join("")}</select>
        <input type="text" data-k="val" title="inlet: normal speed or u,v${dim === 3 ? ",w" : ""}; wall: wall velocity u,v${dim === 3 ? ",w" : ""}; outlet: gauge pressure" placeholder="speed / u,v / p" value="${b.kind === "inlet" ? (b.normal_speed ?? 1) : ""}">
        <input type="text" data-k="therm" title="T=value (fixed temperature) or q=value (heat flux into the domain)" placeholder="T=… / q=…" value="">
      </div></td></tr>`;
  }).join("");
}
function zoneSpecs() {
  const out = {};
  $$("#zoneTable tr[data-zone]").forEach((tr) => {
    const kind = $("select[data-k=kind]", tr).value;
    const val = $("input[data-k=val]", tr).value.trim();
    const th = $("input[data-k=therm]", tr).value.trim();
    const spec = { kind };
    if (val) {
      const nums = val.split(/[ ,;]+/).map(Number).filter((x) => !Number.isNaN(x));
      if (kind === "inlet") { if (nums.length === 1) spec.normal_speed = nums[0]; else spec.velocity = nums; }
      else if (kind === "wall") { if (nums.length > 1) spec.velocity = nums; }
      else if (kind === "outlet" || kind === "pressure") spec.pressure = nums[0] || 0;
    }
    if (th) {
      const m = th.match(/^([TtQq])\s*=\s*([-+0-9.eE]+)/);
      if (m) { spec.thermal = m[1].toLowerCase() === "t" ? "temperature" : "flux"; spec.value = +m[2]; }
      else if (!Number.isNaN(+th)) { spec.thermal = "temperature"; spec.value = +th; }
    }
    out[tr.dataset.zone] = spec;
  });
  return out;
}

/* ------------------------------------------------------------------ run */
function buildSpec() {
  const backend = $("#backend").value, precision = $("#precision").value;
  if (S.source === "structured") {
    const params = {};
    $$("#presetParams input").forEach((i) => { params[i.dataset.param] = Number.isInteger(+i.value) && !i.value.includes(".") && !/e/i.test(i.value) ? parseInt(i.value, 10) : +i.value; });
    const tEnd = $("#tEnd").value.trim(), tol = $("#steadyTol").value.trim();
    return {
      kind: "structured", preset: $("#presetSelect").value, params, mode: $("#mode").value, fmg: $("#fmg").checked,
      checkpoint_seconds: +$("#ckpt").value || 0, field: $("#fieldSelect").value || null,
      max_iter: +$("#maxIter").value, tol: +$("#uTol").value,
      overrides: { backend, precision, scheme: $("#scheme").value, cfl: +$("#cfl").value, poisson: $("#poisson").value,
        max_steps: +$("#maxSteps").value, t_end: tEnd === "" ? null : +tEnd, steady_tol: tol === "" ? null : +tol },
    };
  }
  if (!S.meshId) throw new Error("Load a Fluent mesh first.");
  return {
    kind: "unstructured", mesh_id: S.meshId, checkpoint_seconds: +$("#ckpt").value || 0,
    case: {
      name: S.meshInfo.name, boundaries: zoneSpecs(), density: +$("#uDensity").value, viscosity: +$("#uViscosity").value,
      energy: $("#uEnergy").checked, conductivity: +$("#uConductivity").value, cp: +$("#uCp").value,
      algorithm: $("#algorithm").value, scheme: $("#uScheme").value, relax_u: +$("#relaxU").value, relax_p: +$("#relaxP").value,
      max_iter: +$("#maxIter").value, tol: +$("#uTol").value, backend, precision: precision === "fp32" ? "fp32" : "fp64",
    },
  };
}
async function run() {
  try {
    const spec = buildSpec();
    resetRunView();
    await post("/api/run", spec);
  } catch (e) { logLine({ level: "error", msg: e.message }); }
}
function resetRunView() {
  S.res = { keys: [], series: {}, x: [], labels: [] }; S.mon = []; S.resCount = 0;
  $("#console").innerHTML = ""; $("#validation").innerHTML = '<div class="muted">Running…</div>';
  $("#planTable").innerHTML = '<tr><td class="muted">Measuring backends…</td></tr>';
  drawResiduals(); drawMonitor();
}

/* ------------------------------------------------------------------ events */
function connectEvents() {
  const es = new EventSource("/api/events");
  es.addEventListener("status", (e) => onStatus(JSON.parse(e.data)));
  es.addEventListener("plan", (e) => onPlan(JSON.parse(e.data)));
  es.addEventListener("residuals", (e) => onResiduals(JSON.parse(e.data)));
  es.addEventListener("frame", (e) => onFrame(JSON.parse(e.data)));
  es.addEventListener("geometry", (e) => { S.geometry = parseGeometry(JSON.parse(e.data)); S.cellPaths = null; });
  es.addEventListener("log", (e) => logLine(JSON.parse(e.data)));
  es.onerror = () => { /* the browser reconnects automatically */ };
}
function onStatus(ev) {
  const st = ev.status;
  const pill = $("#statusPill");
  pill.textContent = st + (ev.label && st === "running" ? ` · ${ev.label}` : "");
  pill.className = "pill " + st;
  const busy = st === "running" || st === "planning";
  $("#runBtn").disabled = busy; $("#stopBtn").disabled = !busy;
  if (ev.name) $("#caseTitle").textContent = ev.name;
  if (ev.result) onResult(ev.result);
  if (st === "error") logLine({ level: "error", msg: ev.error });
}
function onPlan(ev) {
  const p = ev.plan;
  $("#planBackend").textContent = `${p.backend} · ${p.precision}`;
  const meas = p.measured_ms_per_step || {};
  let html = p.placements.map((x) => `<tr><td>${esc(x.stage)}<div class="muted">${esc(x.reason)}</div></td><td><span class="tag ${/GPU/i.test(x.device) ? "gpu" : "cpu"}">${esc(x.device)}</span><div class="muted">${esc(x.method)}</div></td></tr>`).join("");
  if (Object.keys(meas).length) html = `<tr><td colspan="2" class="muted">Measured ms/step: ${Object.entries(meas).map(([k, v]) => `<b style="color:var(--text)">${k}</b> ${v}`).join(" · ")}</td></tr>` + html;
  $("#planTable").innerHTML = html;
  const mem = (p.memory || {})[p.backend];
  if (mem) {
    const frac = mem.available ? Math.min(1, mem.required / mem.available) : 0;
    $("#memBox").innerHTML = `<div class="row muted" style="justify-content:space-between"><span>Memory budget (${p.device})</span><span class="mono">${fmtBytes(mem.required)} / ${fmtBytes(mem.available)}</span></div><div class="membar"><div style="width:${(frac * 100).toFixed(1)}%"></div></div>`;
  }
}
const RES_COLORS = { continuity: "#e5e7eb", "x-velocity": "#f87171", "y-velocity": "#34d399", "z-velocity": "#60a5fa", energy: "#fbbf24" };
function onResiduals(ev) {
  const R = S.res;
  S.resCount++;
  R.x.push(S.resCount);
  if (!R.labels.length || R.labels[R.labels.length - 1].label !== ev.label) R.labels.push({ label: ev.label, at: S.resCount });
  for (const [k, v] of Object.entries(ev.scaled)) {
    if (!R.series[k]) { R.series[k] = []; R.keys.push(k); }
    R.series[k][S.resCount - 1] = v;
  }
  $("#sStep").textContent = ev.step.toLocaleString();
  $("#sTime").textContent = ev.dt ? fmt(ev.t, 4) : "–";
  $("#sDt").textContent = ev.dt ? fmt(ev.dt, 3) : "–";
  $("#sRate").textContent = fmt(ev.steps_per_s, 3);
  $("#sMcell").textContent = fmt(ev.mcells_per_s, 3);
  let mname = "", mval = null;
  if (ev.probe) { mname = "probe v"; mval = ev.probe[1]; }
  else if (ev.nusselt !== undefined) { mname = "Nusselt (hot wall)"; mval = ev.nusselt; }
  else if (ev.ke !== undefined) { mname = "kinetic energy"; mval = ev.ke; }
  else if (ev.linear) { mname = "p' AMG-PCG iterations"; mval = ev.linear.pressure_iters; }
  if (mname) { S.monName = mname; S.mon.push(mval); $("#sExtraK").textContent = mname.split(" ")[0]; $("#sExtra").textContent = fmt(mval, 4); $("#monName").textContent = mname; }
  $("#resInfo").textContent = Object.entries(ev.scaled).map(([k, v]) => `${k[0]}:${v.toExponential(1)}`).join(" ");
  if (ev.timers) drawTimers(ev.timers);
  scheduleCharts();
}
let chartPending = false;
function scheduleCharts() { if (chartPending) return; chartPending = true; requestAnimationFrame(() => { chartPending = false; drawResiduals(); drawMonitor(); }); }
function onFrame(ev) {
  S.frame = ev;
  if (ev.mesh === "structured") ev.values = b64f32(ev.data), ev.u = b64f32(ev.u), ev.v = b64f32(ev.v);
  else ev.values = b64f32(ev.data);
  const sel = $("#fieldSelect");
  if (sel.options.length !== ev.fields.length || Array.from(sel.options).some((o, i) => o.value !== ev.fields[i])) {
    sel.innerHTML = ev.fields.map((f) => `<option value="${f}">${f}</option>`).join("");
  }
  sel.value = ev.field;
  $("#empty").classList.add("hidden");
  $("#frameInfo").textContent = ev.mesh === "structured" ? `${ev.grid[0]}×${ev.grid[1]} · step ${ev.step}` : `${ev.cells.toLocaleString()} cells · it ${ev.step}`;
  draw();
}
function onResult(res) {
  const v = res.validation;
  if (v && v.checks) {
    $("#validation").innerHTML = v.checks.map((c) => `<div class="check"><span class="badge ${c.pass ? "pass" : "check"}">${c.pass ? "PASS" : "CHECK"}</span><span>${esc(c.name)}${c.reference !== undefined && c.reference !== null ? ` <span class="muted">(ref ${c.reference})</span>` : ""}</span><span class="val">${fmt(c.value, 4)} <span class="muted">${esc(c.unit || "")}</span></span></div>`).join("");
  } else if (res.mass_balance) {
    const mb = res.mass_balance; const net = Object.values(mb).reduce((a, b) => a + b, 0);
    $("#validation").innerHTML = `<div class="check"><span class="badge ${Math.abs(net) < 1e-6 * Math.max(1e-30, ...Object.values(mb).map(Math.abs)) + 1e-9 ? "pass" : "check"}">MASS</span><span>Net boundary mass flow</span><span class="val">${net.toExponential(2)}</span></div>` +
      Object.entries(mb).map(([k, v]) => `<div class="check"><span class="muted">${esc(k)}</span><span class="val">${fmt(v, 5)}</span></div>`).join("") +
      `<div class="check"><span class="badge ${res.converged ? "pass" : "check"}">${res.converged ? "CONV" : "ITER"}</span><span>${res.iterations} iterations</span></div>`;
  }
  if (res.timers) drawTimers(res.timers);
}
function logLine(ev) {
  const c = $("#console");
  const div = document.createElement("div");
  const m = ev.msg || "";
  div.className = ev.level === "error" ? "error" : ev.level === "warn" ? "warn" : m.startsWith("[plan]") ? "plan" : m.startsWith("[validation]") ? "val" : "";
  div.textContent = m;
  c.appendChild(div);
  while (c.childNodes.length > 400) c.removeChild(c.firstChild);
  c.scrollTop = c.scrollHeight;
}
const TIMER_COLORS = ["#3fb6ff", "#7c5cff", "#34d399", "#fbbf24", "#f87171", "#22d3ee", "#e879f9"];
function drawTimers(t) {
  const entries = Object.entries(t).filter(([, v]) => v > 0);
  const tot = entries.reduce((a, [, v]) => a + v, 0) || 1;
  $("#timerBar").innerHTML = entries.map(([k, v], i) => `<div title="${k} ${(100 * v / tot).toFixed(1)}%" style="width:${100 * v / tot}%;background:${TIMER_COLORS[i % 7]}"></div>`).join("");
  $("#timerLegend").innerHTML = entries.map(([k, v], i) => `<span><i style="background:${TIMER_COLORS[i % 7]}"></i>${esc(k)} ${(100 * v / tot).toFixed(0)}%</span>`).join("");
}

/* ------------------------------------------------------------------ charts */
function setupCanvas(cv) {
  const r = cv.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
  const w = Math.max(10, Math.round(r.width * dpr)), h = Math.max(10, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, r.width, r.height];
}
function drawResiduals() {
  const cv = $("#resChart"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const R = S.res, n = S.resCount;
  const pad = { l: 40, r: 8, t: 8, b: 18 };
  let lo = Infinity, hi = -Infinity;
  for (const k of R.keys) for (const v of R.series[k]) if (v > 0) { const l = Math.log10(v); lo = Math.min(lo, l); hi = Math.max(hi, l); }
  if (!isFinite(lo)) { lo = -6; hi = 0; }
  lo = Math.floor(Math.max(lo, -16)); hi = Math.ceil(Math.min(Math.max(hi, lo + 1), 4));
  const X = (i) => pad.l + (W - pad.l - pad.r) * (n > 1 ? i / (n - 1) : 0);
  const Y = (l) => pad.t + (H - pad.t - pad.b) * (hi - l) / (hi - lo);
  ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = "#8796a8"; ctx.strokeStyle = "#1e2836"; ctx.lineWidth = 1;
  for (let e = lo; e <= hi; e++) { const y = Y(e); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(`1e${e}`, 4, y + 3); }
  ctx.strokeStyle = "#2b3a50"; ctx.setLineDash([3, 3]);
  for (const lab of R.labels) { if (lab.at > 1) { const x = X(lab.at - 1); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); ctx.fillText(lab.label, x + 3, H - 5); } }
  ctx.setLineDash([]);
  for (const k of R.keys) {
    ctx.strokeStyle = RES_COLORS[k] || "#a5b4fc"; ctx.lineWidth = 1.4; ctx.beginPath();
    let started = false;
    R.series[k].forEach((v, i) => { if (!(v > 0)) return; const x = X(i), y = Y(Math.log10(v)); if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); });
    ctx.stroke();
  }
  $("#resLegend").innerHTML = R.keys.map((k) => `<span><i style="background:${RES_COLORS[k] || "#a5b4fc"}"></i>${k}</span>`).join("");
}
function drawMonitor() {
  const cv = $("#monChart"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const d = S.mon.filter((v) => v !== null && v !== undefined && isFinite(v));
  if (d.length < 2) return;
  let lo = Math.min(...d), hi = Math.max(...d); if (hi === lo) { hi += 1e-12; lo -= 1e-12; }
  ctx.strokeStyle = "#3fb6ff"; ctx.lineWidth = 1.5; ctx.beginPath();
  d.forEach((v, i) => { const x = 4 + (W - 8) * i / (d.length - 1), y = 6 + (H - 12) * (hi - v) / (hi - lo); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = "#8796a8"; ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(fmt(hi, 4), 6, 12); ctx.fillText(fmt(lo, 4), 6, H - 4);
}

/* ------------------------------------------------------------------ viewport */
function parseGeometry(g) {
  if (!g || !g.nodes) return null;
  return { nodes: b64f32(g.nodes), ptr: b64i32(g.ptr), loops: b64i32(g.loops), bbox: g.bbox, cells: g.cells };
}
function viewRect(W, H, lx, ly) {
  const m = { l: 24, r: 96, t: 24, b: 24 };
  const aw = W - m.l - m.r, ah = H - m.t - m.b;
  const s = Math.min(aw / lx, ah / ly);
  const w = lx * s, h = ly * s;
  return { x: m.l + (aw - w) / 2, y: m.t + (ah - h) / 2, w, h, s };
}
const off = document.createElement("canvas");
function draw() {
  const cv = $("#view"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const f = S.frame;
  const cmap = lut($("#cmapSelect").value);
  if (f && f.mesh === "structured") drawStructured(ctx, W, H, f, cmap);
  else if (S.geometry) drawUnstructured(ctx, W, H, f, cmap);
  drawColorbar(f, cmap);
}
function drawStructured(ctx, W, H, f, cmap) {
  const { nx, ny, lx, ly } = f;
  const vr = viewRect(W, H, lx, ly); S.view = { ...vr, lx, ly, mesh: "structured" };
  off.width = nx; off.height = ny;
  const octx = off.getContext("2d"); const img = octx.createImageData(nx, ny);
  const lo = f.min, hi = f.max, inv = hi > lo ? 255 / (hi - lo) : 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = f.values[j * nx + i]; const p = 4 * ((ny - 1 - j) * nx + i);
    if (Number.isNaN(v)) { img.data[p] = 40; img.data[p + 1] = 46; img.data[p + 2] = 56; img.data[p + 3] = 255; continue; }
    const k = 3 * Math.max(0, Math.min(255, Math.round((v - lo) * inv)));
    img.data[p] = cmap[k]; img.data[p + 1] = cmap[k + 1]; img.data[p + 2] = cmap[k + 2]; img.data[p + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = $("#smooth").checked;
  ctx.drawImage(off, vr.x, vr.y, vr.w, vr.h);
  ctx.strokeStyle = "#3a4b62"; ctx.lineWidth = 1; ctx.strokeRect(vr.x, vr.y, vr.w, vr.h);
  if ($("#showStream").checked) drawStreamlines(ctx, f, vr);
  if ($("#showVectors").checked) drawVectors(ctx, f, vr);
}
function sampleVel(f, gx, gy) {
  const nx = f.vnx, ny = f.vny;
  const x = gx - 0.5, y = gy - 0.5;
  let i = Math.floor(x), j = Math.floor(y);
  if (i < -1 || j < -1 || i > nx - 1 || j > ny - 1) return null;
  i = Math.max(0, Math.min(nx - 2, i)); j = Math.max(0, Math.min(ny - 2, j));
  const tx = Math.max(0, Math.min(1, x - i)), ty = Math.max(0, Math.min(1, y - j));
  const id = (a, b) => b * nx + a;
  const u00 = f.u[id(i, j)], u10 = f.u[id(i + 1, j)], u01 = f.u[id(i, j + 1)], u11 = f.u[id(i + 1, j + 1)];
  const v00 = f.v[id(i, j)], v10 = f.v[id(i + 1, j)], v01 = f.v[id(i, j + 1)], v11 = f.v[id(i + 1, j + 1)];
  if ([u00, u10, u01, u11].some(Number.isNaN)) return null;
  const u = (u00 * (1 - tx) + u10 * tx) * (1 - ty) + (u01 * (1 - tx) + u11 * tx) * ty;
  const v = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
  return [u, v];
}
function drawStreamlines(ctx, f, vr) {
  const nx = f.vnx, ny = f.vny;
  const sx = vr.w / nx, sy = vr.h / ny; // px per velocity cell
  const occN = 2, occ = new Uint8Array(nx * occN * ny * occN);
  const ar = (f.lx / nx) / (f.ly / ny); // physical cell aspect
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 0.9;
  const seeds = []; const ns = 26;
  for (let a = 0; a < ns; a++) for (let b = 0; b < ns; b++) seeds.push([(a + 0.5) / ns * nx, (b + 0.5) / ns * ny]);
  for (const [sx0, sy0] of seeds) {
    const ox = Math.floor(sx0 * occN), oy = Math.floor(sy0 * occN);
    if (occ[oy * nx * occN + ox]) continue;
    for (const dir of [1, -1]) {
      let x = sx0, y = sy0; const pts = [[x, y]]; const mine = [];
      for (let k = 0; k < 400; k++) {
        const a = sampleVel(f, x, y); if (!a) break;
        let [u, v] = a; u /= ar; const m = Math.hypot(u, v); if (m < 1e-12) break;
        const h = 0.35 * dir;
        const xm = x + 0.5 * h * u / m, ym = y + 0.5 * h * v / m;
        const b = sampleVel(f, xm, ym); if (!b) break;
        let [u2, v2] = b; u2 /= ar; const m2 = Math.hypot(u2, v2); if (m2 < 1e-12) break;
        x += h * u2 / m2; y += h * v2 / m2;
        if (x < 0 || y < 0 || x > nx || y > ny) break;
        const cx = Math.floor(x * occN), cy = Math.floor(y * occN), ci = cy * nx * occN + cx;
        if (occ[ci] && !mine.includes(ci)) break;
        if (!mine.includes(ci)) { mine.push(ci); if (mine.length > 6) occ[mine[mine.length - 7]] = 1; }
        pts.push([x, y]);
      }
      mine.forEach((ci) => { occ[ci] = 1; });
      if (pts.length < 4) continue;
      ctx.beginPath();
      pts.forEach(([px, py], i) => { const X = vr.x + px * sx, Y = vr.y + vr.h - py * sy; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke();
    }
  }
}
function drawVectors(ctx, f, vr) {
  const nx = f.vnx, ny = f.vny; const step = Math.max(1, Math.round(Math.max(nx, ny) / 28));
  let umax = 1e-30; for (let k = 0; k < f.u.length; k++) { const m = Math.hypot(f.u[k], f.v[k]); if (m > umax) umax = m; }
  const L = step * Math.min(vr.w / nx, vr.h / ny) * 0.9;
  ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 1;
  for (let j = Math.floor(step / 2); j < ny; j += step) for (let i = Math.floor(step / 2); i < nx; i += step) {
    const u = f.u[j * nx + i], v = f.v[j * nx + i]; if (Number.isNaN(u)) continue;
    const m = Math.hypot(u, v) / umax; if (m < 1e-3) continue;
    const X = vr.x + (i + 0.5) * vr.w / nx, Y = vr.y + vr.h - (j + 0.5) * vr.h / ny;
    const dx = u / umax * L, dy = -v / umax * L; const ex = X + dx, ey = Y + dy;
    ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(ex, ey); ctx.stroke();
    const a = Math.atan2(dy, dx), hl = Math.max(2.5, 0.3 * Math.hypot(dx, dy));
    ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - hl * Math.cos(a - 0.45), ey - hl * Math.sin(a - 0.45)); ctx.lineTo(ex - hl * Math.cos(a + 0.45), ey - hl * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill();
  }
}
function drawUnstructured(ctx, W, H, f, cmap) {
  const g = S.geometry; const [x0, y0, x1, y1] = g.bbox;
  const vr = viewRect(W, H, x1 - x0, y1 - y0); S.view = { ...vr, mesh: "unstructured", x0, y0 };
  const key = `${W}x${H}`;
  if (!S.cellPaths || S.cellPaths.key !== key) {
    const paths = new Array(g.cells), cent = new Float32Array(2 * g.cells);
    for (let c = 0; c < g.cells; c++) {
      const p = new Path2D(); let cx = 0, cy = 0; const s = g.ptr[c], e = g.ptr[c + 1];
      for (let k = s; k < e; k++) {
        const n = g.loops[k]; const X = vr.x + (g.nodes[2 * n] - x0) * vr.s, Y = vr.y + vr.h - (g.nodes[2 * n + 1] - y0) * vr.s;
        k === s ? p.moveTo(X, Y) : p.lineTo(X, Y); cx += g.nodes[2 * n]; cy += g.nodes[2 * n + 1];
      }
      p.closePath(); paths[c] = p; cent[2 * c] = cx / (e - s); cent[2 * c + 1] = cy / (e - s);
    }
    S.cellPaths = { key, paths }; S.cellCentres = cent;
  }
  const edges = $("#showEdges").checked;
  const lo = f ? f.min : 0, hi = f ? f.max : 1, inv = hi > lo ? 255 / (hi - lo) : 0;
  ctx.lineWidth = 0.5;
  for (let c = 0; c < g.cells; c++) {
    if (f) { const k = 3 * Math.max(0, Math.min(255, Math.round((f.values[c] - lo) * inv))); ctx.fillStyle = `rgb(${cmap[k]},${cmap[k + 1]},${cmap[k + 2]})`; }
    else ctx.fillStyle = "#1a2433";
    ctx.fill(S.cellPaths.paths[c]);
    if (edges || !f) { ctx.strokeStyle = f ? "rgba(0,0,0,0.35)" : "#3b4d66"; ctx.stroke(S.cellPaths.paths[c]); }
  }
  if (!f) $("#empty").classList.add("hidden");
}
function drawColorbar(f, cmap) {
  const cv = $("#colorbar"); const [ctx, W, H] = setupCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  if (!f) return;
  const bw = 14, x = 4;
  for (let y = 0; y < H - 20; y++) { const k = 3 * Math.round(255 * (1 - y / (H - 21))); ctx.fillStyle = `rgb(${cmap[k]},${cmap[k + 1]},${cmap[k + 2]})`; ctx.fillRect(x, 10 + y, bw, 1); }
  ctx.strokeStyle = "#3a4b62"; ctx.strokeRect(x, 10, bw, H - 20);
  ctx.fillStyle = "#c9d4e0"; ctx.font = "10.5px ui-monospace, monospace";
  for (let i = 0; i <= 4; i++) { const v = f.max - (f.max - f.min) * i / 4; ctx.fillText(fmt(v, 3), x + bw + 4, 14 + (H - 24) * i / 4); }
}
function initHover() {
  const vp = $("#viewport"), hv = $("#hover");
  vp.addEventListener("mousemove", (e) => {
    const f = S.frame, v = S.view; if (!f || !v) { hv.style.display = "none"; return; }
    const r = vp.getBoundingClientRect(); const px = e.clientX - r.left, py = e.clientY - r.top;
    if (px < v.x || py < v.y || px > v.x + v.w || py > v.y + v.h) { hv.style.display = "none"; return; }
    let txt;
    if (v.mesh === "structured") {
      const fx = (px - v.x) / v.w, fy = 1 - (py - v.y) / v.h;
      const i = Math.min(f.nx - 1, Math.floor(fx * f.nx)), j = Math.min(f.ny - 1, Math.floor(fy * f.ny));
      txt = `x=${fmt(fx * f.lx, 4)} y=${fmt(fy * f.ly, 4)}  ${f.field}=${fmt(f.values[j * f.nx + i], 5)}`;
    } else if (S.cellCentres) {
      const X = v.x0 + (px - v.x) / v.s, Y = v.y0 + (v.y + v.h - py) / v.s; const c = S.cellCentres; let best = 0, bd = Infinity;
      for (let k = 0; k < c.length / 2; k++) { const d = (c[2 * k] - X) ** 2 + (c[2 * k + 1] - Y) ** 2; if (d < bd) { bd = d; best = k; } }
      txt = `x=${fmt(X, 4)} y=${fmt(Y, 4)}  cell ${best}  ${f.field}=${fmt(f.values[best], 5)}`;
    }
    hv.textContent = txt; hv.style.display = "block";
  });
  vp.addEventListener("mouseleave", () => { hv.style.display = "none"; });
}

/* ------------------------------------------------------------------ boot */
async function restoreState() {
  const st = await api("/api/state");
  for (const ev of st.log) logLine(ev);
  for (const ev of st.residuals) onResiduals(ev);
  if (st.job) {
    onStatus({ status: st.job.status, result: st.job.result });
    if (st.job.plan) onPlan({ plan: st.job.plan });
  }
}
function init() {
  initTabs(); initMeshUpload(); initHover();
  $("#runBtn").addEventListener("click", run);
  $("#stopBtn").addEventListener("click", () => post("/api/stop"));
  $("#mode").addEventListener("change", () => setSource(S.source));
  $("#fieldSelect").addEventListener("change", (e) => post("/api/view", { field: e.target.value }));
  ["#cmapSelect", "#showStream", "#showVectors", "#showEdges", "#smooth"].forEach((s) => $(s).addEventListener("change", draw));
  $("#benchBtn").addEventListener("click", async () => {
    $("#benchBtn").disabled = true; $("#benchBox").textContent = "Benchmarking every backend…";
    try { renderBench(await post("/api/benchmark")); } catch (e) { $("#benchBox").textContent = e.message; }
    $("#benchBtn").disabled = false;
  });
  $("#estimateBtn").addEventListener("click", async () => {
    try {
      const spec = buildSpec();
      const r = await post("/api/estimate", spec.kind === "structured" ? spec : { kind: "unstructured", mesh_id: spec.mesh_id, energy: spec.case.energy });
      $("#estimateOut").innerHTML = Object.entries(r).filter(([k]) => k !== "cells").map(([k, v]) => `<b>${k}</b> (${v.device}): ${fmtBytes(v.required)} of ${fmtBytes(v.available)} ${v.fits ? '<span style="color:var(--ok)">fits</span>' : '<span style="color:var(--err)">does not fit</span>'}`).join("<br>") + `<br>${(r.cells || 0).toLocaleString()} cells`;
    } catch (e) { $("#estimateOut").textContent = e.message; }
  });
  $$("[data-export]").forEach((b) => b.addEventListener("click", async () => {
    const r = await fetch(`/api/export/${b.dataset.export}`);
    if (!r.ok) { const j = await r.json().catch(() => ({})); logLine({ level: "warn", msg: `export: ${j.error || r.statusText}` }); return; }
    const blob = await r.blob(); const a = document.createElement("a");
    const cd = r.headers.get("Content-Disposition") || ""; const m = cd.match(/filename="([^"]+)"/);
    a.href = URL.createObjectURL(blob); a.download = m ? m[1] : `flowtrace.${b.dataset.export}`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }));
  window.addEventListener("resize", () => { S.cellPaths = null; draw(); drawResiduals(); drawMonitor(); });
  setSource("structured");
  Promise.all([loadSystem(), loadPresets()]).then(restoreState).catch((e) => logLine({ level: "error", msg: e.message }));
  connectEvents();
  draw(); drawResiduals();
}
document.addEventListener("DOMContentLoaded", init);
