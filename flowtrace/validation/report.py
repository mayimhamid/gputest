"""Benchmark report: results.json -> report.html (self-contained), report.pdf, validation_table.md.

The HTML report embeds every plot as inline SVG, so it opens offline and can
be attached to a validation dossier as a single file. Plots need matplotlib
(``pip install flowtrace[report]``); without it the report keeps the tables
and lists the plot data instead.
"""

from __future__ import annotations

import html
import io
import os
from typing import List

_CSS = """
:root{--bg:#fff;--fg:#1d2433;--muted:#5b6475;--line:#d9dee7;--ok:#13795b;--bad:#b42318;--card:#f6f8fb;--accent:#2457c5}
@media (prefers-color-scheme: dark){:root{--bg:#11151c;--fg:#e6e9ef;--muted:#9aa4b5;--line:#2a3140;--ok:#3ccf91;--bad:#ff7a6b;--card:#171c25;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{font-weight:600;color:var(--muted)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}.plots svg{width:100%;height:auto;background:#fff;border-radius:6px}
code{font-family:ui-monospace,Consolas,monospace;font-size:13px}.summary{display:flex;gap:24px;flex-wrap:wrap}
.summary div{min-width:120px}.big{font-size:26px;font-weight:700}
"""


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if f == 0:
        return "0"
    a = abs(f)
    if a >= 1e4 or a < 1e-3:
        return f"{f:.3e}"
    return f"{f:.4g}"


def _plot_svg(p: dict) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=100)
    for s in p.get("series", []):
        x, y = s.get("x") or [], s.get("y") or []
        if not x:
            continue
        style = s.get("style", "line")
        if style == "ref":
            ax.plot(x, y, "o", mfc="none", color="#c0392b", ms=5, label=s.get("label"))
        elif style == "marker":
            ax.plot(x, y, "s-", ms=4, label=s.get("label"))
        else:
            ax.plot(x, y, "-", lw=1.6, label=s.get("label"))
    if p.get("logx"):
        ax.set_xscale("log")
    if p.get("logy"):
        ax.set_yscale("log")
    ax.set_title(p.get("title", ""), fontsize=10)
    ax.set_xlabel(p.get("xlabel", ""))
    ax.set_ylabel(p.get("ylabel", ""))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg"):]


def _metric_rows(r: dict) -> str:
    rows = []
    for name, m in (r.get("metrics") or {}).items():
        ok = m.get("passed")
        cls = "" if ok is None else ("ok" if ok else "bad")
        verdict = "" if ok is None else ("pass" if ok else "FAIL")
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{_fmt(m.get('value'))} {html.escape(m.get('unit') or '')}</td>"
            f"<td>{_fmt(m.get('reference'))}</td><td>{html.escape(m.get('tolerance') or '')}</td>"
            f"<td class='{cls}'>{verdict}</td><td class='muted'>{html.escape(m.get('source') or '')}</td></tr>")
    return ("<table><tr><th>Quantity</th><th>FlowTrace</th><th>Reference</th><th>Acceptance</th><th></th>"
            "<th>Source</th></tr>" + "".join(rows) + "</table>")


def render_html(data: dict) -> str:
    m = data.get("machine", {})
    res = data.get("results", [])
    n_ok = sum(bool(r.get("passed")) for r in res)
    parts: List[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>FlowTrace Validation Report</title><style>", _CSS, "</style></head><body><main>",
        "<h1>FlowTrace validation report</h1>",
        f"<p class='muted'>Generated {html.escape(str(m.get('date', '')))} by "
        f"<code>flowtrace benchmark --tier {html.escape(str(data.get('tier')))}</code>. Every number below was "
        "computed by the benchmark scripts in this run; nothing is copied from elsewhere.</p>",
        "<div class='card summary'>",
        f"<div><div class='big'>{n_ok}/{len(res)}</div><div class='muted'>benchmarks passed</div></div>",
        f"<div><div class='big'>{html.escape(str(m.get('flowtrace', '?')))}</div><div class='muted'>FlowTrace</div></div>",
        "</div>",
        "<div class='card'><b>Machine</b><br>",
        f"{html.escape(str(m.get('cpu')))} ({html.escape(str(m.get('cores')))}), {m.get('ram_gb')} GB RAM<br>",
        f"GPU: {html.escape(str(m.get('gpu') or 'none used'))} &middot; backends: {html.escape(', '.join(m.get('backends', [])))}<br>",
        f"{html.escape(str(m.get('platform')))}, Python {html.escape(str(m.get('python')))}</div>",
        "<h2>Summary</h2><table><tr><th>Benchmark</th><th>Category</th><th>Result</th><th>Time</th></tr>",
    ]
    for r in res:
        cls = "ok" if r.get("passed") else "bad"
        parts.append(f"<tr><td><a href='#{html.escape(r['id'])}'>{html.escape(r.get('title') or r['id'])}</a></td>"
                     f"<td>{html.escape(r.get('category', ''))}</td><td class='{cls}'>"
                     f"{'pass' if r.get('passed') else 'FAIL'}</td><td>{_fmt(r.get('seconds'))} s</td></tr>")
    parts.append("</table>")
    for r in res:
        parts.append(f"<h2 id='{html.escape(r['id'])}'>{html.escape(r.get('title') or r['id'])}</h2>")
        parts.append(f"<p class='muted'><code>{html.escape(r['id'])}</code> &middot; {html.escape(r.get('category', ''))}"
                     f" &middot; {_fmt(r.get('seconds'))} s</p>")
        if r.get("error"):
            parts.append(f"<p class='bad'>Error: {html.escape(r['error'])}</p>")
        parts.append(_metric_rows(r))
        svgs = [s for s in (_plot_svg(p) for p in r.get("plots", [])) if s]
        if svgs:
            parts.append("<div class='plots'>" + "".join(svgs) + "</div>")
    parts.append("<h2>Reproduce</h2><pre><code>pip install -e .[report]\n"
                 f"python -m flowtrace benchmark --tier {html.escape(str(data.get('tier')))} --out benchmarks/results"
                 "</code></pre></main></body></html>")
    return "".join(parts)


def render_markdown(data: dict) -> str:
    """Compact table of every metric with a reference (used by README/docs)."""
    lines = ["| Benchmark | Quantity | FlowTrace | Reference | Acceptance | Result |",
             "|---|---|---|---|---|---|"]
    for r in data.get("results", []):
        for name, m in (r.get("metrics") or {}).items():
            if m.get("passed") is None:
                continue
            ref = _fmt(m.get("reference"))
            lines.append(f"| `{r['id']}` | {name} | {_fmt(m.get('value'))} {m.get('unit') or ''} | {ref} | "
                         f"{m.get('tolerance') or ''} | {'pass' if m.get('passed') else '**FAIL**'} |")
    mach = data.get("machine", {})
    lines.append("")
    lines.append(f"_Measured {mach.get('date', '')} on {mach.get('cpu', '?')}"
                 f"{', GPU ' + mach['gpu'] if mach.get('gpu') else ''}; tier `{data.get('tier')}`._")
    return "\n".join(lines) + "\n"


def write_pdf(data: dict, path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return False
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        m = data.get("machine", {})
        txt = [f"FlowTrace validation report  ({data.get('tier')} tier)", "",
               f"{m.get('date')}  FlowTrace {m.get('flowtrace')}", f"{m.get('cpu')} ({m.get('cores')})",
               f"GPU: {m.get('gpu') or 'none used'}", ""]
        for r in data.get("results", []):
            txt.append(f"[{'PASS' if r.get('passed') else 'FAIL'}] {r.get('title') or r['id']}")
            for name, mm in (r.get("metrics") or {}).items():
                ref = "" if mm.get("reference") is None else f"  (ref {_fmt(mm['reference'])})"
                txt.append(f"      {name}: {_fmt(mm.get('value'))} {mm.get('unit') or ''}{ref}")
        fig.text(0.05, 0.97, "\n".join(txt[:110]), va="top", family="monospace", fontsize=6.5)
        pdf.savefig(fig)
        plt.close(fig)
        for r in data.get("results", []):
            for p in r.get("plots", []):
                fig, ax = plt.subplots(figsize=(8.27, 5.5))
                for s in p.get("series", []):
                    if not s.get("x"):
                        continue
                    fmt = "o" if s.get("style") == "ref" else ("s-" if s.get("style") == "marker" else "-")
                    ax.plot(s["x"], s["y"], fmt, mfc="none" if s.get("style") == "ref" else None, label=s.get("label"))
                if p.get("logx"):
                    ax.set_xscale("log")
                if p.get("logy"):
                    ax.set_yscale("log")
                ax.set_title(f"{r['id']}: {p.get('title', '')}")
                ax.set_xlabel(p.get("xlabel", ""))
                ax.set_ylabel(p.get("ylabel", ""))
                ax.grid(alpha=0.3)
                ax.legend()
                pdf.savefig(fig)
                plt.close(fig)
    return True


def write_report(data: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    p = os.path.join(out_dir, "report.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(render_html(data))
    files["html"] = p
    p = os.path.join(out_dir, "validation_table.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(data))
    files["md"] = p
    p = os.path.join(out_dir, "report.pdf")
    if write_pdf(data, p):
        files["pdf"] = p
    return files
