"""Draw pedigrees and animate how search candidates evolve.

- `layout(ped)`: layered positions (generations top to bottom; order within a
  row refined by parents' and children's positions to reduce crossings).
- `draw_pedigree(ax, ped)`: matplotlib drawing. Observed people are coloured
  circles (same colour for the same person everywhere), latent people grey
  squares; a red ring marks an inbred person.
- `write_gif(...)`: frames of [references | best | per reference: closest, latest removed
  closest | latest fit]; a removal is outlined in red on its frame.
- `write_html(...)`: one self-contained web page (no server) with a method
  selector, play/pause and a generation slider; explanations come from
  webviz/help.js (shared with the explorer) and are inlined.

Snapshots come from runs with `RunConfig(snapshot=True)` (RunResult.snapshots).
"""

from __future__ import annotations

import io as _io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .genetics.kinship import inbreeding
from .genetics.pedigree import Pedigree

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
           "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#2ca02c"]


def colors_for(observed_ids: Sequence[str]) -> dict[str, str]:
    return {pid: PALETTE[i % len(PALETTE)] for i, pid in enumerate(observed_ids)}


# ---- layout -----------------------------------------------------------------

def layout(ped: Pedigree, observed_order: Sequence[str] | None = None, sweeps: int = 4) -> dict:
    """{"nodes": [{id, x, y, observed, inbred}], "edges": [[parent, child]]}; y = generation."""
    try:
        order = ped.topological_order()
    except Exception:
        return {"nodes": [], "edges": [], "error": "invalid (cycle)"}
    rank = {pid: i for i, pid in enumerate(observed_order or [])}
    children = ped.children_map()
    depth: dict[str, int] = {}
    for pid in order:
        ps = ped[pid].parents
        depth[pid] = 1 + max(depth[p] for p in ps) if ps else 0
    for _ in range(len(order)):          # founders sit just above their children
        changed = False
        for pid in reversed(order):
            kids = children[pid]
            if kids:
                want = min(depth[k] for k in kids) - 1
                if want > depth[pid]:
                    depth[pid] = want
                    changed = True
        if not changed:
            break
    levels: dict[int, list[str]] = {}
    for pid in sorted(order, key=lambda p: (not ped[p].observed, rank.get(p, len(rank)), p)):
        levels.setdefault(depth[pid], []).append(pid)

    def assign():
        return {pid: i - (len(row) - 1) / 2 for row in levels.values() for i, pid in enumerate(row)}

    x = assign()
    for s in range(sweeps):
        down = s % 2 == 0
        for d in sorted(levels, reverse=not down):
            def key(pid):
                nb = ped[pid].parents if down else children[pid]
                return float(np.mean([x[n] for n in nb])) if nb else x[pid]
            levels[d].sort(key=key)
            x = assign()
    f = inbreeding(ped)
    nodes = [{"id": pid, "x": float(x[pid]), "y": int(depth[pid]),
              "observed": bool(ped[pid].observed), "inbred": bool(f[pid] > 1e-12)}
             for pid in order]
    edges = [[p, pid] for pid in order for p in ped[pid].parents]
    return {"nodes": nodes, "edges": edges}


# ---- matplotlib ---------------------------------------------------------------

def draw_pedigree(ax, ped: Pedigree | None, colors: dict[str, str], title: str = "",
                  observed_order: Sequence[str] | None = None) -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    if ped is None:
        ax.text(0.5, 0.5, "none yet", ha="center", va="center", color="#888", transform=ax.transAxes)
        return
    lay = layout(ped, observed_order)
    if lay.get("error"):
        ax.text(0.5, 0.5, lay["error"], ha="center", va="center", color="#c00", transform=ax.transAxes)
        return
    pos = {n["id"]: (n["x"], -n["y"]) for n in lay["nodes"]}
    for p, c in lay["edges"]:
        ax.annotate("", xy=pos[c], xytext=pos[p],
                    arrowprops=dict(arrowstyle="-|>", color="#999", lw=1, shrinkA=13, shrinkB=13))
    for n in lay["nodes"]:
        x, y = pos[n["id"]]
        ring = "#d62728" if n["inbred"] else "#555"
        if n["observed"]:
            ax.scatter([x], [y], s=650, c=colors.get(n["id"], "#4e79a7"), edgecolors=ring,
                       linewidths=2.5 if n["inbred"] else 1, zorder=3)
            ax.text(x, y, n["id"], ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", zorder=4)
        else:
            ax.scatter([x], [y], s=420, marker="s", c="#dddddd", edgecolors=ring,
                       linewidths=2.5 if n["inbred"] else 1, zorder=3)
            ax.text(x, y, n["id"], ha="center", va="center", fontsize=7, color="#333", zorder=4)
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2   # same scale in every panel
    w, h = max(max(xs) - min(xs) + 1.6, 5.0), max(max(ys) - min(ys) + 1.4, 3.6)
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)


def _subsample(snapshots: list[dict], max_frames: int) -> list[dict]:
    """Evenly spaced frames, always including generations with a removal event."""
    if len(snapshots) <= max_frames:
        return snapshots
    idx = set(np.linspace(0, len(snapshots) - 1, max_frames).round().astype(int).tolist())
    events = [i for i, s in enumerate(snapshots) if s.get("events")]
    if len(events) > max_frames:
        events = [events[j] for j in np.linspace(0, len(events) - 1, max_frames).round().astype(int)]
    return [snapshots[i] for i in sorted(idx | set(events))]


def _fmt_obj(names: Sequence[str], values: Sequence[float]) -> str:
    return ", ".join(f"{n}={v:.3g}" for n, v in zip(names, values))


def _meta(item: dict, names: Sequence[str], sep: str = " · ") -> str:
    """Objectives, fit status and distances to each reference of a recorded candidate."""
    parts = []
    if item["worst_error"] >= 1e5:
        parts.append("invalid (penalty on every objective)")
    else:
        parts.append(_fmt_obj(names, item["objectives"]))
        parts.append("fits" if item["fits"] else f"worst pair error {item['worst_error']:.3g}")
    for label, d in item["distances"].items():
        parts.append(f"to {label}: {d['structure']} edits, IBD diff {d['relationship']:.3f}")
    return sep.join(parts)


def snapshot_panels(snap: dict, objective_names: Sequence[str],
                    previous_generation: int = -1) -> list[dict]:
    """[{title, meta, pedigree, alert}] for one snapshot: best; per reference the
    closest candidates and the latest removed closest candidate; latest fit.
    `alert` marks a removal newer than `previous_generation` (the previous frame)."""
    panels = []
    for name, items in snap["records"].items():
        for j, item in enumerate(items):
            if name == "best":
                title = "Best in population"
            else:
                title = f"Closest to {name.split(':', 1)[1]}" + (f" #{j + 1}" if len(items) > 1 else "")
            panels.append({"title": title, "meta": _meta(item, objective_names),
                           "pedigree": item["pedigree"], "alert": False})
        if name.startswith("closest:"):
            label = name.split(":", 1)[1]
            ev = snap.get("last_removed", {}).get(label)
            if ev is None:
                panels.append({"title": f"Removed closest to {label}", "meta": "none removed so far",
                               "pedigree": None, "alert": False})
            else:
                alert = ev["generation"] > previous_generation
                lost = ev["lost"]
                meta = (f"removed in generation {ev['generation']}"
                        f"{' (NEW)' if alert else ''}: was {ev['distance_before']} edits away"
                        f" · {_meta(lost, objective_names)}"
                        f" · replaced as closest by a candidate {ev['distance_after']} edits away"
                        f" ({_fmt_obj(objective_names, ev['replaced_by']['objectives'])})")
                panels.append({"title": f"Removed closest to {label}", "meta": meta,
                               "pedigree": lost["pedigree"], "alert": alert})
    fit_meta = (f"{snap['n_fits']} fits so far · latest found at evaluation {snap['latest_fit_at']:,}"
                if snap["latest_fit"] is not None else "0 fits so far")
    panels.append({"title": "Latest fit", "meta": fit_meta, "pedigree": snap["latest_fit"],
                   "alert": False})
    return panels


def _wrap(text: str, width: int = 46) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def write_gif(path: str | Path, snapshots: list[dict], observed_ids: Sequence[str],
              references: dict[str, Pedigree] | None = None, title: str = "",
              objective_names: Sequence[str] = (), max_frames: int = 80,
              frame_ms: int = 250, last_ms: int = 2500) -> Path:
    """Animated GIF: [references | best | closest to each reference | latest fit]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    colors = colors_for(observed_ids)
    references = references or {}
    frames = []
    prev_gen = -1
    for s in _subsample(snapshots, max_frames):
        panels = ([{"title": f"Reference: {label}", "meta": "", "pedigree": ped, "alert": False}
                   for label, ped in references.items()]
                  + snapshot_panels(s, objective_names, prev_gen))
        prev_gen = s["generation"]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.8), dpi=80,
                                 squeeze=False)
        for ax, panel in zip(axes[0], panels):
            draw_pedigree(ax, panel["pedigree"], colors, "", observed_ids)
            ax.set_title(f"{panel['title']}\n{_wrap(panel['meta'])}" if panel["meta"]
                         else panel["title"], fontsize=8,
                         color="#d62728" if panel["alert"] else "black")
            if panel["alert"]:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                       edgecolor="#d62728", linewidth=4, clip_on=False))
        fig.suptitle(f"{title}   generation {s['generation']}, {s['evals']:,} evaluations",
                     fontsize=11)
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [frame_ms] * (len(frames) - 1) + [last_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return path


# ---- web page -----------------------------------------------------------------

def frames_json(snapshots: list[dict], observed_ids: Sequence[str],
                objective_names: Sequence[str] = (), max_frames: int = 150) -> list[dict]:
    out = []
    prev_gen = -1
    for s in _subsample(snapshots, max_frames):
        out.append({"generation": s["generation"], "evals": s["evals"],
                    "population": s["population"],
                    "panels": [{"title": p["title"], "meta": p["meta"], "alert": p["alert"],
                                "layout": layout(p["pedigree"], observed_ids)
                                if p["pedigree"] is not None else None}
                               for p in snapshot_panels(s, objective_names, prev_gen)]})
        prev_gen = s["generation"]
    return out


def write_html(path: str | Path, methods: list[dict], observed_ids: Sequence[str],
               references: dict[str, Pedigree] | None = None,
               title: str = "Pedigree search") -> Path:
    """Self-contained viewer. `methods`: [{"label", "objective_names", "snapshots",
    "summary" (optional dict), "kind" (optional: ibd, king or kinship)}];
    `references` are drawn as fixed panels."""
    data = {
        "title": title,
        "colors": colors_for(observed_ids),
        "references": [{"label": label, "layout": layout(ped, observed_ids)}
                       for label, ped in (references or {}).items()],
        "methods": [{"label": m["label"], "summary": m.get("summary", {}),
                     "objectives": list(m.get("objective_names", ())), "kind": m.get("kind"),
                     "frames": frames_json(m["snapshots"], observed_ids,
                                           m.get("objective_names", ()))} for m in methods],
    }
    help_js = (Path(__file__).parent / "webviz" / "help.js").read_text()
    html = (_HTML.replace("__TITLE__", _escape(title))
            .replace("__HELP__", help_js.replace("</", "<\\/"))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#9a9aa0; --card:#f6f6f7; --border:#dcdce0;
        --latent:#e2e2e6; --accent:#2563eb; --bad:#d62728; --ref:#eef3fd; }
@media (prefers-color-scheme: dark) { :root { --bg:#151518; --fg:#ececf0; --muted:#a0a0a8; --line:#77777f;
        --card:#1f1f24; --border:#34343c; --latent:#3a3a42; --accent:#6ea0ff; --bad:#ff6b6b; --ref:#1c2433; } }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg);
       font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }
h1 { font-size:18px; margin:0 0 4px; }
.sub { color:var(--muted); margin-bottom:12px; max-width:900px; }
.controls { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; margin-bottom:12px; }
select, button { font:inherit; padding:4px 10px; border:1px solid var(--border); border-radius:6px;
                 background:var(--card); color:var(--fg); cursor:pointer; }
input[type=range] { flex:1 1 240px; min-width:160px; accent-color:var(--accent); }
.status { color:var(--muted); font-variant-numeric:tabular-nums; }
.panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px; }
.panel { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:10px; }
.panel.ref { background:var(--ref); }
.panel.alert { border:3px solid var(--bad); }
.panel.alert h2, .panel.alert .meta { color:var(--bad); }
.panel h2 { font-size:13px; margin:0 0 2px; }
.panel .meta { font-size:12px; color:var(--muted); min-height:48px; }
svg { width:100%; height:280px; display:block; }
.summary { margin-top:10px; font-size:12px; color:var(--muted); }
.panel .note { min-height:30px; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }
</style></head><body>
<header><h1 id="title"></h1>
  <label title="show short explanations of every part of the page"><input type="checkbox" id="explain"> Explain</label></header>
<div class="note">Each frame is one generation of one run, recorded from a run database. Choose the run under Method,
then play or drag the slider. The text under each panel gives that candidate's objectives, whether it fits, and its
edits and IBD diff to each reference.</div>
<div id="guide"></div>
<div class="controls">
  <label>Method <select id="method"></select></label>
  <button id="play">Play</button>
  <label>Speed <select id="speed"><option value="400">slow</option><option value="180" selected>normal</option>
    <option value="60">fast</option></select></label>
  <input type="range" id="slider" min="0" value="0">
  <span class="status" id="status"></span>
</div>
<div class="note">Evaluations: pedigrees scored so far. Population: slots in the population. Frames may skip
generations in long runs (at most 150 frames per run).</div>
<div class="panels" id="panels"></div>
<div class="summary" id="summary"></div>
<div class="note">Run summary: totals for the selected run at its end, e.g. evaluations and fits found (n_fits: distinct
fits in the fit store), plus recall and when the truth was first found when a reference is known.</div>
<script>
__HELP__
</script>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function draw(svg, lay, note) {
  svg.innerHTML = "";
  if (!lay || !lay.nodes || !lay.nodes.length) {
    const t = el("text", {x:"50%", y:"50%", "text-anchor":"middle", fill:"var(--muted)"}, svg);
    t.textContent = (lay && lay.error) || note || "none yet"; return; }
  const xs = lay.nodes.map(n => n.x), ys = lay.nodes.map(n => n.y);
  // Same scale in every panel: pad small pedigrees to a minimum extent.
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const w = Math.max(Math.max(...xs) - Math.min(...xs) + 1.6, 5), h = Math.max(Math.max(...ys) - Math.min(...ys) + 1.4, 3.6);
  svg.setAttribute("viewBox", `${(cx - w / 2) * 100} ${(cy - h / 2) * 100} ${w * 100} ${h * 100}`);
  const defs = el("defs", {}, svg);
  const m = el("marker", {id:"arr" + svg.id, viewBox:"0 0 10 10", refX:"9", refY:"5", markerWidth:"7",
    markerHeight:"7", orient:"auto-start-reverse"}, defs);
  el("path", {d:"M0 0L10 5L0 10z", fill:"var(--line)"}, m);
  const pos = {}; lay.nodes.forEach(n => pos[n.id] = [n.x * 100, n.y * 100]);
  lay.edges.forEach(([p, c]) => { const [ax, ay] = pos[p], [bx, by] = pos[c];
    const dx = bx - ax, dy = by - ay, d = Math.hypot(dx, dy) || 1, r = 26;
    el("line", {x1: ax + dx / d * r, y1: ay + dy / d * r, x2: bx - dx / d * r, y2: by - dy / d * r,
      stroke:"var(--line)", "stroke-width":"2", "marker-end":`url(#arr${svg.id})`}, svg); });
  lay.nodes.forEach(n => { const [x, y] = pos[n.id];
    const ring = n.inbred ? "var(--bad)" : "rgba(0,0,0,.35)", sw = n.inbred ? 5 : 1.5;
    if (n.observed) el("circle", {cx:x, cy:y, r:24, fill: DATA.colors[n.id] || "var(--accent)", stroke:ring, "stroke-width":sw}, svg);
    else el("rect", {x:x - 19, y:y - 19, width:38, height:38, rx:4, fill:"var(--latent)", stroke:ring, "stroke-width":sw}, svg);
    const t = el("text", {x, y: y + 5, "text-anchor":"middle", "font-size": n.observed ? 15 : 13,
      "font-weight": n.observed ? 700 : 400, fill: n.observed ? "#fff" : "var(--fg)"}, svg);
    t.textContent = n.id; }); }
function helpKey(title) {
  return title.startsWith("Reference") ? "references" : title.startsWith("Best") ? "best"
    : title.startsWith("Closest") ? "closest" : title.startsWith("Removed") ? "removed"
    : title.startsWith("Latest fit") ? "latest_fit" : null;
}
function panel(title, ref) {
  const d = document.createElement("div"); d.className = "panel" + (ref ? " ref" : "");
  const h = document.createElement("h2"); h.textContent = title; d.appendChild(h);
  const n = document.createElement("div"); n.className = "note"; n.textContent = HELP.sections[helpKey(title)] || ""; d.appendChild(n);
  const m = document.createElement("div"); m.className = "meta"; d.appendChild(m);
  const s = el("svg", {id: "s" + Math.random().toString(36).slice(2)}); d.appendChild(s);
  $("panels").appendChild(d); return {d, h, m, s};
}
let mi = 0, fi = 0, timer = null, slots = [];
function build() {
  $("panels").innerHTML = "";
  DATA.references.forEach(r => { const p = panel("Reference: " + r.label, true);
    p.m.textContent = "fixed; never shown to the search"; draw(p.s, r.layout); });
  slots = DATA.methods[mi].frames[0].panels.map(p => panel(p.title, false));
}
function render() {
  const M = DATA.methods[mi], F = M.frames[fi];
  $("slider").max = M.frames.length - 1; $("slider").value = fi;
  $("status").textContent = `generation ${F.generation} · ${F.evals.toLocaleString()} evaluations · population ${F.population} · frame ${fi + 1}/${M.frames.length}`;
  F.panels.forEach((p, i) => { const s = slots[i]; if (!s) return;
    s.d.className = "panel" + (p.alert ? " alert" : "");
    s.h.textContent = p.title; s.m.textContent = p.meta; draw(s.s, p.layout, "none"); });
  const s = M.summary || {}; $("summary").textContent = Object.keys(s).length ?
    "Run summary: " + Object.entries(s).map(([k, v]) => `${k}: ${v}`).join(" · ") : "";
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "Play"; }
function play() { if (timer) return stop(); $("play").textContent = "Pause";
  timer = setInterval(() => { const n = DATA.methods[mi].frames.length;
    if (fi >= n - 1) return stop(); fi++; render(); }, +$("speed").value); }
$("title").textContent = DATA.title;
const GUIDE = HELP.guide([], null, null, ["Badges"]);
$("guide").replaceWith(GUIDE);
const showObjectives = () => GUIDE.setObjectives(DATA.methods[mi].objectives, DATA.methods[mi].kind);
HELP.explainSwitch($("explain"));
showObjectives();
DATA.methods.forEach((m, i) => { const o = document.createElement("option"); o.value = i; o.textContent = m.label; $("method").appendChild(o); });
$("method").onchange = e => { stop(); mi = +e.target.value; fi = 0; build(); render(); showObjectives(); };
$("slider").oninput = e => { stop(); fi = +e.target.value; render(); };
$("play").onclick = () => { if (fi >= DATA.methods[mi].frames.length - 1) fi = 0; play(); };
$("speed").onchange = () => { if (timer) { stop(); play(); } };
build(); render();
</script></body></html>
"""
