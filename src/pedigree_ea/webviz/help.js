// Explanations shared by the explorer (index.html, served as /help.js) and the
// self-contained pages written by viz.write_html (inlined there). Elements with
// class "note" and the guide are shown only while the "Explain" switch is on.

const HELP = {
  legend: [
    ["obs", "Circle", "An observed person (a sample in the data), labelled with its ID; same colour everywhere."],
    ["lat", "Grey square", "A latent person: unobserved, added by the search to connect observed people."],
    ["", "Arrow", "Points from parent to child; rows are generations."],
    ["inb", "Red ring", "That person is inbred (their parents are related)."],
    ["", "Badges", "fits: every pair's predicted relatedness is within tolerance of the data, i.e. a solution. " +
      "invalid: not a possible pedigree (e.g. someone is their own ancestor). inbred: someone in it is inbred."],
    ["", "Edits", "Parent links to add or remove to turn a candidate into a reference, matching latent people as well as possible."],
    ["", "IBD diff", "Largest difference in expected IBD0/1/2 between a candidate and a reference, over observed pairs."],
    ["", "Fit store", "Every fit found is kept permanently, separately from the population, so fits are never lost " +
      "even when the population drops them."]],
  sections: {
    references: "Known pedigrees (e.g. the true one) for comparison. The search never sees them; only the recorder does.",
    best: "The population member with the lowest first objective; ties broken by the next objective, and so on.",
    closest: "Population members needing the fewest edits to become the reference (ties: smallest IBD difference).",
    removed: "The last time the closest member to a reference left the population, and what replaced it. " +
      "Red frame: it happened in this generation.",
    latest_fit: "The most recently found fit, from the fit store. It may no longer be in the population.",
    pareto: "Members no other member beats on every objective (plain Pareto ranking; fit status not used).",
    population: "Every distinct pedigree in the population, best Pareto rank first."},
  props: {
    objectives: "Objective values recomputed with today's code (see the guide for what each means).",
    entry: "Objective values the run itself recorded when the pedigree first appeared.",
    fit: "'fits' if every pair is within tolerance; otherwise the largest error over all pairs.",
    distances: "Edits and IBD diff to each reference (see the guide).",
    structure: "Number of latent people that matter (after pruning), and whether anyone is inbred.",
    count: "How many population slots hold this pedigree in this generation.",
    lineage: "Generation it first appeared, the operator that made it, and the candidates (#) it was made from."},
  objectives: {
    ibd_total: "sum over pairs of the pair's error (predicted vs observed relatedness)",
    ibd_worst: "largest error over all pairs",
    n_bad_pairs: "number of pairs whose error is above tolerance",
    excess_total: "sum over pairs of how far each error exceeds tolerance (0 for every fit)",
    excess_worst: "how far the largest error exceeds tolerance (0 for every fit)",
    n_latent: "number of latent people that matter (simpler pedigrees score lower)",
    inbreeding: "total inbreeding coefficient over the pedigree's people"},
  kinds: {
    ibd: "Pair error = largest of |ΔIBD0|, |ΔIBD1|, |ΔIBD2| between pedigree and data.",
    king: "Data are KING kinship and IBS0; IBD0 is estimated as IBS0 / IBS0 of unrelated pairs. " +
      "Pair error combines |Δkinship| and |ΔIBD0| (each within its own tolerance).",
    kinship: "Pair error = |Δkinship| between pedigree and data."},
};

HELP.css = `
.note { color:var(--muted); font-size:11.5px; margin:1px 0 6px; font-weight:400; }
aside label + .note { margin:-2px 0 5px 20px; }
body:not(.explain) .note, body:not(.explain) .guide { display:none; }
.guide { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:6px 12px; margin:8px 0; font-size:12.5px; }
.guide summary { cursor:pointer; font-weight:600; }
.guide dl { display:grid; grid-template-columns:max-content 1fr; gap:3px 12px; margin:6px 0; }
.guide dt { font-weight:600; }
.guide dd { margin:0; color:var(--muted); }
.sym { display:inline-block; width:14px; height:14px; vertical-align:-2px; margin-right:4px; }
.sym.obs { border-radius:50%; background:var(--accent); }
.sym.lat { border-radius:3px; background:var(--latent); border:1px solid var(--line); }
.sym.inb { border-radius:50%; border:3px solid var(--bad); }`;
(() => { const s = document.createElement("style"); s.textContent = HELP.css; document.head.appendChild(s); })();

function _helpEl(tag, props, ...kids) { const e = document.createElement(tag); Object.assign(e, props || {});
  kids.forEach(k => e.append(k)); return e; }

// A list of [term, text, symbol class] as a definition list.
HELP.dl = rows => { const dl = _helpEl("dl");
  rows.forEach(([term, text, sym]) => dl.append(
    _helpEl("dt", {}, ...(sym ? [_helpEl("span", {className:"sym " + sym})] : []), term), _helpEl("dd", {textContent:text})));
  return dl; };

// The "How to read this page" panel: drawing legend (minus terms in `omit`),
// `extra` rows, then the run's objectives (in ranking order) and pair error for `kind`.
HELP.guide = (extra = [], objectives = null, kind = null, omit = []) => {
  const d = _helpEl("details", {className:"guide", open:true}, _helpEl("summary", {textContent:"How to read this page"}));
  d.append(HELP.dl([...HELP.legend.filter(([, term]) => !omit.includes(term)).map(([sym, term, text]) => [term, text, sym]),
    ...extra]));
  d.objectives = _helpEl("div"); d.append(d.objectives);
  d.setObjectives = (objs, k) => { d.objectives.innerHTML = "";
    if (!objs || !objs.length) return;
    d.objectives.append(_helpEl("div", {}, _helpEl("b", {textContent:"Objectives "}),
      "(all minimized; listed in the order used for “best”). ", HELP.kinds[k] || ""));
    d.objectives.append(HELP.dl(objs.map(o => [o, HELP.objectives[o] || ""]))); };
  d.setObjectives(objectives, kind);
  return d; };

// Wire the "Explain" checkbox: toggles body.explain, remembered per browser.
HELP.explainSwitch = (checkbox, onchange) => {
  let on = true; try { const v = localStorage.getItem("pedviz:explain"); if (v !== null) on = JSON.parse(v); } catch (e) {}
  checkbox.checked = on;
  const apply = () => { document.body.classList.toggle("explain", checkbox.checked);
    try { localStorage.setItem("pedviz:explain", JSON.stringify(checkbox.checked)); } catch (e) {}
    if (onchange) onchange(); };
  checkbox.onchange = apply; apply(); };
