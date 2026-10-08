"""Run several methods on one target into a run-record database, then render
a GIF per method and one comparison web page from it (see pedigree_ea.viz).
For interactive exploration, open the database with scripts/visualize.py.

A method is written "representation[:strategy[:objective+objective...]]", e.g.
"direct", "direct:poss", "neat:nsga2:excess_total+excess_worst+n_latent".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

from ... import viz
from ...data.cases import Case
from ...genetics.pedigree import Pedigree
from ...store.reader import Database
from ..engine import Stopping
from .logger import RunLogger, make_stamp
from .run import RunConfig, baseline_config, case_references, run_case


def parse_method(spec: str, **overrides) -> RunConfig:
    parts = spec.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"method must be rep[:strategy[:objectives]], got {spec!r}")
    rep = parts[0]
    strategy = parts[1] if len(parts) > 1 and parts[1] else None
    if len(parts) == 3:
        overrides["objectives"] = tuple(parts[2].split("+"))
    return baseline_config(rep, strategy, **overrides)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def render_runs(db_path: str | Path, run_ids: Sequence[int], labels: Sequence[str], out_dir: str | Path,
                name: str, stamp: str, references: Mapping[str, Pedigree] | None = None,
                record_closest: int = 1, gif: bool = True, html: bool = True,
                summaries: Sequence[dict] | None = None) -> dict[str, list[Path]]:
    """GIF per run and one comparison page, made from a run-record database.
    Stored references (e.g. truth) are used, plus `references`."""
    db = Database(db_path)
    out_dir = Path(out_dir)
    written: dict[str, list[Path]] = {"html": [], "gif": []}
    entries, refs = [], {}
    for i, (rid, label) in enumerate(zip(run_ids, labels)):
        view = db.run(rid, references)
        refs = view.refs
        snaps = view.snapshots(list(refs), record_closest)
        entries.append({"label": label, "objective_names": view.objective_names, "snapshots": snaps,
                        "summary": (summaries[i] if summaries else view.summary) or {}})
        if gif:
            path = out_dir / f"anim_{_slug(name)}_{_slug(label)}_{stamp}.gif"
            viz.write_gif(path, snaps, view.ids, refs, title=f"{name} · {label}",
                          objective_names=view.objective_names)
            written["gif"].append(path)
    if html and entries:
        path = out_dir / f"anim_{_slug(name)}_{stamp}.html"
        viz.write_html(path, entries, view.ids, refs, title=f"How candidates evolve: {name}")
        written["html"].append(path)
    return written


def animate_methods(case: Case, methods: Sequence[str], bounds: dict, seed: int = 0,
                    max_evals: int = 20000, out_dir: str | Path = "plt", gif: bool = True,
                    html: bool = True, references: Mapping[str, Pedigree] | None = None,
                    record_closest: int = 1, results_dir: str | Path = "results",
                    log_dir: str | Path = "logs", stamp: str | None = None,
                    verbose: bool = True) -> dict[str, list[Path]]:
    """Run each method on `case` (bounds: max_latent, tol[, tol_ibd0]) into
    results/db/anim_<case>_<stamp>.sqlite, then render plt/anim_<case>_<stamp>.html
    and plt/anim_<case>_<method>_<stamp>.gif from it.

    References = the case's truth (label "truth") plus `references`; they are
    stored with the runs, never shown to the search. Returns
    {"db": [path], "html": [...], "gif": [...]}.
    """
    stamp = stamp or make_stamp()
    refs = {label: ped.pruned() for label, ped in case_references(case, references).items()}
    db_path = Path(results_dir) / "db" / f"anim_{_slug(case.name)}_{stamp}.sqlite"
    run_ids, summaries = [], []
    for spec in methods:
        cfg = parse_method(spec, **bounds, seed=seed, stopping=Stopping(max_evals=max_evals))
        logger = RunLogger(f"anim_{_slug(case.name)}_{_slug(spec)}", log_dir, results_dir,
                           stamp=stamp, db_path=db_path)
        res, sc = run_case(cfg, case, logger, references)
        summary = {"fits found": sc["n_fits"], "evaluations": res.n_evals,
                   "seconds": round(res.seconds, 1)}
        if sc.get("recall") is not None:
            summary["recall"] = round(sc["recall"], 2)
        if case.truth is not None:
            summary["truth found at evaluation"] = sc["truth_at_eval"]
        if verbose:
            print(f"{spec}: " + ", ".join(f"{k} {v}" for k, v in summary.items()), flush=True)
        run_ids.append(logger.db_run_id)
        summaries.append(summary)
    out = render_runs(db_path, run_ids, list(methods), out_dir, case.name, stamp, refs,
                      record_closest, gif, html, summaries)
    return {"db": [db_path], **out}
