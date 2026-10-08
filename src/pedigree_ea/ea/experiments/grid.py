"""Grid training: run every combination of settings on a set of targets.

A grid spec (dict or JSON file) has:
    name     label used in file names
    cases    test-case names under `case_dir` (default "test_cases"); bounds
             (max_latent, tol) come from each case's answer key, so recall is
             measured against it
    files    real-data inputs: {"path", "target" ("king" | "ibd" | "kinship"),
             "max_latent" (int or list), "tol", "tol_ibd0", "key" (bool: compute
             a brute-force answer key), "name", "unrelated_ibs0",
             "references" ({label: pedigree TSV path}, stored with the runs and
             never shown to the search)}; paths are relative to the current directory
    seeds    number of seeds, or a list of seeds
    stopping Stopping fields, e.g. {"max_evals": 20000}
    fixed    RunConfig fields applied to every run
    grid     RunConfig fields -> list of values; every combination is run.
             `strategy` picks that strategy's default parameters and objectives;
             `strategy_params` dicts are merged onto those defaults;
             `objectives: null` keeps the strategy's default.

Example:
    {"name": "poss_vs_nsga2", "cases": ["avuncular"], "seeds": 3,
     "stopping": {"max_evals": 20000},
     "grid": {"representation": ["direct", "neat"], "strategy": ["nsga2", "poss"]}}

API:    run_grid("configs/x.json", workers=4)
        run_grids(["configs/x.json", "configs/y.json"], workers=8)   # one shared pool
Script: python scripts/grid.py configs/x.json [configs/y.json ...] --workers 8

With logging, all files of one call share one stamp: per-run logs/results (as
run_case writes them), and per grid logs/grid_<name>_<stamp>.jsonl (one line
per finished run) and results/grid_<name>_<stamp>.json / .csv / _summary.csv.
"""

from __future__ import annotations

import itertools
import shutil
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...data import io
from ...reference.brute_force import enumerate_pedigrees
from ...data.cases import Case, list_cases, load_case
from .logger import RunLogger, make_stamp
from .serialize import to_jsonable, write_csv, write_json
from ...store.codec import config_digest, config_label
from ...store.merge import merge
from .run import RunConfig, baseline_config, run_case
from ..engine import Stopping

DEFAULT_FILE_TOL = {"ibd": 0.15, "kinship": 0.05, "king": 0.05}


@dataclass
class GridSpec:
    name: str = "grid"
    grid: dict[str, list] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    cases: list[str] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    seeds: int | list[int] = 3
    stopping: dict[str, Any] = field(default_factory=lambda: {"max_evals": 20000})
    case_dir: str = "test_cases"

    @classmethod
    def from_dict(cls, d: dict) -> GridSpec:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown grid spec keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> GridSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def seed_list(self) -> list[int]:
        return list(range(self.seeds)) if isinstance(self.seeds, int) else list(self.seeds)


@dataclass
class GridResult:
    stamp: str
    rows: list[dict]
    summary: list[dict]
    paths: dict[str, str] = field(default_factory=dict)


# ---- settings ---------------------------------------------------------------

def expand(grid: dict[str, list]) -> list[dict[str, Any]]:
    """All combinations of the grid's values, in a stable order."""
    if not grid:
        return [{}]
    keys = list(grid)
    for k in keys:
        if not isinstance(grid[k], list) or not grid[k]:
            raise ValueError(f"grid[{k!r}] must be a non-empty list")
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _short(v: Any) -> str:
    if isinstance(v, dict):
        return ",".join(f"{k}={_short(x)}" for k, x in v.items()) or "{}"
    if isinstance(v, (list, tuple)):
        return "+".join(map(str, v))
    return "default" if v is None else str(v)


def setting_label(setting: dict[str, Any], grid: dict[str, list]) -> str:
    """Only the axes that vary are named, e.g. 'strategy=poss objectives=excess_total+n_latent'."""
    varying = [k for k in setting if len(grid.get(k, [])) > 1]
    return " ".join(f"{k}={_short(setting[k])}" for k in varying) or "single"


def make_config(setting: dict[str, Any], fixed: dict[str, Any], stopping: dict[str, Any],
                bounds: dict[str, Any], seed: int) -> RunConfig:
    """Bounds (max_latent, tol, tol_ibd0) come from the target; a value set in
    `fixed` or the grid overrides them (recall is then measured against an
    answer key made with other bounds)."""
    merged = {**bounds, **fixed, **setting}
    for k in ("seed", "stopping"):
        if k in merged:
            raise ValueError(f"set {k!r} with the spec's seeds/stopping, not in fixed or grid")
    rep = merged.pop("representation", "direct")
    strategy = merged.pop("strategy", None)
    if merged.get("objectives", "missing") is None:
        merged.pop("objectives")
    if "objectives" in merged:
        merged["objectives"] = tuple(merged["objectives"])
    return baseline_config(rep, strategy, **merged, seed=seed, stopping=Stopping(**stopping))


# ---- targets ----------------------------------------------------------------

def _file_bounds(f: dict, kind: str) -> tuple[list[int], float, float]:
    latents = f.get("max_latent", 2)
    return (latents if isinstance(latents, list) else [latents],
            f.get("tol", DEFAULT_FILE_TOL[kind]), f.get("tol_ibd0", 0.15))


def key_requests(spec: GridSpec) -> list[tuple]:
    """Answer keys the spec's files ask for, as hashable requests
    (path, target, unrelated_ibs0, max_latent, tol, tol_ibd0)."""
    reqs = []
    for f in spec.files:
        if not f.get("key", False):
            continue
        _, kind, _ = io.read_pairwise(f["path"], f.get("target", "king"), f.get("unrelated_ibs0"))
        latents, tol, tol_ibd0 = _file_bounds(f, kind)
        reqs += [(str(f["path"]), f.get("target", "king"), f.get("unrelated_ibs0"), k, tol, tol_ibd0)
                 for k in latents]
    return reqs


def compute_key(req: tuple) -> list:
    """Brute-force answer key for one request (top-level so workers can run it)."""
    path, target, unrelated_ibs0, k, tol, tol_ibd0 = req
    data, _, _ = io.read_pairwise(path, target, unrelated_ibs0)
    return enumerate_pedigrees(data, max_latent=k, tol=tol, tol_ibd0=tol_ibd0)


def load_targets(spec: GridSpec, log: Callable[[str], None] = print,
                 keys: dict[tuple, list] | None = None) -> list[tuple[Case, dict]]:
    """(case, bounds) per target; bounds = max_latent, tol (and tol_ibd0).
    Answer keys for files come from `keys` (see key_requests) or are computed here."""
    keys = {} if keys is None else keys
    targets = []
    available = {p.name: p for p in list_cases(spec.case_dir)}
    for name in spec.cases:
        if name not in available:
            raise ValueError(f"no test case {name!r} in {spec.case_dir}")
        case = load_case(available[name])
        key = case.info.get("answer_key")
        if not key:
            raise ValueError(f"test case {name!r} has no answer key (bounds unknown)")
        targets.append((case, {"max_latent": key["max_latent"], "tol": key["tol"]}))
    for f in spec.files:
        data, kind, note = io.read_pairwise(f["path"], f.get("target", "king"),
                                            f.get("unrelated_ibs0"))
        latents, tol, tol_ibd0 = _file_bounds(f, kind)
        for k in latents:
            name = f"{f.get('name', Path(f['path']).stem)}_L{k}"
            solutions = None
            if f.get("key", False):
                req = (str(f["path"]), f.get("target", "king"), f.get("unrelated_ibs0"), k, tol, tol_ibd0)
                if req not in keys:
                    t0 = time.time()
                    keys[req] = compute_key(req)
                    log(f"answer key for {name}: {len(keys[req])} pedigrees ({time.time() - t0:.1f}s)")
                solutions = keys[req]
            refs = {label: io.read_pedigree(path) for label, path in f.get("references", {}).items()}
            case = Case(name, note, None, data, {"source": str(f["path"]), "kind": kind,
                                                 "references": refs}, solutions)
            targets.append((case, {"max_latent": k, "tol": tol, "tol_ibd0": tol_ibd0}))
    if not targets:
        raise ValueError(f"grid spec {spec.name!r} has no cases and no files")
    return targets


# ---- running ----------------------------------------------------------------

def _run_one(task: dict) -> dict:
    """One run (top-level so worker processes can pickle it)."""
    cfg = task["config"]
    logger = (RunLogger(task["run_name"], task["log_dir"], task["results_dir"], stamp=task["stamp"],
                        db_path=task["shard"], grid=task["row"]["grid"]) if task["log"] else None)
    res, sc = run_case(cfg, task["case"], logger, task["case"].info.get("references"))
    return {**task["row"], **sc, "n_evals": res.n_evals, "seconds": round(res.seconds, 3),
            "stop_reason": res.stop_reason, "n_phenotypes": len(res.archive.phenotypes),
            "n_pareto": len(res.archive.pareto)}


def summarize(rows: list[dict]) -> list[dict]:
    """Per (target, setting): mean recall, truth found, fits, time."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["target"], r["setting_id"]), []).append(r)
    out = []
    for (target, sid), rs in sorted(groups.items()):
        def mean(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return round(statistics.fmean(vals), 4) if vals else None
        truth = [r["truth_at_eval"] for r in rs if r.get("truth_at_eval") is not None]
        out.append({"target": target, "setting_id": sid, "setting": rs[0]["setting"],
                    "runs": len(rs), "recall": mean("recall"),
                    "recall_outbred": mean("recall_outbred"), "n_fits": mean("n_fits"),
                    "truth_found": len(truth), "truth_eval_median":
                        statistics.median(truth) if truth else None,
                    "extra": sum(r.get("extra") or 0 for r in rs), "seconds": mean("seconds")})
    return out


def _as_spec(spec: GridSpec | dict | str | Path) -> GridSpec:
    if isinstance(spec, (str, Path)):
        return GridSpec.from_json(spec)
    if isinstance(spec, dict):
        return GridSpec.from_dict(spec)
    return spec


def _plan(spec: GridSpec, gi: int, targets, stamp, log, log_dir, results_dir) -> tuple[list, list]:
    settings = expand(spec.grid)
    tasks = []
    for case, bounds in targets:
        for sid, setting in enumerate(settings):
            label = setting_label(setting, spec.grid)
            for seed in spec.seed_list():
                cfg = make_config(setting, spec.fixed, spec.stopping, bounds, seed)
                d = cfg.to_dict()
                shard = (_shard_dir(spec, stamp, results_dir) /
                         f"{case.name}__{config_label(d)}__cfg-{config_digest(d)[:8]}__s{seed}.sqlite")
                tasks.append({
                    "grid": gi, "config": cfg, "case": case, "log": log, "stamp": stamp,
                    "shard": str(shard),
                    "log_dir": str(log_dir), "results_dir": str(results_dir),
                    "run_name": f"{spec.name}_{case.name}_g{sid:03d}_s{seed}",
                    "row": {"grid": spec.name, "target": case.name, "setting_id": sid,
                            "setting": label, "seed": seed,
                            **{k: _short(v) for k, v in setting.items()}},
                })
    return settings, tasks


def _shard_dir(spec: GridSpec, stamp: str, results_dir) -> Path:
    return Path(results_dir) / "db" / f"grid_{spec.name}_{stamp}_shards"


def run_grids(specs: list[GridSpec | dict | str | Path], workers: int = 1, log: bool = True,
              log_dir: str | Path = "logs", results_dir: str | Path = "results",
              stamp: str | None = None, verbose: bool = True,
              keep_shards: bool = False) -> list[GridResult]:
    """Run several grids, sharing one pool of `workers` processes.

    Runs from all grids are interleaved in the pool, and answer keys for data
    files are computed once per distinct (file, bounds) and in parallel. Each
    grid gets its own progress log, result files and GridResult; all files of
    one call share one stamp. With `log`, each run writes a database shard
    (results/db/grid_<name>_<stamp>_shards/<case>__<rep>-<strategy>__cfg-<hash>__s<seed>.sqlite)
    and each grid's shards are merged into results/db/grid_<name>_<stamp>.sqlite;
    shards are deleted afterwards unless `keep_shards`.
    """
    specs = [_as_spec(s) for s in specs]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"grid names must be unique (they name the output files): {names}")
    say = print if verbose else (lambda *a, **k: None)
    stamp = stamp or make_stamp()
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    progress: list = []
    try:
        # 1. answer keys, deduplicated across grids
        requests = list(dict.fromkeys(r for s in specs for r in key_requests(s)))
        keys: dict[tuple, list] = {}
        if requests:
            say(f"computing {len(requests)} answer key(s)")
            t0 = time.time()
            if pool:
                futures = {pool.submit(compute_key, r): r for r in requests}
                for fut in as_completed(futures):
                    keys[futures[fut]] = fut.result()
            else:
                keys = {r: compute_key(r) for r in requests}
            say(f"answer keys done ({time.time() - t0:.1f}s)")

        # 2. plan every grid
        plans = []
        for gi, spec in enumerate(specs):
            settings, tasks = _plan(spec, gi, load_targets(spec, say, keys), stamp, log,
                                    log_dir, results_dir)
            plans.append((spec, settings, tasks))
            say(f"grid {spec.name!r}: {len(tasks)} runs")
        all_tasks = [t for _, _, tasks in plans for t in tasks]
        say(f"{len(all_tasks)} runs in {len(specs)} grid(s), {max(workers, 1)} worker(s) "
            f"(stamp {stamp})")

        if log:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            progress = [open(Path(log_dir) / f"grid_{spec.name}_{stamp}.jsonl", "a")
                        for spec, _, _ in plans]
        rows: list[list[dict]] = [[] for _ in plans]
        n_done = 0

        def done(task: dict, row: dict) -> None:
            nonlocal n_done
            n_done += 1
            gi = task["grid"]
            rows[gi].append(row)
            if progress:
                progress[gi].write(json.dumps(to_jsonable(row)) + "\n")
                progress[gi].flush()
            rec = f"{row['recall']:.2f}" if row.get("recall") is not None else "-"
            prefix = f"{row['grid']} | " if len(specs) > 1 else ""
            say(f"[{n_done}/{len(all_tasks)}] {prefix}{row['target']} | {row['setting']} | "
                f"seed {row['seed']}: recall {rec}, fits {row['n_fits']}, "
                f"truth@{row['truth_at_eval']}, {row['seconds']}s")

        # 3. run everything
        if pool:
            futures = {pool.submit(_run_one, t): t for t in all_tasks}
            for fut in as_completed(futures):
                done(futures[fut], fut.result())
        else:
            for t in all_tasks:
                done(t, _run_one(t))
    finally:
        for f in progress:
            f.close()
        if pool:
            pool.shutdown()

    # 4. write each grid's results
    results = []
    for (spec, settings, _), grid_rows in zip(plans, rows):
        grid_rows.sort(key=lambda r: (r["target"], r["setting_id"], r["seed"]))
        summary = summarize(grid_rows)
        result = GridResult(stamp, grid_rows, summary)
        if log:
            out = Path(results_dir)
            out.mkdir(parents=True, exist_ok=True)
            base = out / f"grid_{spec.name}_{stamp}"
            write_json(f"{base}.json", {"name": spec.name, "stamp": stamp, "spec": asdict(spec),
                                        "settings": settings, "rows": grid_rows,
                                        "summary": summary})
            write_csv(Path(f"{base}.csv"), grid_rows)
            write_csv(Path(f"{base}_summary.csv"), summary)
            shard_dir = _shard_dir(spec, stamp, results_dir)
            db = merge(sorted(shard_dir.glob("*.sqlite")), out / "db" / f"grid_{spec.name}_{stamp}.sqlite")
            if not keep_shards:
                shutil.rmtree(shard_dir, ignore_errors=True)
            result.paths = {"json": f"{base}.json", "csv": f"{base}.csv",
                            "summary_csv": f"{base}_summary.csv", "db": str(db),
                            "progress": str(Path(log_dir) / f"grid_{spec.name}_{stamp}.jsonl")}
        results.append(result)
    return results


def run_grid(spec: GridSpec | dict | str | Path, workers: int = 1, log: bool = True,
             log_dir: str | Path = "logs", results_dir: str | Path = "results",
             stamp: str | None = None, verbose: bool = True) -> GridResult:
    """Run one grid (see run_grids). `spec` is a GridSpec, a dict, or a JSON path."""
    return run_grids([spec], workers, log, log_dir, results_dir, stamp, verbose)[0]


def format_summary(summary: list[dict]) -> str:
    """Plain-text table of a grid summary."""
    def fmt(v, spec=".2f"):
        return "-" if v is None else format(v, spec)
    lines = [f"{'target':24s} {'setting':44s} {'recall':>6s} {'outbr':>6s} {'truth':>6s} "
             f"{'fits':>6s} {'sec':>6s}"]
    for s in summary:
        lines.append(f"{s['target'][:24]:24s} {s['setting'][:44]:44s} {fmt(s['recall']):>6s} "
                     f"{fmt(s['recall_outbred']):>6s} {s['truth_found']:>2d}/{s['runs']:<3d} "
                     f"{fmt(s['n_fits'], '.1f'):>6s} {fmt(s['seconds'], '.1f'):>6s}")
    return "\n".join(lines)
