"""Optional file logging for runs.

One run = one timestamp stamp shared by all its files:
    results/db/<name>_<stamp>.sqlite       the run record (default; see store/): every
                                           population member and fit stored once,
                                           population changes per generation, lineage.
                                           Pictures and web pages are made from it.
    logs/<name>_<stamp>.jsonl              progress, one JSON record per generation
    results/<name>_<stamp>.json            config, summary, scores, fits
    results/<name>_<stamp>_solutions/      fitting pedigrees as .tsv (only with save_solutions)

Folders default to `logs/` and `results/` under the current directory (run
scripts from the repo root); they are created on first write.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ...data import io
from .serialize import to_jsonable


def make_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class RunLogger:
    def __init__(self, name: str, log_dir: str | Path = "logs",
                 results_dir: str | Path = "results", stamp: str | None = None,
                 save_solutions: bool = False, db: bool = True, db_path: str | Path | None = None,
                 grid: str | None = None, keyframe_every: int = 50, scope: str = "population"):
        """`db`: write the run record database (default `results/db/<name>_<stamp>.sqlite`,
        or `db_path`). `scope`: "population" (members and fits) or "all" (every
        distinct evaluated candidate)."""
        self.name = name
        self.stamp = stamp or make_stamp()
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.save_solutions = save_solutions
        self.db = db
        self._db_path = Path(db_path) if db_path else None
        self.grid = grid
        self.keyframe_every = keyframe_every
        self.scope = scope
        self.db_run_id: int | None = None
        self._log_file = None

    @property
    def base(self) -> str:
        return f"{self.name}_{self.stamp}"

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"{self.base}.jsonl"

    @property
    def db_file(self) -> Path:
        return self._db_path or self.results_dir / "db" / f"{self.base}.sqlite"

    @property
    def result_path(self) -> Path:
        return self.results_dir / f"{self.base}.json"

    def log(self, record: dict) -> None:
        if self._log_file is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "a")
        self._log_file.write(json.dumps(to_jsonable(record)) + "\n")
        self._log_file.flush()

    def save_result(self, result: dict, fits=None) -> Path:
        """Write the result summary; `fits` (list of archive.Found) go to the solutions folder."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        out = {"name": self.name, "stamp": self.stamp, "log": str(self.log_path), **result}
        if self.db:
            out["db"] = str(self.db_file)
            out["db_run_id"] = self.db_run_id
        if fits and self.save_solutions:
            sol_dir = self.results_dir / f"{self.base}_solutions"
            sol_dir.mkdir(exist_ok=True)
            for k, f in enumerate(fits, 1):
                io.write_pedigree(f.pedigree, sol_dir / f"s{k:04d}.tsv")
            out["solutions_dir"] = str(sol_dir)
        self.result_path.write_text(json.dumps(to_jsonable(out), indent=2) + "\n")
        return self.result_path

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
