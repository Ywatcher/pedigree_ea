"""Writing experiment outputs: JSON-safe values and CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(x: Any) -> Any:
    """Convert numpy values, tuples and dict keys so json.dumps accepts them;
    NaN and infinity become None."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_jsonable(obj), indent=indent) + "\n")
    return path


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    """Rows as CSV; columns = union of keys in first-seen order; nested values as JSON."""
    path = Path(path)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(to_jsonable(v)) if isinstance(v, (dict, list, tuple)) else v
                        for k, v in r.items()})
    return path
