"""Explore a run-record database in the browser.

Usage (from the repo root):
    python scripts/visualize.py results/db/<file>.sqlite [--reference PATH[:LABEL] ...]
                                [--port 8765] [--open]

Pick a run, choose which sections (best, closest to a reference, removed
closest, latest fit, Pareto front, whole population) and properties to show,
and move through generations with the slider, the generation box, the step
buttons, the event list, or by clicking the progress chart. Click a candidate
for its per-pair observed vs predicted values and lineage.

References stored in the database (e.g. a test case's truth) are available
automatically; --reference adds pedigree TSVs (matched to runs with the same
observed people), e.g. references/task3_siblings.tsv:siblings.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedigree_ea import io  # noqa: E402
from pedigree_ea.webviz import serve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db", type=Path, help="run-record database (.sqlite)")
    ap.add_argument("--reference", action="append", default=[], help="PATH[:LABEL] pedigree TSV")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    args = ap.parse_args()
    if not args.db.exists():
        raise SystemExit(f"no such database: {args.db}")
    refs = {}
    for spec in args.reference:
        path, _, label = spec.partition(":")
        refs[label or Path(path).stem] = io.read_pedigree(path)
    serve(args.db, refs, args.host, args.port, args.open)


if __name__ == "__main__":
    main()
