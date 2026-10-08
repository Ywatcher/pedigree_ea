"""Grid training from JSON specs (see src/pedigree_ea/ea/experiments/grid.py for the format).

Several specs share one pool of worker processes; each grid still gets its own
summary and result files (all with one stamp).

Usage: python scripts/grid.py configs/a.json [configs/b.json ...] [--workers 8]
           [--seeds 3] [--max-evals 20000] [--no-log] [--quiet]
           [--results-dir results] [--log-dir logs]

Run from the repo root: logs/ and results/ are written there.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedigree_ea.ea.experiments.grid import GridSpec, format_summary, run_grids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("specs", type=Path, nargs="+", help="grid spec JSON files")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (shared by all grids)")
    ap.add_argument("--seeds", type=int, help="override the number of seeds (all grids)")
    ap.add_argument("--max-evals", type=int, help="override stopping.max_evals (all grids)")
    ap.add_argument("--no-log", action="store_true", help="write no files")
    ap.add_argument("--results-dir", default="results", help="where results and databases go")
    ap.add_argument("--log-dir", default="logs", help="where progress logs go")
    ap.add_argument("--quiet", action="store_true", help="only print the summaries")
    args = ap.parse_args()

    specs = []
    for path in args.specs:
        spec = GridSpec.from_json(path)
        if args.seeds is not None:
            spec.seeds = args.seeds
        if args.max_evals is not None:
            spec.stopping = {**spec.stopping, "max_evals": args.max_evals}
        specs.append(spec)
    results = run_grids(specs, workers=args.workers, log=not args.no_log, verbose=not args.quiet,
                        log_dir=args.log_dir, results_dir=args.results_dir)
    for spec, result in zip(specs, results):
        print(f"\n=== {spec.name}")
        print(format_summary(result.summary))
        for kind, path in result.paths.items():
            print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
