"""Run the baseline of each implemented representation on the test cases.

Each case is run with its answer key's bounds (max_latent, tol), so recall is
measured against the complete solution set.

With --log, every run writes logs/<rep>_<case>_s<seed>_<stamp>.jsonl and
results/<rep>_<case>_s<seed>_<stamp>.json (+ _solutions/), and the table is
saved as results/baselines_<stamp>.json; all files of one invocation share the
same stamp. Run from the repo root.

Usage: python scripts/run_baselines.py [--reps direct neat cgp] [--cases ...]
                                       [--seeds 3] [--max-evals 20000] [--log]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from pedigree_ea.data.cases import list_cases, load_case  # noqa: E402
from pedigree_ea.ea import IMPLEMENTED, RunLogger, Stopping, baseline_config, make_stamp, run_case  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", nargs="*", default=list(IMPLEMENTED))
    ap.add_argument("--cases", nargs="*", help="case names (default: all with an answer key)")
    ap.add_argument("--case-dir", default=str(ROOT / "test_cases"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-evals", type=int, default=20000)
    ap.add_argument("--log", action="store_true", help="write logs/ and results/ files")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    stamp = make_stamp()
    rows = []
    cases = [load_case(p) for p in list_cases(args.case_dir)]
    cases = [c for c in cases if c.info.get("answer_key") and c.solutions
             and (not args.cases or c.name in args.cases)]
    print(f"{'case':22s} {'rep':7s} {'recall':>7s} {'outbred':>7s} {'truth':>6s} "
          f"{'evals->truth':>12s} {'extra':>5s} {'sec':>6s}")
    for case in cases:
        ak = case.info["answer_key"]
        for rep in args.reps:
            scores = []
            for seed in range(args.seeds):
                cfg = baseline_config(rep, max_latent=ak["max_latent"], tol=ak["tol"], seed=seed,
                                      stopping=Stopping(max_evals=args.max_evals))
                logger = (RunLogger(f"{rep}_{case.name}_s{seed}", args.log_dir, args.results_dir,
                                    stamp=stamp) if args.log else None)
                res, sc = run_case(cfg, case, logger)
                sc.update(case=case.name, rep=rep, seed=seed, seconds=res.seconds)
                scores.append(sc)
                rows.append(sc)
            truth = [s["truth_at_eval"] for s in scores if s["truth_at_eval"] is not None]
            outb = [s["recall_outbred"] for s in scores if s["recall_outbred"] is not None]
            print(f"{case.name:22s} {rep:7s} {np.mean([s['recall'] for s in scores]):7.2f} "
                  f"{(np.mean(outb) if outb else float('nan')):7.2f} "
                  f"{len(truth)}/{len(scores):<4d} "
                  f"{(np.median(truth) if truth else float('nan')):12.0f} "
                  f"{sum(s['extra'] for s in scores):5d} "
                  f"{np.mean([s['seconds'] for s in scores]):6.1f}", flush=True)
    if args.log:
        out = Path(args.results_dir) / f"baselines_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"stamp": stamp, "args": vars(args), "runs": rows},
                                  indent=2, default=str) + "\n")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
