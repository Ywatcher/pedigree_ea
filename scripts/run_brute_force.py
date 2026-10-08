"""Enumerate compatible pedigrees without running an evolutionary algorithm."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pedigree_ea import enumerate_pedigrees, inbreeding, io


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--target", choices=["king", "ibd", "kinship"], default="king")
    ap.add_argument("--max-latent", type=int, default=2)
    ap.add_argument("--tol", type=float)
    ap.add_argument("--tol-ibd0", type=float, default=0.15)
    ap.add_argument("--unrelated-ibs0", type=float)
    ap.add_argument("--outbred-only", action="store_true")
    ap.add_argument("--max-results", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.max_latent < 0 or (args.max_results is not None and args.max_results < 1):
        ap.error("max-latent must be nonnegative and max-results must be positive")

    target, kind, note = io.read_pairwise(args.input, args.target, args.unrelated_ibs0)
    tol = args.tol if args.tol is not None else {"king": 0.05, "kinship": 0.05, "ibd": 0.15}[kind]
    if tol < 0 or args.tol_ibd0 <= 0:
        ap.error("tol must be nonnegative and tol-ibd0 must be positive")
    print(f"Input: {args.input}\nTarget: {note}\nObserved people: {target.ids}", flush=True)
    start = perf_counter()
    solutions = enumerate_pedigrees(
        target, max_latent=args.max_latent, tol=tol, tol_ibd0=args.tol_ibd0,
        check_sex=True, allow_inbreeding=not args.outbred_only,
        max_results=args.max_results,
    )
    summary = {
        "input": str(args.input.resolve()), "target": kind,
        "max_latent": args.max_latent, "tol": tol, "tol_ibd0": args.tol_ibd0,
        "check_sex": True, "allow_inbreeding": not args.outbred_only,
        "max_results": args.max_results,
        "complete_within_bounds": args.max_results is None or len(solutions) < args.max_results,
        "n_solutions": len(solutions),
        "n_outbred": sum(all(f <= 1e-12 for f in inbreeding(p).values()) for p in solutions),
        "seconds": perf_counter() - start,
    }
    if kind == "king":
        summary["unrelated_ibs0"] = target.unrelated_ibs0
    out = args.out or ROOT / "results" / f"brute_{args.input.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True, exist_ok=False)
    for i, pedigree in enumerate(solutions, 1):
        io.write_pedigree(pedigree, out / f"s{i:04d}.tsv")
        print(f"\nSolution {i}:\n{pedigree}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{json.dumps(summary, indent=2)}\nSaved to {out}")


if __name__ == "__main__":
    main()
