"""Write exact (noise-free) IBD test cases for the standard pedigrees.

For each pedigree in synth.STANDARD: the truth pedigree, its exact IBD among
observed people, and an answer key of every pedigree with identical expected
IBD (within --tol), up to the number of latent people the truth has. The answer
key is skipped or capped when observed + latent would exceed --max-people,
since brute force grows very fast.

Usage: python scripts/make_test_cases.py [--out test_cases] [--max-people 7]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedigree_ea import enumerate_pedigrees, expected_ibd, inbreeding, synth  # noqa: E402
from pedigree_ea.canonical import PedigreeSet  # noqa: E402
from pedigree_ea.cases import Case, save_case  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="test_cases")
    ap.add_argument("--max-people", type=int, default=7,
                    help="cap on observed + latent people for the answer key")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--only", nargs="*", help="case names to (re)generate")
    args = ap.parse_args()

    for name, make in synth.STANDARD.items():
        if args.only and name not in args.only:
            continue
        truth = make()
        target = expected_ibd(truth)
        n_obs, n_latent = len(truth.observed_ids), len(truth.latent_ids)
        max_latent = min(n_latent, args.max_people - n_obs)
        info = {"source": f"synth.STANDARD['{name}']", "noise": None}
        solutions = None
        if max_latent >= 0:
            t0 = time.time()
            solutions = enumerate_pedigrees(target, max_latent=max_latent, tol=args.tol)
            seconds = time.time() - t0
            found = PedigreeSet()
            for s in solutions:
                found.add(s)
            n_outbred = sum(all(f < 1e-12 for f in inbreeding(s).values()) for s in solutions)
            info["answer_key"] = {
                "method": "brute_force.enumerate_pedigrees",
                "max_latent": max_latent, "tol": args.tol,
                "check_sex": True, "allow_inbreeding": True,
                "complete_for_truth_size": max_latent == n_latent,
                "n_solutions": len(solutions), "n_outbred": n_outbred,
                "truth_included": truth in found,
                "seconds": round(seconds, 2),
            }
        else:
            info["answer_key"] = None
        case = Case(name, (make.__doc__ or name.replace("_", " ")).strip(), truth, target,
                    info, solutions)
        save_case(case, args.out)
        ak = info["answer_key"]
        summary = ("no answer key" if ak is None else
                   f"{ak['n_solutions']} solutions ({ak['n_outbred']} outbred), "
                   f"max_latent={max_latent}/{n_latent}, truth included={ak['truth_included']}, "
                   f"{ak['seconds']}s")
        print(f"{name:22s} {summary}", flush=True)


if __name__ == "__main__":
    main()
