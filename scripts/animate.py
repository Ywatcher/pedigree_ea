"""Animate how candidates evolve for several methods: runs go into a database
(results/db/anim_<target>_<stamp>.sqlite), then one GIF per method and a web page
to switch between methods are rendered from it. For interactive exploration use
scripts/visualize.py on the database.

Method = representation[:strategy[:objective+objective...]].

References (shown as fixed panels and tracked by the recorder, never seen by the
search): a test case's truth, plus any --reference PATH[:LABEL] (pedigree TSV).
Each generation records the best candidate and the --closest K candidates most
similar to each reference, with their objectives.

Examples (run from the repo root):
    python scripts/animate.py --case avuncular --methods direct direct:poss neat cgp
    python scripts/animate.py --file data/task03/task3_500kb_kin.kin0 \\
        --max-latent 2 --reference references/task3_siblings.tsv:siblings \\
        --methods direct direct:nsga2:excess_total+excess_worst+n_bad_pairs+n_latent

Outputs go to plt/: anim_<target>_<stamp>.html and anim_<target>_<method>_<stamp>.gif.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedigree_ea import io  # noqa: E402
from pedigree_ea.data.cases import Case, load_case  # noqa: E402
from pedigree_ea.ea.experiments.animate import animate_methods  # noqa: E402
from pedigree_ea.ea.experiments.grid import DEFAULT_FILE_TOL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="test case name in test_cases/")
    src.add_argument("--file", type=Path, help="data file (.kin0, .genome, .seg, IBD .csv)")
    ap.add_argument("--target", default="king", help="for KING tables: king | ibd | kinship")
    ap.add_argument("--reference", action="append", default=[],
                    help="PATH[:LABEL] of a reference pedigree TSV (repeatable)")
    ap.add_argument("--closest", type=int, default=1, help="candidates recorded per reference")
    ap.add_argument("--methods", nargs="+", default=["direct", "direct:poss", "neat", "cgp"])
    ap.add_argument("--max-latent", type=int, help="default: the case's answer key, or 2")
    ap.add_argument("--tol", type=float)
    ap.add_argument("--tol-ibd0", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-evals", type=int, default=20000)
    ap.add_argument("--out-dir", default="plt")
    ap.add_argument("--no-gif", action="store_true", help="only write the web page")
    args = ap.parse_args()

    if args.case:
        case = load_case(ROOT / "test_cases" / args.case)
        key = case.info.get("answer_key") or {}
        bounds = {"max_latent": args.max_latent if args.max_latent is not None
                  else key.get("max_latent", 2),
                  "tol": args.tol if args.tol is not None else key.get("tol", 0.01)}
    else:
        data, kind, note = io.read_pairwise(args.file, args.target)
        bounds = {"max_latent": args.max_latent if args.max_latent is not None else 2,
                  "tol": args.tol if args.tol is not None else DEFAULT_FILE_TOL[kind],
                  "tol_ibd0": args.tol_ibd0}
        case = Case(args.file.stem, note, None, data, {}, None)
    references = {}
    for spec in args.reference:
        path, _, label = spec.partition(":")
        references[label or Path(path).stem] = io.read_pedigree(path)
    out = animate_methods(case, args.methods, bounds, seed=args.seed, max_evals=args.max_evals,
                          out_dir=args.out_dir, gif=not args.no_gif, references=references,
                          record_closest=args.closest)
    for kind, paths in out.items():
        for p in paths:
            print(f"{kind}: {p}")


if __name__ == "__main__":
    main()
