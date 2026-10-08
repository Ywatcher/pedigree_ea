"""Search pedigrees for pairwise data read from a file.

Input formats (by extension / content):
    .kin0 / .kin   KING-robust table (PLINK2 --make-king-table or KING)
                   --target king     kinship + IBS0 (IBD0 = IBS0 / unrelated IBS0),
                                     the measured values only (default)
                   --target ibd      approximate IBD0/1/2 derived from kinship + IBS0
                   --target kinship  kinship only; negative values set to 0
    .genome        PLINK --genome (IBD0/1/2)
    .seg           KING --ibdseg (IBD0/1/2)
    .csv           this project's IBD CSV (id1,id2,ibd0,ibd1,ibd2)
Optionally --psam reads sample sexes (reported; not yet used by the search).

For each --max-latent bound, every implemented baseline runs for --seeds
seeds; with --key, brute force first computes the complete answer key (slow
beyond ~7 people in total) and recall is reported. All fitting pedigrees found
are pooled and the best are printed.

With --log, files go to logs/ and results/ with one shared stamp:
per-run logs and results, results/<name>_key_L<k>_<stamp>/ (answer key) and
results/<name>_found_L<k>_<stamp>/ (pooled pedigrees found). Run from the repo root.

Example (Task03):
    python scripts/run_on_file.py data/task03/task3_500kb_kin.kin0 --key --log
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from pedigree_ea import PedigreeSet, enumerate_pedigrees, inbreeding, io  # noqa: E402
from pedigree_ea.data.cases import Case  # noqa: E402
from pedigree_ea.ea import IMPLEMENTED, RunLogger, Stopping, baseline_config, make_stamp, run_case  # noqa: E402

DEFAULT_TOL = {"ibd": 0.15, "kinship": 0.05, "king": 0.05}


def is_outbred(ped) -> bool:
    return all(f < 1e-12 for f in inbreeding(ped).values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("--target", choices=["king", "ibd", "kinship"], default="king",
                    help="for KING tables: kinship + IBS0 (default), approximate IBD, or kinship")
    ap.add_argument("--unrelated-ibs0", type=float, help="IBS0 baseline for unrelated pairs")
    ap.add_argument("--psam", type=Path)
    ap.add_argument("--max-latent", type=int, nargs="*", default=[2])
    ap.add_argument("--tol", type=float,
                    help="fit tolerance (default 0.15 for IBD, 0.05 for kinship)")
    ap.add_argument("--tol-ibd0", type=float, default=0.15,
                    help="IBD0 tolerance for the king target (default 0.15)")
    ap.add_argument("--key", action="store_true", help="compute the brute-force answer key")
    ap.add_argument("--reps", nargs="*", default=list(IMPLEMENTED))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-evals", type=int, default=20000)
    ap.add_argument("--top", type=int, default=5, help="best pooled pedigrees to print")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    try:
        data, kind, note = io.read_pairwise(args.input, args.target, args.unrelated_ibs0)
    except ValueError as e:
        raise SystemExit(str(e))
    tol = args.tol if args.tol is not None else DEFAULT_TOL[kind]
    name = args.input.stem
    stamp = make_stamp()
    extra = f", tol_ibd0={args.tol_ibd0}" if kind == "king" else ""
    print(f"input: {args.input} ({note})\npeople: {', '.join(data.ids)}; tol={tol}{extra}")
    for (a, b), v in data.items():
        if kind == "kinship":
            shown = f"kinship {v:.4f}"
        elif kind == "king":
            shown = f"kinship {v.kinship:7.4f}  IBS0 {v.ibs0:.4f}  IBD0~{data.ibd0(a, b):.2f}"
        else:
            shown = "IBD0/1/2 " + "  ".join(f"{x:.2f}" for x in v)
        print(f"  {a}-{b}: {shown}")
    if args.psam:
        sexes = io.read_psam(args.psam)
        print("sexes (not yet used by the search):",
              ", ".join(f"{k}={v or '?'}" for k, v in sexes.items()))

    for k in args.max_latent:
        print(f"\n=== max_latent={k}")
        key = None
        if args.key:
            t0 = time.time()
            key = enumerate_pedigrees(data, max_latent=k, tol=tol, tol_ibd0=args.tol_ibd0)
            print(f"answer key: {len(key)} pedigrees ({sum(map(is_outbred, key))} outbred), "
                  f"{time.time() - t0:.1f}s")
            if args.log:
                d = Path("results") / f"{name}_key_L{k}_{stamp}"
                d.mkdir(parents=True, exist_ok=True)
                for i, p in enumerate(key, 1):
                    io.write_pedigree(p, d / f"s{i:04d}.tsv")
        case = Case(f"{name}_L{k}", note, None, data, {}, key)
        pooled = PedigreeSet()
        best: list = []
        for rep in args.reps:
            scores = []
            for seed in range(args.seeds):
                cfg = baseline_config(rep, max_latent=k, tol=tol, tol_ibd0=args.tol_ibd0, seed=seed,
                                      stopping=Stopping(max_evals=args.max_evals))
                logger = RunLogger(f"{rep}_{name}_L{k}_s{seed}", stamp=stamp) if args.log else None
                res, sc = run_case(cfg, case, logger)
                scores.append(sc)
                for f in res.archive.fits:
                    if pooled.add(f.pedigree):
                        best.append(f)
            line = f"  {rep:7s} fits found per run: {np.mean([s['n_fits'] for s in scores]):.1f}"
            if key is not None:
                outb = [s["recall_outbred"] for s in scores if s["recall_outbred"] is not None]
                line += (f"  recall {np.mean([s['recall'] for s in scores]):.2f}"
                         f"  outbred recall {np.mean(outb) if outb else float('nan'):.2f}")
            print(line, flush=True)
        print(f"pooled: {len(best)} distinct fitting pedigrees "
              f"({sum(is_outbred(f.pedigree) for f in best)} outbred)")
        # Rank: worst-pair error, then fewer latent people, outbred first.
        best.sort(key=lambda f: (f.objectives[1], f.objectives[2], not is_outbred(f.pedigree)))
        for i, f in enumerate(best[:args.top], 1):
            tag = "outbred" if is_outbred(f.pedigree) else "inbred"
            print(f"--- #{i}: worst-pair error {f.objectives[1]:.3f}, total {f.objectives[0]:.3f}, "
                  f"latent {int(f.objectives[2])}, {tag}")
            print("    " + str(f.pedigree).replace("\n", "\n    "))
        if args.log:
            d = Path("results") / f"{name}_found_L{k}_{stamp}"
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(best, 1):
                io.write_pedigree(f.pedigree, d / f"s{i:04d}.tsv")
    if args.log:
        print(f"\nfiles saved with stamp {stamp} in logs/ and results/")


if __name__ == "__main__":
    main()
