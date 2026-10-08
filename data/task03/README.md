# Task03 pairwise measurements

`task3_500kb_kin.kin0` is the KING-robust kinship/IBS0 table for the five Task03 samples (100, 200, 300, 400, and 500). It is copied unchanged from the original local Task03 analysis file, `../bioinfo_tasks/Task03_IBD/task3_500kb_kin.kin0`.

The local copy is the input used by Task03 grid configurations and examples; the original analysis directory is not required. The comparison pedigree is `references/task3_siblings.tsv` at the repository root. This table contains measurements, not a known true pedigree.

From the repository root:

```bash
python scripts/run_brute_force.py data/task03/task3_500kb_kin.kin0 --target king --max-latent 2 --tol 0.05 --tol-ibd0 0.15
```
