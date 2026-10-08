# Pedigree EA

Evolutionary search for **many distinct pedigrees compatible with pairwise genetic relationship estimates**. Observed people keep their identities; the search can introduce unobserved (latent) ancestors and compare alternative family structures.

This is a research project for comparing pedigree encodings, mutation operators, objectives, and search strategies. Small synthetic cases have bounded, exhaustive answer keys, so experiments can measure how many compatible structures a search recovers rather than only its best error.

## What is implemented

- Pedigree model, expected kinship, IBD0/1/2, Jacquard identity coefficients, and inbreeding; batched NumPy evaluation with an exact fallback for inbred pairs.
- Three encodings: direct parent arrays, NEAT-like edge genes, and CGP-like ordered parent genes.
- Three strategies: NSGA-II, `(1 + lambda)` with neutral drift and restarts, and a POSS-style Pareto search.
- An archive of distinct fitting pedigrees, deduplicated under renaming of latent people.
- Twelve synthetic test cases, brute-force reference enumeration, and recall scoring.
- PLINK/KING readers, parameter grids, SQLite run recording, animations, and a browser explorer with candidate lineage and pairwise residuals.

Cyclic graph and tree-GP encodings are design placeholders, not implemented search methods. The NEAT-like encoding does not yet implement speciation.

## Setup

Python **3.12 or newer** is required. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install pytest matplotlib pillow
```

The package declares NumPy, NetworkX, and DEAP as dependencies. Matplotlib and Pillow support visualizations; pytest is for tests. For JupyterLab, also install `jupyterlab` and `ipykernel` into the environment and select that environment's kernel.

The existing local Conda environment is named `pedigree_ea`. For notebook/API imports, install the package there as well:

```bash
conda activate pedigree_ea
python -m pip install -e .
```

Scripts and pytest add `src/` to the import path themselves; notebooks need the editable installation. External analysis tools such as KING, PRIMUS, and Ped-sim are kept in a separate environment.

## First experiment

Run one encoding on a small case with a known answer key:

```bash
python scripts/run_baselines.py --cases parent_child --reps direct --seeds 1 --max-evals 1000 --log
```

The table reports answer-key recall, outbred recall, whether the generating pedigree was found, evaluations until it was found, extra fits, and runtime. Recall is meaningful only for the answer key's recorded bounds and tolerance.

To compare all implemented encodings on selected cases:

```bash
python scripts/run_baselines.py --cases parent_child full_sibs avuncular --seeds 3 --log
```

Default baseline pairings are direct + NSGA-II without crossover, NEAT-like + NSGA-II with crossover, and CGP-like + `(1 + lambda)`.

### Python and notebooks

```python
from pedigree_ea.data.cases import load_case
from pedigree_ea.ea import Stopping, baseline_config, run_case

case = load_case("test_cases/parent_child")
config = baseline_config(
    "direct",
    max_latent=case.info["answer_key"]["max_latent"],
    tol=case.info["answer_key"]["tol"],
    seed=0,
    stopping=Stopping(max_evals=1000),
)
result, scores = run_case(config, case)
print(scores)
for fit in result.archive.fits:
    print(fit.pedigree)
```

Use `baseline_config("direct", strategy="poss", ...)` to select another strategy. `RunConfig` also exposes objectives, representation and strategy parameters, and recording settings. Stopping rules include evaluation/time limits, windows without new fits or Pareto points, and a target number of fits; windows count evaluations.

## Inputs and fit criteria

| Target | Observations per pair | Fit rule |
|---|---|---|
| IBD | IBD0, IBD1, IBD2 | Every component differs by at most `tol` |
| Kinship | Kinship coefficient | Absolute difference is at most `tol` |
| KING | Kinship and IBS0 | Kinship difference is at most `tol`; estimated IBD0 difference is at most `tol_ibd0` |

For KING targets, observed IBD0 is approximated as IBS0 divided by an unrelated-pair IBS0 baseline. The baseline can be supplied explicitly; otherwise it is estimated from low-kinship pairs. Negative KING kinship is treated as zero during evaluation. This approximation is distinct from directly measured IBD-state proportions.

Pedigrees have at most two unordered parents per person and no directed cycles. Missing parents represent unrelated unknown founders. Sex consistency is checked by default, but known sample sexes loaded with the CLI's `--psam` option are currently only reported, not used by the search. Inbreeding is allowed and can be included as an objective.

The default objectives minimize summed pairwise error (`ibd_total`), worst-pair error (`ibd_worst`), and relevant latent-person count (`n_latent`). Despite their names, the error objectives also work with kinship and KING targets. Optional objectives include `n_bad_pairs`, `excess_total`, `excess_worst`, and `inbreeding`; excess objectives ignore error within tolerance.

All three strategies default to `fits_first=True`: the population prioritizes fits when choosing survivors, separately from objectives, so latent complexity alone does not eliminate a fitting pedigree. NSGA-II and POSS remain subject to their population/archive capacities, and `(1 + lambda)` can restart. The shared results archive retains every distinct fit discovered, including candidates that do not survive selection, each with the genotype that produced it (for future strategies that use fits as parents). Set `strategy_params={"fits_first": False}` to compare with ordinary objective-based retention.

### Search data from a file

```bash
python scripts/run_on_file.py test_cases/full_sibs/ibd.csv --reps direct --max-latent 2 --tol 0.000001 --seeds 1 --max-evals 2000 --log
```

Supported CLI inputs are the project's IBD CSV, PLINK `--genome` `.genome` files, KING `--ibdseg` `.seg` tables, and KING-robust `.kin0`/`.kin` tables. A KING `.seg` table is not the same format as Ped-sim's segment output. For KING tables, choose `--target king` (default), `ibd` (approximate IBD), or `kinship`; use `--unrelated-ibs0` to supply the baseline. Add `--key` for exhaustive enumeration on a small problem.

For brute-force enumeration alone, without running an EA:

```bash
python scripts/run_brute_force.py ../bioinfo_tasks/Task03_IBD/task3_500kb_kin.kin0 --target king --max-latent 2 --tol 0.05 --tol-ibd0 0.15
```

This writes pedigree TSVs and a summary to a new directory in `results/`. Use `--outbred-only` to exclude inbreeding, or `--max-results N` to stop after N fits (then the answer key is potentially incomplete). `--out PATH` selects a new output directory. The Python API is `enumerate_pedigrees(target, max_latent=2, tol=0.05, tol_ibd0=0.15)`; the input can be IBD, kinship, or KING data. Enumeration is intended for small bounded problems, roughly six or seven total observed plus latent people, and has no built-in time limit or checkpointing.

The package also provides kinship CSV readers for use through the Python API. The top-level `example` file is an illustrative relationship chart, not an input-format specification.

## Test cases

Each `test_cases/<name>/` contains:

```text
case.json         description, observed IDs, enumeration bounds and statistics
truth.tsv         generating pedigree
ibd.csv           pairwise search input
solutions/*.tsv   compatible pedigrees enumerated within the recorded bounds
```

Cases cover parent–child, trio, full and half siblings, grandparents, avuncular relationships, first and double first cousins, three generations, siblings with children, two families, and related parents.

Pedigree TSV columns are `id`, `parent1`, `parent2`, `sex`, `observed`. Parent `0` means missing; sex is `M`, `F`, or `U`; observed is `1` or `0`. IBD CSV columns are `id1,id2,ibd0,ibd1,ibd2`.

Regenerate selected cases with:

```bash
python scripts/make_test_cases.py --only parent_child full_sibs
```

Enumeration grows rapidly. The generator caps total people at seven by default, which can restrict latent bounds below the truth's size. Check `case.json`, especially `complete_for_truth_size` and `truth_included`, before interpreting recall.

## Grids and recorded runs

```bash
python scripts/grid.py configs/grid_encodings.json --workers 4
```

Grid specs choose cases or files, seeds, stopping rules, fixed settings, and parameter axes. Several specs can share one worker pool. `grid_task3.json`, `grid_poss_vs_nsga2.json`, and `grid_fits_first.json` reference data in the sibling directory `../bioinfo_tasks/Task03_IBD/`; adjust those paths or remove the file targets if that data is unavailable.

Logged runs write progress JSONL to `logs/`, result JSON and summaries to `results/`, and SQLite records to `results/db/`. Grid database shards are merged into one database per grid. These output directories are git-ignored.

To inspect a database, replace the path below with one printed by a logged run:

```bash
python scripts/visualize.py results/db/<file>.sqlite
```

Visit `http://127.0.0.1:8765`. The explorer shows population history, best and fitting candidates, Pareto fronts, distances to references, removal events, per-pair predictions, and lineage. Reference pedigrees are used only for recording and analysis; the search does not see them.

For GIFs and a comparison HTML page:

```bash
python scripts/animate.py --case avuncular --methods direct direct:poss neat cgp --max-evals 2000
```

Generated animations go to `plt/`; the tracked figures there support the reports.

## Development and project map

```bash
python -m pytest
```

Optionally install `pytest-xdist` and use `python -m pytest -n auto` for parallel tests.

| Path | Purpose |
|---|---|
| `src/pedigree_ea/genetics/` | Pedigree model, exact and batched genetics, identity and distance |
| `src/pedigree_ea/data/` | Readers, synthetic pedigrees, saved test cases |
| `src/pedigree_ea/reference/` | Bounded exhaustive enumeration |
| `src/pedigree_ea/ea/` | Shared evaluator, archive, engine, encodings, strategies, experiments |
| `src/pedigree_ea/store/` | SQLite recording, merging, and replay |
| `src/pedigree_ea/webviz/` | Local browser explorer |
| `scripts/`, `configs/` | Command-line workflows and experiment grids |
| `tests/`, `test_cases/` | Automated checks and benchmark data |

See [the file guide](docs/files.md) for module details, [the setup and baseline report](docs/report_01_setup_baselines.md) and [the recording and population–archive report](docs/report_02_recorder_gap.md) for experiment analysis. Report results describe their recorded code versions and may predate current selection behavior. [problem.md](problem.md) states the goal; [ideas.md](ideas.md) collects design alternatives.

Open work includes cyclic and tree-GP encodings, known-sex constraints in search, explicit policies for inbreeding, a better observation-noise model, and measuring whether fit retention improves topology recall and continued exploration.
