# What each file is for

Environment: conda env `pedigree_ea` (Python, numpy, networkx, deap, pytest, matplotlib).
External tools (PRIMUS, KING, Ped-sim, ...) go in a separate env, not this one.

Run everything from the repo root, e.g. `conda run -n pedigree_ea python -m pytest`.

## Top level

| Path | Purpose |
|---|---|
| `problem.md` | Problem statement: find many pedigrees compatible with pairwise IBD; stopping goals |
| `ideas.md` | Design ideas: genotype spaces, operators, mutation size, objectives |
| `example` | Example relationship chart and notes on test-data sources (not an input format) |
| `pyproject.toml` | Package metadata; pytest config (`src` on the path, tests in `tests/`) |
| `.gitignore` | Ignores bytecode, editor files, and run outputs (`logs/`, `results/`) |
| `docs/files.md` | This file |
| `docs/report.md` | Report on the current setup and baseline results (copy of the shared doc) |
| `plt/` | Figures used by the report |

## `src/pedigree_ea/` — core (independent of any EA design)

| File | Purpose |
|---|---|
| `__init__.py` | Public API re-exports of the core modules |
| `pedigree.py` | `Pedigree` data structure: people (observed/latent), 0–2 unordered parents, optional sex; validity checks (cycles, unknown parents, sex consistency); `pruned()` removes latent people that cannot affect observed relationships |
| `pairs.py` | `PairData`: generic container of values for unordered pairs of people |
| `kinship.py` | Exact expected kinship (recursive), inbreeding, `KinshipData`, kinship→degree helpers |
| `ibd.py` | Exact expected IBD0/1/2 (`expected_ibd`), Jacquard's 9 identity coefficients (backward gene dropping); `IBD`, `IBDData` (the search input); rough IBD from kinship + IBS0 |
| `king.py` | `KingData`: measured KING-robust kinship + IBS0 per pair, unrelated-IBS0 baseline, IBD0 estimate (IBS0 / baseline), conversions to kinship or approximate IBD |
| `batch.py` | Pedigrees as fixed-size `(N, 2)` parent arrays and **batched numpy evaluation**: cycle detection, kinship (Henderson's A = LDLᵀ), IBD (closed form for outbred pairs, exact fallback for inbred), latent relevance, sex conflicts; conversion to/from `Pedigree` |
| `canonical.py` | Structure identity: two pedigrees are the same if they differ only by latent-person names; `PedigreeSet` deduplicates |
| `brute_force.py` | Exhaustive enumeration of all pedigrees fitting a target (IBD or kinship) within bounds; ground truth for small cases |
| `synth.py` | Standard named pedigrees (`STANDARD`), random pedigree generator, exact or noisy observations |
| `cases.py` | Save/load test cases on disk (truth, IBD input, answer key) |
| `io.py` | File formats: pedigree TSV, IBD CSV, kinship CSV; readers for PLINK `--genome`, KING `--ibdseg`, KING-robust tables (PLINK2 `--make-king-table` `.kin0`: kinship + IBS0 as `KingData`, kinship only, or approximate IBD0/1/2) and PLINK2 `.psam` (ids, sex) |

## `src/pedigree_ea/ea/` — evolutionary search

| File | Purpose |
|---|---|
| `__init__.py` | Public API of the EA package |
| `problem.py` | `Problem`: IBD, kinship, or KING (kinship + IBS0) target, `max_latent`, `tol`, objectives (`ibd_total`, `ibd_worst`, `n_bad_pairs`, `n_latent`, `inbreeding`); batched evaluation with a cache; defines when a pedigree "fits" |
| `archive.py` | `Archive`: everything a run found, independent of the population — distinct phenotypes, fitting structures (with the evaluation they appeared at), Pareto front |
| `selection.py` | Pareto ranking, crowding distance, NSGA-II survivor selection, binary tournament |
| `engine.py` | `Engine` (shared bookkeeping: evaluate, archive, operator statistics, logging, stopping), `Stopping` rules, strategies `nsga2` and `one_plus_lambda` |
| `logger.py` | Optional `RunLogger`: per-generation progress to `logs/<name>_<stamp>.jsonl`, results to `results/<name>_<stamp>.json` (+ `_solutions/`); one stamp per run |
| `run.py` | `RunConfig`, baseline representation/strategy pairings (`BASELINES`), `run`, `run_case`, `score` (recall vs answer key, outbred recall, extra fits, evaluation at which the truth was found) |

## `src/pedigree_ea/ea/representations/` — genotype encodings

Every encoding decodes to the same `(N, 2)` parent array, so evaluation, archive and metrics are shared.

| File | Status | Purpose |
|---|---|---|
| `base.py` | — | `Representation` interface: required `random`, `decode`, `operators`; optional `crossover`, `distance`, `genotype_objectives`; shared `mutate` |
| `dag.py` | — | Helpers on one parent array; `Builder` adds edges while refusing cycles / third parents (used by decoders) |
| `direct.py` | implemented | Genotype = the parent array; operators from `ideas.md` (small/medium/major) |
| `neat.py` | implemented, open points marked | Edge genes with innovation numbers, start minimal and grow, crossover aligned by innovation. Open: latent identity across genomes, speciation, "fitter parent" under multiple objectives |
| `cgp.py` | implemented | Each person reserves 2 input genes (parents) pointing to earlier positions in an evolvable order; always acyclic; neutral inactive genes |
| `cyclic.py` | **not implemented** | Cyclic genotype + DAG decoder; open questions listed in the module docstring |
| `treegp.py` | **not implemented** | Tree GP (program builds the pedigree); open questions listed in the module docstring |
| `__init__.py` | — | `REPRESENTATIONS` registry, `IMPLEMENTED` list |

## `scripts/`

| File | Purpose |
|---|---|
| `make_test_cases.py` | Regenerates `test_cases/` from `synth.STANDARD`: exact IBD + brute-force answer keys |
| `run_baselines.py` | Runs each implemented representation's baseline on the test cases and prints recall; `--log` writes `logs/` and `results/` with one shared stamp |
| `run_on_file.py` | Runs the search on real data from a file (`.kin0`/`.kin` as kinship + IBS0 by default, `.genome`, `.seg`, IBD `.csv`; optional `--psam`); `--key` adds a brute-force answer key; prints the best pedigrees found; `--log` saves files |

## `tests/`

| File | Covers |
|---|---|
| `test_pedigree.py` | Pedigree structure, validity, pruning |
| `test_kinship.py` | Kinship values for standard relationships, inbreeding, degrees |
| `test_ibd.py` | IBD values; exact vs simulation; fast path vs full Jacquard; monotonicity |
| `test_canonical.py` | Structure identity and deduplication |
| `test_io.py` | File round trips, PLINK/KING readers |
| `test_synth.py` | Standard and random pedigrees, observations |
| `test_brute_force.py` | Enumeration with kinship and IBD targets |
| `test_case_files.py` | Saved `test_cases/` are consistent (truth reproduces input; every solution fits) |
| `test_king.py` | KING kinship + IBS0 target: reading, IBD0 estimate, siblings vs parent-child, brute force and EA agreement |
| `test_batch.py` | Batched evaluation equals exact code; conversions; pruning count; sex conflicts |
| `test_ea.py` | Contract tests for every implemented representation; selection; problem; baselines; logger files |

## Data and outputs (outside `src/`)

| Path | Purpose |
|---|---|
| `test_cases/<name>/` | One clean test case: `case.json` (description, settings, answer-key stats), `truth.tsv`, `ibd.csv` (search input), `solutions/*.tsv` (answer key) |
| `logs/` | Run progress logs (`*.jsonl`), git-ignored, created on first write |
| `results/` | Run results (`*.json`, `*_solutions/`) and batch summaries, git-ignored |
