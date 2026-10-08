# What each file is for

Environment: conda env `pedigree_ea` (Python, numpy, networkx, deap, pytest, pytest-xdist, matplotlib).
External tools (PRIMUS, KING, Ped-sim, ...) go in a separate env, not this one.

Run everything from the repo root, e.g. `conda run -n pedigree_ea python -m pytest` (add `-n auto` to run tests in parallel with pytest-xdist).

## Top level

| Path | Purpose |
|---|---|
| `problem.md` | Problem statement: find many pedigrees compatible with pairwise IBD; stopping goals |
| `ideas.md` | Design ideas: genotype spaces, operators, mutation size, objectives |
| `example` | Example relationship chart and notes on test-data sources (not an input format) |
| `pyproject.toml` | Package metadata; pytest config (`src` on the path, tests in `tests/`) |
| `.gitignore` | Ignores bytecode, editor files, and run outputs (`logs/`, `results/`) |
| `docs/files.md` | This file |
| `docs/report_01_setup_baselines.md` | Report 1: setup and baseline results |
| `docs/report_02_recorder_gap.md` | Report 2: recorder design for diagnosing the EA, current results, and the population–archive gap |
| `plt/` | Figures used by the reports (tracked); `plt/anim_*` animations from runs (git-ignored) |
| `configs/` | Grid specs for `scripts/grid.py` (`grid_poss_vs_nsga2.json`, `grid_encodings.json`, `grid_task3.json`, `grid_fits_first.json`) |

## `src/pedigree_ea/` — package layout

```
pedigree_ea/
  genetics/      pedigree model and exact genetics
  data/          file formats, test cases on disk, synthetic pedigrees
  reference/     exact brute-force answer keys
  ea/            evolutionary search
    representations/   genotype encodings
    strategies/        search strategies
    experiments/       runs, grids, animations, logging
  store/         run-record database (SQLite): write, merge, read back
  webviz/        browser explorer for run-record databases
  viz.py         drawing and animating pedigrees
```

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports the most used names; `from pedigree_ea import io, synth, cases, batch` works |
| `viz.py` | Pedigree layout and drawing; GIFs and a self-contained web page (method selector, play, slider) of how candidates evolve, with reference, closest and removed-closest panels |

## `src/pedigree_ea/genetics/` — pedigree model and exact genetics

| File | Purpose |
|---|---|
| `pedigree.py` | `Pedigree`: people (observed/latent), 0–2 unordered parents, optional sex; validity checks (cycles, unknown parents, sex consistency); `pruned()` removes latent people that cannot affect observed relationships |
| `pairs.py` | `PairData`: values for unordered pairs of people |
| `kinship.py` | Exact expected kinship (recursive), inbreeding, `KinshipData`, kinship→degree helpers |
| `ibd.py` | Exact expected IBD0/1/2 (`expected_ibd`), Jacquard's 9 identity coefficients (backward gene dropping); `IBD`, `IBDData`; rough IBD from kinship + IBS0 |
| `king.py` | `KingData`: measured KING-robust kinship + IBS0 per pair, unrelated-IBS0 baseline, IBD0 estimate (IBS0 / baseline) |
| `canonical.py` | Structure identity: same pedigree up to latent names; `PedigreeSet` deduplicates; `canonical_graph` / `canonical_form`: exact canonical bytes of any directed graph (cycles and weights allowed), the database identity of a pedigree |
| `similarity.py` | `structure_distance` (fewest parent-child edits, latent people matched) and `relationship_distance` (largest IBD difference over observed pairs) |
| `batch.py` | Fixed-size `(N, 2)` parent arrays and **batched numpy evaluation**: cycles, kinship (A = LDLᵀ), IBD (closed form for outbred pairs, exact fallback), latent relevance, sex conflicts; conversion to/from `Pedigree` |

## `src/pedigree_ea/data/` — data in and out

| File | Purpose |
|---|---|
| `io.py` | Pedigree TSV, IBD CSV, kinship CSV; PLINK `--genome`, KING `--ibdseg`, KING-robust `.kin0` (as `KingData`, kinship only, or approximate IBD), PLINK2 `.psam`; `read_pairwise` picks the reader by file type |
| `cases.py` | Save/load test cases on disk (truth, input, answer key) |
| `synth.py` | Standard named pedigrees (`STANDARD`), random pedigree generator, exact or noisy observations |

## `src/pedigree_ea/reference/`

| File | Purpose |
|---|---|
| `brute_force.py` | Exhaustive enumeration of all pedigrees fitting a target (IBD, kinship or KING) within bounds; answer keys for small cases |

## `src/pedigree_ea/store/` — run-record database

Each distinct pedigree is stored once; per-run tables only index candidates. Stored: identity, inputs, runtime facts (population changes, lineage, objectives at entry, when a candidate became a fit), provenance. Everything else is derived on read.

| File | Purpose |
|---|---|
| `schema.py` | Tables (shared: people sets, candidates, targets, configs, code versions, references; per run: runs, generations, membership changes, keyframes, appearances), `SCHEMA_VERSION`, `connect` |
| `codec.py` | Encoding targets, configs (digest without the seed), population blobs, code version |
| `writer.py` | `RunWriter`: writes one run (used by the Engine through `RunLogger`) |
| `merge.py` | `merge`: combines databases by content key (grid shards → one database per grid) |
| `reader.py` | `Database` / `RunView`: population at any generation, recomputed objectives, best / Pareto / closest to any reference / removal events, per-pair values, lineage, snapshots for pictures |

## `src/pedigree_ea/webviz/` — browser explorer

| File | Purpose |
|---|---|
| `server.py` | Local HTTP server and JSON API over a database (runs, series, frames, candidates) |
| `index.html` | The page: run selector; references, sections, properties and chart series to show; slider, step buttons, generation box, event jumps and clickable progress chart; candidate detail with per-pair observed vs predicted values and lineage; `?run=&gen=` links; an Explain switch for the notes and guide from `help.js` |
| `help.js` | Explanations of every part of the pages (drawing legend, sections, properties, objectives, pair error per data type), the "How to read this page" guide and the Explain switch; served by the explorer and inlined into `viz.write_html` pages |

## `src/pedigree_ea/ea/` — evolutionary search

| File | Purpose |
|---|---|
| `__init__.py` | Public API of the EA package |
| `problem.py` | `Problem`: IBD, kinship or KING target, `max_latent`, `tol` (+ `tol_ibd0`), objectives (`ibd_total`, `ibd_worst`, `n_bad_pairs`, `excess_total`, `excess_worst`, `n_latent`, `inbreeding`); batched evaluation with a cache; defines when a pedigree fits |
| `archive.py` | `Archive`: everything a run found — distinct phenotypes, fitting structures (with the evaluation they appeared at), Pareto front |
| `pareto.py` | Dominance, Pareto fronts, crowding distance, NSGA-II survivor selection, binary tournament; `fit_first_ranks` (fits before non-fits, fits not competing) and `select_by_ranks` |
| `engine.py` | `Engine` (shared bookkeeping: evaluate, archive, operator statistics, logging, snapshots, stopping), `Stopping`, `RunResult` |
| `recording.py` | What snapshots record (hidden from the search): best candidate, the candidates closest to each reference with objectives and distances, and removal events when the closest candidate leaves the population |

## `src/pedigree_ea/ea/strategies/` — search strategies

Each strategy is a function `strategy(engine, **params)` that only decides what to vary and what survives; the shared `Engine` does the rest. Each module also defines `NAME`, `DEFAULTS` and `OBJECTIVES`.

| File | Purpose |
|---|---|
| `base.py` | The strategy contract: what a strategy may call on the `Engine`, and the module constants |
| `nsga2.py` | NSGA-II: (mu + lambda) with Pareto fronts and crowding, optional crossover, duplicates pushed back |
| `one_plus_lambda.py` | (1 + lambda) with Pareto acceptance (neutral drift) and restarts; the CGP baseline |
| `poss.py` | POSS-style: archive of non-dominated solutions as the population, starts empty, keeps distinct equally scored pedigrees, multi-edit mutation |
| `__init__.py` | Registry: `STRATEGIES`, `STRATEGY_DEFAULTS`, `STRATEGY_OBJECTIVES`; add a new module to `_MODULES` |

## `src/pedigree_ea/ea/experiments/` — running and recording experiments

| File | Purpose |
|---|---|
| `run.py` | `RunConfig` (incl. `snapshot`, `record_closest`, `record_rank_by`), baseline pairings, `baseline_config`, `run` / `run_case` (references for the recorder), `score` (recall, outbred recall, extra fits, when the truth was found) |
| `grid.py` | Grid training API: `GridSpec` (cases, files, seeds, stopping, fixed settings, grid axes); `run_grids` runs several grids in one shared process pool (answer keys computed once and in parallel), `run_grid` runs one; per-run database shards merged into one database per grid; summaries |
| `animate.py` | Runs selected methods on one case or file into a database, then renders a GIF per method and a comparison web page to `plt/` from it (`render_runs` works on any database) |
| `logger.py` | Optional `RunLogger`: the run record database (default, `results/db/<name>_<stamp>.sqlite`), per-generation progress to `logs/<name>_<stamp>.jsonl`, results to `results/<name>_<stamp>.json`; TSV solutions only with `save_solutions` |
| `serialize.py` | JSON-safe conversion, JSON and CSV writing shared by logger and grid |

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
| `animate.py` | `--case` or `--file`, `--methods rep[:strategy[:objectives]]`, `--reference PATH[:LABEL]` (repeatable), `--closest K`, `--rank-by`; writes `plt/anim_*.gif` and `plt/anim_<target>_<stamp>.html` |
| `grid.py` | Grid training from one or more JSON specs in `configs/`, sharing one pool (`--workers`); prints a summary per grid and writes `logs/` and `results/` |
| `visualize.py` | Opens a run database in the browser explorer (`--reference PATH:LABEL` adds references, `--open` starts a browser) |
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
| `test_viz.py` | Layout, snapshots, GIF and web page output |
| `test_grid.py` | Grid expansion, configs, file output, parallel = serial, several grids in one pool, shared answer keys; POSS finds both parent-child orientations; excess objectives |
| `test_batch.py` | Batched evaluation equals exact code; conversions; pruning count; sex conflicts |
| `test_store.py` | Canonical form; writer → reader (population replay, fits, lineage, events equal in-memory recording); references added later; merge deduplication; grid database; web API |
| `test_fits_first.py` | Fit status keeps candidates: fit-first ranks, duplicates by structure, NSGA-II holds the truth once found, POSS keeps every fit, (1+λ) never leaves a fit |
| `test_recording.py` | Structure and relationship distances; recording hidden from the search; removal events; truth as default reference |
| `test_ea.py` | Contract tests for every implemented representation; Pareto selection; problem; baselines; logger files |

## Data and outputs (outside `src/`)

| Path | Purpose |
|---|---|
| `test_cases/<name>/` | One clean test case: `case.json` (description, settings, answer-key stats), `truth.tsv`, `ibd.csv` (search input), `solutions/*.tsv` (answer key) |
| `logs/` | Run progress logs (`*.jsonl`), git-ignored, created on first write |
| `results/` | Run results (`*.json`) and batch summaries; `results/db/` run-record databases (grid shards are merged and removed); git-ignored |
| `references/` | Reference pedigrees for real data (e.g. `task3_siblings.tsv`), shown and tracked by the recorder |
