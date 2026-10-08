"""SQLite schema for run records.

Shared tables (deduplicated by content key; ids are local to a file):
    people_sets     observed ids (their order defines pair order)
    candidates      each distinct pedigree once: canonical bytes (genetics/canonical.py)
    targets         the input data (kind + JSON payload), by digest
    configs         run configuration without the seed, by digest
    code_versions   git commit + dirty flag
    refs            reference pedigrees (label -> candidate)

Per-run tables:
    runs            one row per run
    run_refs        references given to a run
    generations     per-generation statistics
    membership      population changes per generation (candidate, +n / -n)
    keyframes       full population every `keyframe_every` generations
    appearances     first appearance of a candidate in a run: evaluation,
                    generation, operator, parent candidates, objectives at entry,
                    whether it entered the population, when it became a fit

Everything else (objectives under any setting, per-pair values, best,
Pareto front, closest to a reference, removal events) is derived on read.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

TABLES = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS people_sets (id INTEGER PRIMARY KEY, ids TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY, people_set_id INTEGER NOT NULL, canon BLOB NOT NULL,
    n_latent INTEGER NOT NULL, valid INTEGER NOT NULL,
    UNIQUE (people_set_id, canon));
CREATE TABLE IF NOT EXISTS targets (
    id INTEGER PRIMARY KEY, people_set_id INTEGER NOT NULL, kind TEXT NOT NULL,
    payload TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, source TEXT);
CREATE TABLE IF NOT EXISTS configs (
    id INTEGER PRIMARY KEY, digest TEXT NOT NULL UNIQUE, label TEXT NOT NULL, config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS code_versions (
    id INTEGER PRIMARY KEY, commit_id TEXT, dirty INTEGER, UNIQUE (commit_id, dirty));
CREATE TABLE IF NOT EXISTS refs (
    id INTEGER PRIMARY KEY, people_set_id INTEGER NOT NULL, label TEXT NOT NULL,
    candidate_id INTEGER NOT NULL, UNIQUE (people_set_id, label, candidate_id));
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, name TEXT, grid TEXT, stamp TEXT, case_name TEXT,
    people_set_id INTEGER, target_id INTEGER, config_id INTEGER, seed INTEGER,
    code_version_id INTEGER, objectives TEXT, keyframe_every INTEGER, scope TEXT,
    started REAL, finished REAL, summary TEXT, score TEXT);
CREATE TABLE IF NOT EXISTS run_refs (run_id INTEGER, ref_id INTEGER, PRIMARY KEY (run_id, ref_id));
CREATE TABLE IF NOT EXISTS generations (
    run_id INTEGER, gen INTEGER, evals INTEGER, seconds REAL, population INTEGER,
    n_phenotypes INTEGER, n_fits INTEGER, n_pareto INTEGER, PRIMARY KEY (run_id, gen));
CREATE TABLE IF NOT EXISTS membership (run_id INTEGER, gen INTEGER, candidate_id INTEGER, delta INTEGER);
CREATE INDEX IF NOT EXISTS membership_run_gen ON membership (run_id, gen);
CREATE TABLE IF NOT EXISTS keyframes (run_id INTEGER, gen INTEGER, members BLOB, PRIMARY KEY (run_id, gen));
CREATE TABLE IF NOT EXISTS appearances (
    run_id INTEGER, candidate_id INTEGER, first_eval INTEGER, first_gen INTEGER,
    operator TEXT, parents TEXT, entry_objectives BLOB, in_population INTEGER,
    fit_at_eval INTEGER, PRIMARY KEY (run_id, candidate_id));
"""


def connect(path, create: bool = True) -> sqlite3.Connection:
    """Open a database; with `create`, make the tables. Refuses other schema versions."""
    conn = sqlite3.connect(str(path))
    if create:
        conn.executescript(TABLES)
        conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        conn.commit()
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None or int(row[0]) != SCHEMA_VERSION:
        conn.close()
        raise ValueError(f"{path}: schema version {row and row[0]}, expected {SCHEMA_VERSION}")
    return conn
