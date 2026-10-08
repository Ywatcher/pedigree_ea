"""Run records in SQLite: each distinct pedigree once, per-run population
changes and lineage; everything else derived on read.

    schema.py   tables, SCHEMA_VERSION, connect()
    codec.py    targets, configs, population blobs, code version
    writer.py   RunWriter: one run into a file (used by the Engine via RunLogger)
    merge.py    merge(): combine files (shards -> one database)
    reader.py   RunView: rebuild populations, objectives, best / Pareto / closest /
                removal events, per-pair values, snapshots for pictures
"""

from .merge import merge
from .schema import SCHEMA_VERSION, connect
from .writer import RunWriter

__all__ = ["SCHEMA_VERSION", "connect", "RunWriter", "merge"]
