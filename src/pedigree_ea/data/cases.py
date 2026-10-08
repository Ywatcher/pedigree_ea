"""Test cases on disk: a truth pedigree, the observed IBD, and an answer key.

Layout of one case directory:
    case.json         name, description, how the data was made, answer-key settings
    truth.tsv         the generating pedigree, including latent people
    ibd.csv           pairwise IBD0/1/2 among observed people (the search input)
    solutions/*.tsv   every pedigree that fits `ibd.csv` within the answer-key
                      bounds (from brute_force.enumerate_pedigrees), if computed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import io
from ..genetics.ibd import IBDData
from ..genetics.king import KingData
from ..genetics.kinship import KinshipData
from ..genetics.pedigree import Pedigree


@dataclass
class Case:
    name: str
    description: str
    truth: Pedigree | None      # None for real data
    target: IBDData | KinshipData | KingData
    info: dict[str, Any] = field(default_factory=dict)  # source, noise, answer_key, ...
    solutions: list[Pedigree] | None = None


def save_case(case: Case, root: str | Path) -> Path:
    d = Path(root) / case.name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": case.name, "description": case.description,
            "observed": case.target.ids, **case.info}
    (d / "case.json").write_text(json.dumps(meta, indent=2) + "\n")
    io.write_pedigree(case.truth, d / "truth.tsv")
    io.write_ibd(case.target, d / "ibd.csv")
    sol_dir = d / "solutions"
    if sol_dir.exists():
        for f in sol_dir.glob("*.tsv"):
            f.unlink()
    if case.solutions is not None:
        sol_dir.mkdir(exist_ok=True)
        for k, ped in enumerate(case.solutions, 1):
            io.write_pedigree(ped, sol_dir / f"s{k:04d}.tsv")
    return d


def load_case(path: str | Path) -> Case:
    d = Path(path)
    meta = json.loads((d / "case.json").read_text())
    target = io.read_ibd(d / "ibd.csv")
    target = IBDData(dict(target.items()), meta.pop("observed"))
    sol_dir = d / "solutions"
    solutions = ([io.read_pedigree(f) for f in sorted(sol_dir.glob("*.tsv"))]
                 if sol_dir.exists() else None)
    return Case(meta.pop("name"), meta.pop("description"), io.read_pedigree(d / "truth.tsv"),
                target, meta, solutions)


def list_cases(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/case.json"))
