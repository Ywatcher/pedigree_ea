"""Plain-text formats for pedigrees, pairwise IBD and kinship.

IBD input (comma-separated); ibd0 may be left empty to mean 1 - ibd1 - ibd2:
    id1,id2,ibd0,ibd1,ibd2
    A,B,0.25,0.5,0.25
Output of IBD tools can be read directly (`read_plink_genome`, `read_king_seg`).

KING-robust kinship tables (PLINK2 `--make-king-table` .kin0, KING .kin0/.kin):
`read_king` gives kinship + IBS0 (KingData), `read_king_kinship` kinship only,
`read_king_ibd` approximate IBD0/1/2 estimated from the IBS0 column.
`read_psam` reads PLINK2 sample ids and sex.

Pedigree (tab-separated, PLINK-like):
    id  parent1  parent2  sex  observed
    A   0        0        M    1
    B   A        L1       F    1
    L1  0        0        U    0
"0" means no parent; sex is M, F or U (unknown); observed is 1 or 0.

Kinship (comma-separated):
    id1,id2,kinship
    A,B,0.25
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..genetics.ibd import IBD, IBDData
from ..genetics.king import KingData
from ..genetics.kinship import KinshipData
from ..genetics.pedigree import Pedigree, PedigreeError

MISSING = "0"
PEDIGREE_COLUMNS = ["id", "parent1", "parent2", "sex", "observed"]
KINSHIP_COLUMNS = ["id1", "id2", "kinship"]
IBD_COLUMNS = ["id1", "id2", "ibd0", "ibd1", "ibd2"]


def write_ibd(data: IBDData, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(IBD_COLUMNS)
        for (a, b), v in data.items():
            w.writerow([a, b, *map(repr, v)])


def read_ibd(path: str | Path) -> IBDData:
    values = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            k1, k2 = float(row["ibd1"]), float(row["ibd2"])
            k0 = float(row["ibd0"]) if row.get("ibd0", "").strip() else 1.0 - k1 - k2
            values[(row["id1"], row["id2"])] = IBD(k0, k1, k2)
    return IBDData(values)


def _read_whitespace_table(path: str | Path) -> list[dict[str, str]]:
    with open(path) as f:
        header = f.readline().split()
        return [dict(zip(header, line.split())) for line in f if line.strip()]


def read_plink_genome(path: str | Path) -> IBDData:
    """PLINK `--genome` output (columns IID1, IID2, Z0, Z1, Z2). Family ids are ignored."""
    return IBDData({(r["IID1"], r["IID2"]): (r["Z0"], r["Z1"], r["Z2"])
                    for r in _read_whitespace_table(path)})


def read_king_seg(path: str | Path) -> IBDData:
    """KING `--ibdseg` output (columns ID1, ID2, IBD1Seg, IBD2Seg). Family ids are ignored."""
    return IBDData({(r["ID1"], r["ID2"]): IBD.from_k1_k2(float(r["IBD1Seg"]), float(r["IBD2Seg"]))
                    for r in _read_whitespace_table(path)})


# Column aliases in KING / PLINK2 kinship tables (lower-case, without '#').
_KING_COLUMNS = {
    "id1": "id1", "iid1": "id1", "id2": "id2", "iid2": "id2",
    "kinship": "kinship", "ibs0": "ibs0", "hethet": "hethet",
    "nsnp": "nsnp", "n_snp": "nsnp", "obs_ct": "nsnp",
}
def read_king_table(path: str | Path) -> list[dict]:
    """Rows of a KING-robust table with keys id1, id2, kinship and, when present,
    ibs0, hethet, nsnp (numbers as float). Family ids are ignored."""
    with open(path) as f:
        header = [h.lstrip("#").lower() for h in f.readline().split()]
        names = [_KING_COLUMNS.get(h) for h in header]
        if not {"id1", "id2", "kinship"} <= set(names):
            raise ValueError(f"{path}: not a KING kinship table (columns: {header})")
        rows = []
        for line in f:
            if not line.strip():
                continue
            row = {}
            for name, value in zip(names, line.split()):
                if name is not None:
                    row[name] = value if name in ("id1", "id2") else float(value)
            rows.append(row)
    return rows


def read_king_kinship(path: str | Path, clip_negative: bool = False) -> KinshipData:
    """Kinship from a KING-robust table. KING gives negative values for
    unrelated pairs; `clip_negative` sets them to 0 (no pedigree predicts < 0)."""
    rows = read_king_table(path)
    return KinshipData({(r["id1"], r["id2"]): max(r["kinship"], 0.0) if clip_negative
                        else r["kinship"] for r in rows})


def read_king(path: str | Path, unrelated_ibs0: float | None = None) -> KingData:
    """Kinship and IBS0 from a KING-robust table (needs an IBS0 column).

    `unrelated_ibs0` is the IBS0 expected for unrelated pairs; by default the
    median IBS0 of pairs with kinship below KING's unrelated cut-off (0.0442).
    """
    rows = read_king_table(path)
    if any("ibs0" not in r for r in rows):
        raise ValueError(f"{path}: no IBS0 column; use read_king_kinship")
    return KingData({(r["id1"], r["id2"]): (r["kinship"], r["ibs0"]) for r in rows},
                    unrelated_ibs0=unrelated_ibs0)


def read_king_ibd(path: str | Path, unrelated_ibs0: float | None = None) -> tuple[IBDData, float]:
    """Approximate IBD0/1/2 from a KING-robust table's kinship and IBS0 columns.
    Returns (data, IBS0 baseline used). See ibd.ibd_from_kinship_ibs0 for caveats."""
    data = read_king(path, unrelated_ibs0)
    return data.approx_ibd(), data.unrelated_ibs0


def read_pairwise(path: str | Path, target: str = "king", unrelated_ibs0: float | None = None
                  ) -> tuple[IBDData | KinshipData | KingData, str, str]:
    """Read pairwise data by file type. Returns (data, kind, description).

    .kin0/.kin   KING-robust table; `target` = "king" (kinship + IBS0, default),
                 "ibd" (approximate IBD0/1/2) or "kinship" (negatives set to 0)
    .genome      PLINK --genome (IBD)        .seg   KING --ibdseg (IBD)
    .csv         IBD CSV (id1,id2,ibd0,ibd1,ibd2)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".kin0", ".kin"):
        if target == "kinship":
            return read_king_kinship(path, clip_negative=True), "kinship", \
                "KING kinship, negative values set to 0"
        data = read_king(path, unrelated_ibs0)
        base = f"unrelated IBS0 baseline {data.unrelated_ibs0:.4f}"
        if target == "ibd":
            return data.approx_ibd(), "ibd", f"approximate IBD from KING kinship + IBS0 ({base})"
        if target != "king":
            raise ValueError(f"target must be 'king', 'ibd' or 'kinship', got {target!r}")
        return data, "king", f"KING kinship + IBS0 -> IBD0 ({base})"
    if suffix == ".genome":
        return read_plink_genome(path), "ibd", "PLINK --genome"
    if suffix == ".seg":
        return read_king_seg(path), "ibd", "KING --ibdseg"
    if suffix == ".csv":
        return read_ibd(path), "ibd", "IBD CSV"
    raise ValueError(f"unknown pairwise file format: {path}")


def read_psam(path: str | Path) -> dict[str, str | None]:
    """PLINK2 .psam: sample id -> sex ('M', 'F' or None). Family ids are ignored."""
    sex_codes = {"1": "M", "2": "F", "m": "M", "f": "F", "male": "M", "female": "F"}
    with open(path) as f:
        header = [h.lstrip("#").upper() for h in f.readline().split()]
        i_id = header.index("IID")
        i_sex = header.index("SEX") if "SEX" in header else None
        out = {}
        for line in f:
            parts = line.split()
            if parts:
                sex = parts[i_sex].lower() if i_sex is not None else ""
                out[parts[i_id]] = sex_codes.get(sex)
    return out


def write_pedigree(ped: Pedigree, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(PEDIGREE_COLUMNS)
        for pid in ped:
            p = ped[pid]
            if pid == MISSING:
                raise PedigreeError(f"id {MISSING!r} is reserved for a missing parent")
            parents = list(p.parents) + [MISSING] * (2 - len(p.parents))
            w.writerow([pid, *parents, p.sex or "U", int(p.observed)])


def read_pedigree(path: str | Path) -> Pedigree:
    ped = Pedigree()
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            parents = [x for x in (row["parent1"], row["parent2"]) if x != MISSING]
            sex = None if row["sex"] == "U" else row["sex"]
            ped.add(row["id"], parents, sex, row["observed"] == "1")
    return ped


def write_kinship(data: KinshipData, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KINSHIP_COLUMNS)
        for (a, b), v in data.items():
            w.writerow([a, b, repr(v)])


def read_kinship(path: str | Path) -> KinshipData:
    with open(path, newline="") as f:
        values = {(row["id1"], row["id2"]): float(row["kinship"]) for row in csv.DictReader(f)}
    return KinshipData(values)
