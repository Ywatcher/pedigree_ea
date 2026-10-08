"""Converting targets, configs, members and code versions to stored values."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..genetics.ibd import IBDData
from ..genetics.king import KingData
from ..genetics.kinship import KinshipData

# Fields that do not change the search (only what is recorded) are left out of
# the config digest, so the same search gets the same config id.
_NOT_SEARCH = ("seed", "snapshot", "record_closest", "record_rank_by")


def digest(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ---- targets ------------------------------------------------------------------

def target_payload(target: IBDData | KinshipData | KingData) -> tuple[str, dict]:
    if isinstance(target, KingData):
        return "king", {"ids": target.ids, "unrelated_ibs0": target.unrelated_ibs0,
                        "pairs": [[a, b, v.kinship, v.ibs0] for (a, b), v in target.items()]}
    if isinstance(target, KinshipData):
        return "kinship", {"ids": target.ids, "pairs": [[a, b, v] for (a, b), v in target.items()]}
    return "ibd", {"ids": target.ids, "pairs": [[a, b, *v] for (a, b), v in target.items()]}


def target_from_payload(kind: str, payload: dict):
    ids = payload["ids"]
    if kind == "king":
        return KingData({(a, b): (k, i) for a, b, k, i in payload["pairs"]}, ids,
                        unrelated_ibs0=payload["unrelated_ibs0"])
    if kind == "kinship":
        return KinshipData({(a, b): v for a, b, v in payload["pairs"]}, ids)
    return IBDData({(a, b): (k0, k1, k2) for a, b, k0, k1, k2 in payload["pairs"]}, ids)


# ---- configs ------------------------------------------------------------------

def search_config(config: dict) -> dict:
    """The config without fields that only affect recording."""
    return {k: v for k, v in config.items() if k not in _NOT_SEARCH}


def config_label(config: dict) -> str:
    return f"{config['representation']}-{config['strategy']}"


def config_digest(config: dict) -> str:
    return digest(search_config(config))


# ---- population membership ----------------------------------------------------

def pack_members(members: dict[int, int]) -> bytes:
    """{candidate id: count} -> int32 pairs."""
    return np.array(sorted(members.items()), dtype=np.int32).tobytes()


def unpack_members(blob: bytes) -> dict[int, int]:
    arr = np.frombuffer(blob, dtype=np.int32).reshape(-1, 2)
    return {int(c): int(n) for c, n in arr}


def pack_floats(values) -> bytes:
    return np.asarray(values, dtype=np.float64).tobytes()


def unpack_floats(blob: bytes | None) -> list[float] | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float64).tolist()


# ---- code version -------------------------------------------------------------

@lru_cache(maxsize=1)
def code_version() -> tuple[str | None, int | None]:
    """(commit, dirty) of the repository holding this package, or (None, None)."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, (1 if status else 0) if commit else None
