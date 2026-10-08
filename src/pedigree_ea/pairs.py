"""Values attached to unordered pairs of people."""

from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, Mapping, TypeVar

Pair = tuple[str, str]
V = TypeVar("V")


def pair_key(a: str, b: str) -> Pair:
    if a == b:
        raise ValueError(f"a pair needs two different people, got {a!r} twice")
    return (a, b) if a < b else (b, a)


class PairData(Generic[V]):
    """Values for unordered pairs of people, e.g. a search target or a prediction.

    Pairs without a value are allowed (unknown/unused). Subclasses convert
    input values with `_convert`.
    """

    def __init__(self, values: Mapping[tuple[str, str], Any], ids: Iterable[str] | None = None):
        self._values: dict[Pair, V] = {pair_key(a, b): self._convert(v)
                                       for (a, b), v in values.items()}
        seen = dict.fromkeys(ids) if ids is not None else {}
        for a, b in self._values:
            seen.setdefault(a)
            seen.setdefault(b)
        self.ids: list[str] = list(seen)

    @staticmethod
    def _convert(v: Any) -> V:
        return v

    def __getitem__(self, pair: tuple[str, str]) -> V:
        return self._values[pair_key(*pair)]

    def get(self, a: str, b: str, default: V | None = None) -> V | None:
        return self._values.get(pair_key(a, b), default)

    def __contains__(self, pair: object) -> bool:
        return isinstance(pair, tuple) and pair_key(*pair) in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Pair]:
        return iter(self._values)

    def items(self):
        return self._values.items()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.ids)} people, {len(self)} pairs)"
