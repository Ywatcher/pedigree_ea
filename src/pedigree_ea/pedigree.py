"""Pedigree data structure.

A pedigree is a set of people. Each person has 0-2 parents, an optional sex,
and is either *observed* (has pairwise data) or *latent* (an unsampled person
introduced to explain the data).

Conventions:
- Parents are stored unordered. Sex is not needed for autosomal kinship; it is
  only used to check that every couple can be one male and one female
  (see `Pedigree.sex_conflicts`).
- A missing parent is an unknown founder, unrelated to everyone else.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable, Iterator

MALE, FEMALE = "M", "F"
SEXES = (MALE, FEMALE)


class PedigreeError(ValueError):
    pass


@dataclass
class Person:
    id: str
    parents: tuple[str, ...] = ()
    sex: str | None = None
    observed: bool = True


class Pedigree:
    def __init__(self, people: Iterable[Person] = ()):
        self._people: dict[str, Person] = {}
        for p in people:
            self.add(p.id, p.parents, p.sex, p.observed)

    # ---- construction -------------------------------------------------

    def add(self, pid: str, parents: Iterable[str] = (), sex: str | None = None,
            observed: bool = True) -> Person:
        """Add a person. Parents may be added later; `problems()` checks references."""
        if pid in self._people:
            raise PedigreeError(f"duplicate id {pid!r}")
        if sex not in (None, MALE, FEMALE):
            raise PedigreeError(f"sex must be 'M', 'F' or None, got {sex!r}")
        person = Person(pid, tuple(parents), sex, observed)
        self._people[pid] = person
        return person

    def remove(self, pid: str) -> None:
        """Remove a person and drop them from their children's parents."""
        del self._people[pid]
        for p in self._people.values():
            if pid in p.parents:
                p.parents = tuple(x for x in p.parents if x != pid)

    def set_parents(self, pid: str, parents: Iterable[str]) -> None:
        self._people[pid].parents = tuple(parents)

    def copy(self) -> Pedigree:
        new = Pedigree()
        new._people = {pid: replace(p) for pid, p in self._people.items()}
        return new

    def relabel(self, mapping: dict[str, str]) -> Pedigree:
        """Return a copy with ids renamed; ids not in `mapping` are kept."""
        new = Pedigree()
        for p in self._people.values():
            new.add(mapping.get(p.id, p.id), (mapping.get(x, x) for x in p.parents),
                    p.sex, p.observed)
        return new

    # ---- access -------------------------------------------------------

    def __contains__(self, pid: object) -> bool:
        return pid in self._people

    def __len__(self) -> int:
        return len(self._people)

    def __iter__(self) -> Iterator[str]:
        return iter(self._people)

    def __getitem__(self, pid: str) -> Person:
        return self._people[pid]

    @property
    def ids(self) -> list[str]:
        return list(self._people)

    @property
    def observed_ids(self) -> list[str]:
        return [pid for pid, p in self._people.items() if p.observed]

    @property
    def latent_ids(self) -> list[str]:
        return [pid for pid, p in self._people.items() if not p.observed]

    def parents(self, pid: str) -> tuple[str, ...]:
        return self._people[pid].parents

    def children(self, pid: str) -> list[str]:
        return [c.id for c in self._people.values() if pid in c.parents]

    def children_map(self) -> dict[str, list[str]]:
        cm: dict[str, list[str]] = {pid: [] for pid in self._people}
        for c in self._people.values():
            for p in c.parents:
                if p in cm:
                    cm[p].append(c.id)
        return cm

    def founders(self) -> list[str]:
        return [pid for pid, p in self._people.items() if not p.parents]

    def mates(self) -> set[frozenset[str]]:
        """Couples, i.e. pairs of people who share a child."""
        return {frozenset(p.parents) for p in self._people.values() if len(set(p.parents)) == 2}

    def ancestors(self, pid: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self._people[pid].parents)
        while stack:
            a = stack.pop()
            if a not in seen and a in self._people:
                seen.add(a)
                stack.extend(self._people[a].parents)
        return seen

    def descendants(self, pid: str) -> set[str]:
        cm = self.children_map()
        seen: set[str] = set()
        stack = list(cm[pid])
        while stack:
            d = stack.pop()
            if d not in seen:
                seen.add(d)
                stack.extend(cm[d])
        return seen

    def topological_order(self) -> list[str]:
        """Parents before children. Raises PedigreeError on a cycle or unknown parent."""
        indeg = {}
        for pid, p in self._people.items():
            for x in p.parents:
                if x not in self._people:
                    raise PedigreeError(f"{pid!r} has unknown parent {x!r}")
            indeg[pid] = len(p.parents)
        cm = self.children_map()
        queue = deque(pid for pid, d in indeg.items() if d == 0)
        order = []
        while queue:
            pid = queue.popleft()
            order.append(pid)
            for c in cm[pid]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != len(self._people):
            raise PedigreeError("pedigree contains a cycle")
        return order

    # ---- validity -----------------------------------------------------

    def sex_conflicts(self) -> list[str]:
        """Check that couples can be assigned opposite sexes.

        The mate graph (edge = two people share a child) must be bipartite, and
        known sexes must agree with one side being male and the other female.
        """
        adj: dict[str, set[str]] = {}
        for couple in self.mates():
            a, b = tuple(couple)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        color: dict[str, int] = {}
        conflicts = []
        for start in adj:
            if start in color:
                continue
            color[start] = 0
            component = [start]
            queue = deque([start])
            while queue:
                x = queue.popleft()
                for y in adj[x]:
                    if y not in color:
                        color[y] = 1 - color[x]
                        component.append(y)
                        queue.append(y)
                    elif color[y] == color[x] and x < y:  # each edge is seen from both ends
                        conflicts.append(f"{x!r} and {y!r} are mates but must have the same sex")
            side_sex: dict[int, str] = {}
            for x in component:
                sex = self._people[x].sex if x in self._people else None
                if sex is None:
                    continue
                if side_sex.setdefault(color[x], sex) != sex:
                    conflicts.append(f"known sexes around {x!r} cannot alternate between mates")
            if len(side_sex) == 2 and side_sex[0] == side_sex[1]:
                conflicts.append(f"mates around {start!r} have the same known sex")
        return conflicts

    def problems(self, check_sex: bool = True) -> list[str]:
        """Return human-readable validity problems; empty means valid."""
        out = []
        for pid, p in self._people.items():
            if len(p.parents) > 2:
                out.append(f"{pid!r} has more than two parents")
            if len(set(p.parents)) != len(p.parents):
                out.append(f"{pid!r} lists the same parent twice")
            if pid in p.parents:
                out.append(f"{pid!r} is their own parent")
        try:
            self.topological_order()
        except PedigreeError as e:
            out.append(str(e))
        if check_sex:
            out.extend(self.sex_conflicts())
        return out

    def validate(self, check_sex: bool = True) -> None:
        problems = self.problems(check_sex)
        if problems:
            raise PedigreeError("; ".join(problems))

    def is_valid(self, check_sex: bool = True) -> bool:
        return not self.problems(check_sex)

    # ---- simplification -----------------------------------------------

    def pruned(self) -> Pedigree:
        """Remove latent people who cannot affect kinship among observed people.

        Repeatedly removes latent people who
        - have no observed descendant (they are nobody's relevant ancestor), or
        - are founders with exactly one child (same as an unknown parent).
        """
        ped = self.copy()
        changed = True
        while changed:
            changed = False
            cm = ped.children_map()
            for pid in ped.latent_ids:
                if not ped[pid].parents and len(cm[pid]) == 1:
                    ped.remove(pid)
                    changed = True
                    break
                if not any(ped[d].observed for d in ped.descendants(pid)):
                    ped.remove(pid)
                    changed = True
                    break
        return ped

    # ---- display ------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Pedigree({len(self)} people, {len(self.observed_ids)} observed, "
                f"{len(self.latent_ids)} latent)")

    def __str__(self) -> str:
        lines = []
        for pid, p in self._people.items():
            tag = "" if p.observed else " (latent)"
            sex = f" [{p.sex}]" if p.sex else ""
            parents = ", ".join(p.parents) if p.parents else "-"
            lines.append(f"{pid}{sex}{tag} <- {parents}")
        return "\n".join(lines)
