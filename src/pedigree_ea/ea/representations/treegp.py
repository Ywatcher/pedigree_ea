"""Tree GP: a genotype is a program that builds the pedigree. NOT IMPLEMENTED:
the function set is still open.

Idea (ideas.md): each genotype is a tree describing how the pedigree is
generated, not the pedigree itself; crossover swaps subtrees. Two sketches:

    AddParents(AddSibling(ExtendAncestor(A)))

    Family(parents=Pair(X1, X2), children=[A, AddSibling(B)])

Open questions to settle before implementing:
- Function set and semantics: what does each node return (a person handle, a
  family, a partial pedigree), and how are subtrees combined?
- Sharing: a pedigree is a DAG (one person can be a parent in several
  families, and a latent person can appear in several places), but a tree
  cannot share subtrees. Allow references to observed people at several
  leaves? Named latent terminals? Or use a graph-based GP instead?
- Observed people: must each appear exactly once, at least once, or freely?
- Invalid programs (cycles, more than two parents): repair while decoding
  (skip the offending edge, as `dag.Builder` does) or reject?
- Bloat control: depth/size limits, or tree size as an extra objective.

Expected shape once settled: genotype as a prefix-encoded int array (numpy
friendly); `decode` interprets it into a `dag.Builder`; operators are subtree
mutation, point mutation and subtree crossover.
"""

from __future__ import annotations

from .base import Representation


class TreeGP(Representation):
    name = "treegp"
    has_crossover = True

    def __init__(self, n_obs: int, max_latent: int, **params):
        raise NotImplementedError("tree GP representation: function set still open, see module docstring")
