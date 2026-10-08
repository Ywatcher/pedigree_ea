"""Cyclic genotype with a DAG decoder. NOT IMPLEMENTED: design still open.

Idea (ideas.md): the genotype is a graph that may contain cycles; a decoder
turns it into a valid pedigree (DAG), and how cyclic the genotype is can be
an optimization target or a constraint.

Open questions to settle before implementing:
- Genotype: a weighted adjacency matrix W (N x N)? Binary edges + weights?
- Decoder: "find a subtree/subgraph with the largest edge weights" -- e.g.
  greedily add edges by decreasing weight, skipping those that close a cycle
  or exceed two parents (an approximation; the exact problem, maximum-weight
  acyclic subgraph / feedback arc set, is NP-hard).
- Edge weights: evolved freely, or derived from the IBD target (e.g. pairs
  with high IBD get stronger candidate edges)?
- Cycle measure: number of edges the decoder drops, their total weight, or the
  minimum number of edges to remove (feedback arc set size)?
- Use of the measure: extra objective, or a threshold that discards genotypes.
- Mutation strength: per-genotype self-adaptive sigma for weight mutation?

Expected shape once settled: a `Representation` subclass whose `decode` uses
`dag.Builder` (which already counts edges rejected because of cycles) and whose
`genotype_objectives` returns the cycle measure.
"""

from __future__ import annotations

from .base import Representation


class Cyclic(Representation):
    name = "cyclic"

    def __init__(self, n_obs: int, max_latent: int, **params):
        raise NotImplementedError("cyclic representation: design still open, see module docstring")
