"""Evolutionary search for pedigrees compatible with pairwise relatedness.

Subpackages:
    genetics    pedigree model, exact kinship / IBD / KING measures, structure
                identity, batched numpy evaluation
    data        file formats (io), test cases on disk (cases), synthetic pedigrees (synth)
    reference   exact brute-force enumeration (answer keys)
    ea          evolutionary search: problem, archive, engine, pareto,
                representations/, strategies/, experiments/
    viz         drawing and animating pedigrees

The most used names are re-exported here, and the modules `batch`, `io`,
`cases` and `synth` can be imported directly: `from pedigree_ea import io, synth`.
"""

from .data import cases, io, synth
from .genetics import batch
from .genetics.canonical import PedigreeSet, same_structure, structure_hash
from .genetics.ibd import IBD, IBDData, expected_ibd, ibd_errors, jacquard
from .genetics.king import KingData, KingPair
from .genetics.kinship import (KinshipData, degree_class, expected_kinship, inbreeding,
                               kinship_matrix, kinship_to_degree, pair_errors)
from .genetics.pairs import PairData, pair_key
from .genetics.pedigree import FEMALE, MALE, Pedigree, PedigreeError, Person
from .reference.brute_force import enumerate_pedigrees

__all__ = [
    "batch", "io", "cases", "synth",
    "Pedigree", "Person", "PedigreeError", "MALE", "FEMALE",
    "PairData", "pair_key",
    "KingData", "KingPair",
    "IBD", "IBDData", "expected_ibd", "ibd_errors", "jacquard",
    "KinshipData", "kinship_matrix", "expected_kinship", "inbreeding", "pair_errors",
    "kinship_to_degree", "degree_class",
    "PedigreeSet", "structure_hash", "same_structure",
    "enumerate_pedigrees",
]
