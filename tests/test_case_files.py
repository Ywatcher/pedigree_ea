"""Consistency checks for the saved cases in test_cases/ (see scripts/make_test_cases.py)."""

from pathlib import Path

import numpy as np
import pytest

from pedigree_ea import PedigreeSet, expected_ibd
from pedigree_ea.data.cases import list_cases, load_case

CASES = list_cases(Path(__file__).resolve().parents[1] / "test_cases")


def test_cases_exist():
    assert CASES, "run scripts/make_test_cases.py"


@pytest.mark.parametrize("path", CASES, ids=[p.name for p in CASES])
def test_case_is_consistent(path):
    case = load_case(path)
    case.truth.validate()
    truth_ibd = expected_ibd(case.truth, case.target.ids)
    for pair, v in case.target.items():
        assert np.asarray(truth_ibd[pair]) == pytest.approx(v, abs=1e-12)

    key = case.info.get("answer_key")
    if key is None:
        return
    assert len(case.solutions) == key["n_solutions"]
    found = PedigreeSet()
    for sol in case.solutions:
        sol.validate()
        assert len(sol.latent_ids) <= key["max_latent"]
        got = expected_ibd(sol, case.target.ids)
        for pair, v in case.target.items():
            assert np.asarray(got[pair]) == pytest.approx(v, abs=key["tol"])
        assert found.add(sol), "duplicate solution"
    assert (case.truth in found) == key["truth_included"]
