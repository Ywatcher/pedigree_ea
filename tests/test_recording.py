import pytest

from pedigree_ea import Pedigree, batch, expected_ibd, synth
from pedigree_ea.data.cases import load_case
from pedigree_ea.ea import Problem, Stopping, baseline_config, run, run_case
from pedigree_ea.ea.recording import Recording
from pedigree_ea.genetics.similarity import relationship_distance, structure_distance

from test_grid import TINY


def test_structure_distance_basics():
    half, grand = synth.half_sibs(), synth.grandparent()
    assert structure_distance(half, half.relabel({"L1": "Z"})) == 0      # latent names ignored
    pc = synth.parent_child()
    assert structure_distance(pc, pc.relabel({"A": "B", "B": "A"})) == 2  # reverse one edge
    assert structure_distance(half, grand) == structure_distance(grand, half) > 0


def test_structure_distance_greedy_is_upper_bound():
    a, b = synth.avuncular(), synth.first_cousins()     # 3 and 4 latent people
    exact = structure_distance(a, b, exact_limit=4)
    greedy = structure_distance(a, b, exact_limit=0)
    assert greedy >= exact


def test_relationship_distance():
    ids = ["A", "B"]
    assert relationship_distance(synth.half_sibs(), synth.grandparent(), ids) == 0
    assert relationship_distance(synth.half_sibs(), synth.full_sibs(), ids) == pytest.approx(0.25)


def test_recording_finds_reference_and_stays_hidden_from_search():
    truth = synth.full_sibs()
    target = expected_ibd(truth)
    cfg = baseline_config("direct", max_latent=2, tol=1e-6, snapshot=True,
                          stopping=Stopping(max_evals=2000))
    with_refs = run(cfg, target, references={"truth": truth})
    without = run(cfg, target)
    # Same seed, same search: references change only what is recorded.
    assert [h["best"] for h in with_refs.history] == [h["best"] for h in without.history]
    last = with_refs.snapshots[-1]["records"]["closest:truth"][0]
    assert last["distances"]["truth"]["structure"] == 0 and last["fits"]


def test_removal_event_when_closest_leaves():
    truth = synth.half_sibs()
    prob = Problem(expected_ibd(truth), max_latent=1, tol=1e-6)
    rec = Recording(prob, {"truth": truth})
    exact = batch.from_pedigree(truth, prob.ids, 1)
    empty = exact.copy()
    empty[:] = -1
    objs = prob.evaluate(exact[None]).objectives
    out1 = rec.record(objs, exact[None], generation=1)
    assert out1["events"] == [] and out1["last_removed"]["truth"] is None
    out2 = rec.record(prob.evaluate(empty[None]).objectives, empty[None], generation=2)
    (ev,) = out2["events"]
    assert ev["generation"] == 2 and ev["distance_before"] == 0 and ev["distance_after"] > 0
    assert out2["last_removed"]["truth"] is ev


def test_run_case_uses_truth_as_reference(tmp_path):
    case = load_case(TINY["case_dir"] + "/half_sibs")
    cfg = baseline_config("direct", max_latent=1, tol=1e-6, snapshot=True,
                          stopping=Stopping(max_evals=300))
    res, _ = run_case(cfg, case)
    assert "closest:truth" in res.snapshots[-1]["records"]
