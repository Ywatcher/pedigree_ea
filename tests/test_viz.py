import json
import re

from PIL import Image

from pedigree_ea import expected_ibd, synth, viz
from pedigree_ea.data.cases import Case
from pedigree_ea.ea import Stopping, baseline_config, run
from pedigree_ea.ea.experiments.animate import animate_methods, parse_method


def test_layout_puts_parents_above_children():
    ped = synth.sibs_with_children()
    lay = viz.layout(ped, ped.observed_ids)
    y = {n["id"]: n["y"] for n in lay["nodes"]}
    assert all(y[p] < y[c] for p, c in lay["edges"])
    assert {n["id"] for n in lay["nodes"]} == set(ped.ids)
    json.dumps(lay)   # plain JSON types only


def test_layout_marks_inbreeding():
    lay = viz.layout(synth.related_parents())
    assert {n["id"]: n["inbred"] for n in lay["nodes"]}["B"] is True


def test_layout_reports_cycle():
    p = synth.parent_child()
    p.set_parents("A", ["B"])
    assert viz.layout(p)["error"]


def test_parse_method():
    cfg = parse_method("neat:poss:excess_total+n_latent", max_latent=1)
    assert (cfg.representation, cfg.strategy, cfg.objectives) == ("neat", "poss", ("excess_total", "n_latent"))
    assert parse_method("direct").strategy == "nsga2"


def test_snapshots_recorded_only_when_asked():
    target = expected_ibd(synth.half_sibs())
    off = run(baseline_config("direct", max_latent=1, tol=1e-6, stopping=Stopping(max_evals=300)), target)
    on = run(baseline_config("direct", max_latent=1, tol=1e-6, snapshot=True,
                             stopping=Stopping(max_evals=300)), target)
    assert off.snapshots == [] and len(on.snapshots) == len(on.history)
    assert {"records", "events", "last_removed", "n_fits", "latest_fit"} <= set(on.snapshots[-1])
    assert list(on.snapshots[-1]["records"]) == ["best"]   # no references given


def test_animate_writes_gif_and_html(tmp_path):
    truth = synth.half_sibs()
    case = Case("half_sibs", "", truth, expected_ibd(truth), {}, None)
    out = animate_methods(case, ["direct", "direct:poss"], {"max_latent": 1, "tol": 1e-6},
                          max_evals=300, out_dir=tmp_path / "plt", results_dir=tmp_path / "results",
                          log_dir=tmp_path / "logs", stamp="S", verbose=False)
    assert out["db"][0].exists()      # rendered from the run database
    gifs = out["gif"]
    assert len(gifs) == 2 and all(p.exists() for p in gifs)
    assert Image.open(gifs[0]).n_frames > 1
    html = out["html"][0].read_text()
    data = json.loads(re.search(r"const DATA = (.*?);\n", html).group(1))
    assert [m["label"] for m in data["methods"]] == ["direct", "direct:poss"]
    assert data["references"][0]["label"] == "truth" and data["references"][0]["layout"]["nodes"]
    titles = [p["title"] for p in data["methods"][0]["frames"][0]["panels"]]
    assert titles == ["Best in population", "Closest to truth", "Removed closest to truth", "Latest fit"]
