"""Portable bundles work independently of the full search package."""

import importlib.util
from pathlib import Path
import sqlite3
import subprocess
import sys
from zipfile import ZipFile

import pytest

from pedigree_ea.data.cases import load_case
from pedigree_ea.ea import RunLogger, Stopping, baseline_config, run_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def packer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("pack_viewer", ROOT / "scripts/pack_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (tmp_path / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_database_bundle_runs_without_search_dependencies(packer, tmp_path):
    case = load_case(ROOT / "test_cases/parent_child")
    config = baseline_config("direct", max_latent=0, tol=1e-6,
                             stopping=Stopping(max_evals=200))
    logger = RunLogger("test", tmp_path / "logs", tmp_path / "results")
    run_case(config, case, logger)
    folder, archive = packer.pack_viewer(logger.db_file, "viewer")
    # Exercise the extracted ZIP, rather than relying on the original folder.
    extracted = tmp_path / "extracted"
    with ZipFile(archive) as z:
        assert not any("__pycache__" in name for name in z.namelist())
        z.extractall(extracted)
    bundle = extracted / folder.name
    code = """
import sys
sys.path.insert(0, 'src')
class RejectSearchDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'deap', 'matplotlib'}:
            raise AssertionError('unnecessary dependency: ' + fullname)
sys.meta_path.insert(0, RejectSearchDependencies())
from pedigree_ea.webviz.server import App
app = App('run.sqlite')
runs = app.handle('/api/runs', {})['runs']
rid = runs[0]['id']
view = app.view(rid)
assert view.series([])
frame = app.handle(f'/api/run/{rid}/frame', {})
assert frame
for cid in view.members_at(view.n_generations):
    assert view.per_pair(cid)
    view.card(cid, [])
assert 'pedigree_ea.ea.engine' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=bundle,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with sqlite3.connect(folder / "run.sqlite") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_html_bundle_and_collision(packer, tmp_path):
    source = tmp_path / "page.html"
    source.write_text("<!doctype html><title>Viewer</title>")
    folder, archive = packer.pack_viewer(source, "page")
    assert (folder / "index.html").read_bytes() == source.read_bytes()
    with ZipFile(archive) as z:
        assert set(z.namelist()) == {"page/index.html", "page/README.txt"}
    with pytest.raises(FileExistsError):
        packer.pack_viewer(source, "page")
    with pytest.raises(ValueError):
        packer.pack_viewer(source, "../escape")
