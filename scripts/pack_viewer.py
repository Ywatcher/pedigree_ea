"""Pack a database explorer or a standalone HTML page under viewer_bundles/."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sqlite3
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]

# Only modules required by the database viewer. Lightweight package initializers
# avoid importing the search engine, experiments, and representation registries.
MODULES = (
    "viz.py", "data/io.py", "ea/pareto.py", "ea/problem.py",
    "ea/experiments/serialize.py", "store/codec.py", "store/schema.py",
    "store/reader.py", "webviz/server.py", "webviz/index.html", "webviz/help.js",
    "genetics/pedigree.py", "genetics/pairs.py", "genetics/kinship.py",
    "genetics/ibd.py", "genetics/king.py", "genetics/canonical.py",
    "genetics/similarity.py", "genetics/batch.py",
)


def pack_viewer(source: Path, name: str | None = None, make_zip: bool = True) -> tuple[Path, Path | None]:
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in {".sqlite", ".html"}:
        raise ValueError("input must be a run-record .sqlite database or an exported .html page")
    name = name or f"{source.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if name in {"", ".", ".."} or any(c in name for c in "/\\"):
        raise ValueError("name must be a single folder name")
    output = ROOT / "viewer_bundles" / name
    zip_path = output.with_name(output.name + ".zip") if make_zip else None
    if output.exists() or (zip_path is not None and zip_path.exists()):
        raise FileExistsError(f"bundle already exists: {output}")
    output.mkdir(parents=True)

    if source.suffix.lower() == ".html":
        shutil.copyfile(source, output / "index.html")
        instructions = "Open index.html in a modern browser. No Python or installation is required.\n"
    else:
        # SQLite backup includes committed WAL contents and produces a standalone
        # consistent copy even if the source has SQLite sidecar files.
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
            if original.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() is None:
                raise ValueError("not a pedigree run-record database")
            with sqlite3.connect(output / "run.sqlite") as destination:
                original.backup(destination)
        pkg = output / "src" / "pedigree_ea"
        for module in MODULES:
            dest = pkg / module
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "src" / "pedigree_ea" / module, dest)
        for folder in ("", "data", "genetics", "ea", "ea/experiments", "store", "webviz"):
            (pkg / folder / "__init__.py").write_text('"""Portable viewer package."""\n')
        (pkg / "__init__.py").write_text("from .data import io\n")
        (pkg / "webviz" / "__init__.py").write_text("from .server import serve\n")
        (output / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts" / "visualize.py", output / "scripts" / "visualize.py")
        (output / "requirements.txt").write_text("numpy\nnetworkx>=3.5\n")
        instructions = (
            "Requires Python 3.12+ and a modern browser.\n\n"
            "Open a terminal in this folder and run:\n"
            "  python -m pip install -r requirements.txt\n"
            "  python scripts/visualize.py run.sqlite\n\n"
            "Visit http://127.0.0.1:8765 and keep the terminal running.\n"
            "Use --port 8766 if the default port is occupied.\n"
            "No DEAP, Matplotlib, Jupyter, or package installation is needed.\n"
            "The database contains all runs and sample IDs from the selected input.\n"
        )
    (output / "README.txt").write_text(instructions)
    if zip_path is not None:
        with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output.parent))
    return output, zip_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="run-record .sqlite database or self-contained .html page")
    ap.add_argument("--name", help="bundle folder name (default: input stem plus timestamp)")
    ap.add_argument("--no-zip", action="store_true", help="generate the folder only")
    args = ap.parse_args()
    try:
        folder, archive = pack_viewer(args.input, args.name, not args.no_zip)
    except (OSError, ValueError, sqlite3.Error) as error:
        ap.exit(1, f"Cannot create viewer bundle: {error}\n")
    print(f"Folder: {folder}")
    if archive:
        print(f"ZIP: {archive} ({archive.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
