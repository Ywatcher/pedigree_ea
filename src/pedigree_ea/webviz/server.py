"""Local web server for exploring run-record databases (see store/).

    python scripts/visualize.py results/db/grid_x_<stamp>.sqlite [--reference PATH:LABEL] [--port 8765]

API (JSON):
    GET /api/runs                                  runs in the database
    GET /api/run/<id>                              run details, references, objectives
    GET /api/run/<id>/series?refs=a,b              per-generation curves and events
    GET /api/run/<id>/frame?gen=G&refs=a,b&k=K     sections for one generation
    GET /api/run/<id>/candidate/<cid>?refs=a,b     one candidate with per-pair values
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlparse

from .. import viz
from ..ea.experiments.serialize import to_jsonable
from ..genetics.pedigree import Pedigree
from ..store.reader import Database

INDEX = Path(__file__).with_name("index.html")


class App:
    def __init__(self, db_path: str | Path, references: Mapping[str, Pedigree] | None = None):
        self.db = Database(db_path)
        self.extra = dict(references or {})

    def view(self, run_id: int):
        ids = None
        row = self.db.conn.execute("SELECT p.ids FROM runs r JOIN people_sets p ON p.id = r.people_set_id"
                                   " WHERE r.id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id}")
        ids = set(json.loads(row[0]))
        extra = {label: ped for label, ped in self.extra.items() if set(ped.observed_ids) == ids}
        return self.db.run(run_id, extra)

    def handle(self, path: str, query: dict[str, list[str]]):
        parts = [p for p in path.split("/") if p]
        refs = [r for r in query.get("refs", [""])[0].split(",") if r]
        if parts == ["api", "runs"]:
            return {"db": str(self.db.path), "runs": self.db.runs()}
        if len(parts) >= 3 and parts[:2] == ["api", "run"]:
            v = self.view(int(parts[2]))
            if len(parts) == 3:
                return {"id": v.run_id, "name": v.name, "case": v.case, "grid": v.grid, "label": v.label,
                        "seed": v.seed, "config": v.config, "kind": v.kind, "ids": v.ids,
                        "objectives": v.objective_names, "generations": v.n_generations,
                        "summary": v.summary, "score": v.score, "colors": viz.colors_for(v.ids),
                        "references": [{"label": r, "layout": viz.layout(p, v.ids)}
                                       for r, p in v.refs.items()]}
            if parts[3] == "series":
                return v.series(refs)
            if parts[3] == "frame":
                return v.frame(int(query.get("gen", ["1"])[0]), refs, int(query.get("k", ["1"])[0]))
            if parts[3] == "candidate" and len(parts) == 5:
                cid = int(parts[4])
                return {**v.card(cid, [r for r in refs if r in v.refs]), "per_pair": v.per_pair(cid),
                        "kind": v.kind}
        raise KeyError(path)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            try:
                data = app.handle(url.path, parse_qs(url.query))
                body = json.dumps(to_jsonable(data)).encode()
                self._send(200, body, "application/json")
            except KeyError as e:
                self._send(404, json.dumps({"error": str(e)}).encode(), "application/json")
            except Exception as e:  # report errors to the page instead of dropping the connection
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(),
                           "application/json")

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass
    return Handler


def serve(db_path: str | Path, references: Mapping[str, Pedigree] | None = None,
          host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    app = App(db_path, references)
    server = HTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{server.server_port}/"
    print(f"serving {db_path} at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
