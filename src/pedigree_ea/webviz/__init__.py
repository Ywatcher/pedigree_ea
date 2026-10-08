"""Web visualizer for run-record databases: `serve(db_path)` or scripts/visualize.py."""

from .server import App, serve

__all__ = ["App", "serve"]
