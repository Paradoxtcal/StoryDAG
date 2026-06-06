"""Flask app: visual causal graph audit with inline editing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

from storydag.cnge.graph import CausalGraph

_GLOBAL_GRAPH_PATH: Path | None = None
_GLOBAL_GRAPH: CausalGraph | None = None


def _resolve_outputs() -> Path:
    from storydag.config import project_root
    return project_root() / "outputs"


def create_app(graph_path: str | Path | None = None) -> Flask:
    """Create a Flask app serving the causal graph auditor.

    Parameters
    ----------
    graph_path : optional path to a ``causal_graph.json``.  If omitted
        the app will list available graphs under ``outputs/``.
    """
    global _GLOBAL_GRAPH_PATH, _GLOBAL_GRAPH

    app = Flask(__name__)

    if graph_path:
        _GLOBAL_GRAPH_PATH = Path(graph_path).resolve()
        _GLOBAL_GRAPH = CausalGraph.load(_GLOBAL_GRAPH_PATH)

    # ── Pages ──────────────────────────────────────────────────────

    @app.route("/")
    def index():
        if _GLOBAL_GRAPH:
            graph_data = _GLOBAL_GRAPH.to_dict()
            return render_template("auditor.html", initial_graph=json.dumps(graph_data, ensure_ascii=False))
        outputs_root = _resolve_outputs()
        candidates = _list_graphs(outputs_root)
        return render_template("auditor.html", initial_graph="null", outputs=json.dumps(candidates, ensure_ascii=False))

    # ── REST API ───────────────────────────────────────────────────

    @app.route("/api/graph", methods=["GET"])
    def get_graph():
        path_str = request.args.get("path", "")
        if path_str:
            graph = CausalGraph.load(Path(path_str))
            return jsonify(graph.to_dict())
        if _GLOBAL_GRAPH:
            return jsonify(_GLOBAL_GRAPH.to_dict())
        return jsonify({"error": "no graph loaded"}), 404

    @app.route("/api/graph", methods=["POST"])
    def load_graph():
        data: Dict[str, Any] = request.get_json(force=True)
        path_str = data.get("path", "")
        if not path_str:
            return jsonify({"error": "path required"}), 400
        global _GLOBAL_GRAPH_PATH, _GLOBAL_GRAPH
        _GLOBAL_GRAPH_PATH = Path(path_str).resolve()
        _GLOBAL_GRAPH = CausalGraph.load(_GLOBAL_GRAPH_PATH)
        return jsonify(_GLOBAL_GRAPH.to_dict())

    @app.route("/api/graph/save", methods=["POST"])
    def save_graph():
        if _GLOBAL_GRAPH_PATH is None:
            return jsonify({"error": "no graph loaded; use POST /api/graph first"}), 400
        data: Dict[str, Any] = request.get_json(force=True)
        graph = CausalGraph.from_dict(data)
        graph.save(_GLOBAL_GRAPH_PATH)
        global _GLOBAL_GRAPH
        _GLOBAL_GRAPH = graph
        return jsonify({"saved": str(_GLOBAL_GRAPH_PATH)})

    @app.route("/api/graph/node", methods=["POST"])
    def upsert_node():
        global _GLOBAL_GRAPH
        if _GLOBAL_GRAPH is None:
            return jsonify({"error": "no graph loaded"}), 400
        body: Dict[str, Any] = request.get_json(force=True)
        node_id = str(body["node_id"])
        label = str(body.get("label", ""))
        ntype = str(body.get("type", "event"))
        source_ids = list(body.get("source_ids", []))

        existing = [n for n in _GLOBAL_GRAPH.nodes if n.node_id == node_id]
        if existing:
            existing[0].label = label
            existing[0].type = ntype
            existing[0].source_ids = source_ids
        else:
            from storydag.cnge.types import GraphNode
            _GLOBAL_GRAPH.nodes.append(GraphNode(node_id=node_id, label=label, type=ntype, source_ids=source_ids))

        return jsonify({"ok": True})

    @app.route("/api/graph/node/<node_id>", methods=["DELETE"])
    def delete_node(node_id: str):
        global _GLOBAL_GRAPH
        if _GLOBAL_GRAPH is None:
            return jsonify({"error": "no graph loaded"}), 400
        _GLOBAL_GRAPH.nodes = [n for n in _GLOBAL_GRAPH.nodes if n.node_id != node_id]
        _GLOBAL_GRAPH.edges = [e for e in _GLOBAL_GRAPH.edges if e.source != node_id and e.target != node_id]
        return jsonify({"ok": True})

    @app.route("/api/graph/edge", methods=["POST"])
    def upsert_edge():
        global _GLOBAL_GRAPH
        if _GLOBAL_GRAPH is None:
            return jsonify({"error": "no graph loaded"}), 400
        body: Dict[str, Any] = request.get_json(force=True)
        edge_id = str(body.get("edge_id", ""))
        source = str(body["source"])
        target = str(body["target"])
        etype = str(body.get("type", "motivates"))
        strength = float(body.get("strength", 1.0))

        existing = [e for e in _GLOBAL_GRAPH.edges if e.edge_id == edge_id]
        if existing:
            existing[0].source = source
            existing[0].target = target
            existing[0].type = etype
            existing[0].strength = strength
        else:
            from storydag.cnge.types import GraphEdge
            _GLOBAL_GRAPH.edges.append(GraphEdge(edge_id=edge_id, source=source, target=target, type=etype, strength=strength))

        return jsonify({"ok": True})

    @app.route("/api/graph/edge/<edge_id>", methods=["DELETE"])
    def delete_edge(edge_id: str):
        global _GLOBAL_GRAPH
        if _GLOBAL_GRAPH is None:
            return jsonify({"error": "no graph loaded"}), 400
        _GLOBAL_GRAPH.edges = [e for e in _GLOBAL_GRAPH.edges if e.edge_id != edge_id]
        return jsonify({"ok": True})

    @app.route("/api/outputs", methods=["GET"])
    def list_outputs():
        root = _resolve_outputs()
        return jsonify(_list_graphs(root))

    return app


def _list_graphs(root: Path) -> list[Dict[str, str]]:
    candidates: list[Dict[str, str]] = []
    if not root.is_dir():
        return candidates
    for child in sorted(root.iterdir()):
        gpath = child / "causal_graph.json"
        if gpath.is_file():
            candidates.append({"title": child.name, "path": str(gpath.resolve())})
    return candidates


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m storydag.cnge.auditor.app [graph.json]``."""
    import argparse
    parser = argparse.ArgumentParser(description="StoryDAG causal graph auditor")
    parser.add_argument("graph", nargs="?", default=None, help="Path to causal_graph.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args(argv)

    app = create_app(args.graph)
    print(f"  🎭 StoryDAG Auditor  http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
