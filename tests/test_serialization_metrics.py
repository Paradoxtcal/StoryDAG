"""Tests for causal closure and consistency metrics."""

from pathlib import Path

from storydag.causopt.optimize import SceneRecord, SceneSequence
from storydag.cnge.graph import build_dag
from storydag.cnge.types import GraphEdge, GraphNode
from storydag.serialization import (
    ScriptBeat,
    ScriptCharacterLine,
    build_script,
    compute_ccr,
    compute_metrics,
    compute_metrics_from_files,
    detect_plot_holes,
    verify_backlinks,
    write_script,
)


def _graph_and_script():
    graph = build_dag(
        [
            GraphNode("n1", "林远得知秘密", "revelation"),
            GraphNode("n2", "林远决定行动", "intention"),
            GraphNode("n3", "林远实施后山计划", "event"),
        ],
        [
            GraphEdge("e1", "n1", "n2", "motivates"),
            GraphEdge("e2", "n2", "n3", "motivates"),
        ],
    )
    sequence = SceneSequence(
        scenes=[
            SceneRecord("S01", 1, ["n1"], {"earliest": 0, "latest": 0}),
            SceneRecord("S02", 2, ["n2"], {"earliest": 1, "latest": 1}),
            SceneRecord("S03", 3, ["n3"], {"earliest": 2, "latest": 2}),
        ],
        satisfied_edges={"e1": 1, "e2": 2},
        score=0.9,
    )
    script = build_script(
        "测试剧本",
        sequence,
        settings={"S01": "城外", "S02": "山路", "S03": "后山"},
        beats_by_scene={
            "S01": [
                ScriptBeat(
                    description="开端",
                    characters=[
                        ScriptCharacterLine(
                            character="林远",
                            type="dialogue",
                            text="我都知道了。",
                            causal_backlink=[],
                        )
                    ],
                )
            ],
            "S02": [
                ScriptBeat(
                    description="决心",
                    characters=[
                        ScriptCharacterLine(
                            character="林远",
                            type="dialogue",
                            text="今夜行动。",
                            causal_backlink=["e1"],
                        )
                    ],
                )
            ],
            "S03": [
                ScriptBeat(
                    description="行动",
                    characters=[
                        ScriptCharacterLine(
                            character="林远",
                            type="action",
                            text="林远潜入后山。",
                            causal_backlink=["e1", "e2"],
                        )
                    ],
                )
            ],
        },
    )
    return graph, script


def test_compute_ccr_full_closure():
    graph, script = _graph_and_script()
    ccr, satisfied, total = compute_ccr(graph, script)
    assert total == 2
    assert satisfied == 2
    assert ccr == 1.0


def test_detect_plot_holes_on_missing_edge():
    graph, script = _graph_and_script()
    script.acts[-1].scenes[-1].assigned_node_ids = []
    plot_holes, _ = detect_plot_holes(graph, script)
    assert "e2" in plot_holes


def test_verify_backlinks_accepts_ordered_references():
    graph, script = _graph_and_script()
    assert verify_backlinks(graph, script) == []


def test_compute_metrics_report_and_cli_files(tmp_path: Path):
    graph, script = _graph_and_script()
    report = compute_metrics(graph, script)
    assert report.ccr == 1.0
    assert report.character_consistency["林远"] is True
    assert report.causal_density["S02"] == 1
    assert report.causal_density["S03"] == 1

    script_path = tmp_path / "script.yaml"
    graph_path = tmp_path / "graph.json"
    write_script(script, script_path)
    graph.save(graph_path)

    loaded_report = compute_metrics_from_files(script_path, graph_path)
    assert loaded_report.ccr == 1.0
    assert loaded_report.total_edges == 2
