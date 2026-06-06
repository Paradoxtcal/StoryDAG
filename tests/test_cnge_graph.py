"""Tests for CNGE DAG construction and serialization."""

import json

import pytest

from storydag.cnge.graph import CausalGraph, build_dag
from storydag.cnge.types import GraphEdge, GraphNode


def _node(node_id: str, label: str, node_type: str = "event") -> GraphNode:
    return GraphNode(node_id=node_id, label=label, type=node_type, source_ids=[node_id])


def _edge(edge_id: str, source: str, target: str, edge_type: str = "motivates", strength: float = 1.0) -> GraphEdge:
    return GraphEdge(edge_id=edge_id, source=source, target=target, type=edge_type, strength=strength)


def test_build_dag_produces_topological_order():
    nodes = [_node("n1", "甲"), _node("n2", "乙"), _node("n3", "丙")]
    edges = [_edge("e1", "n1", "n2"), _edge("e2", "n2", "n3")]
    graph = build_dag(nodes, edges)

    assert graph.topological_order == ["n1", "n2", "n3"]
    assert len(graph.edges) == 2
    assert graph.removed_cycle_edges == []


def test_build_dag_breaks_cycle_by_removing_weakest_edge():
    nodes = [_node("n1", "甲"), _node("n2", "乙"), _node("n3", "丙")]
    edges = [
        _edge("e1", "n1", "n2", strength=1.0),
        _edge("e2", "n2", "n3", strength=1.0),
        _edge("e3", "n3", "n1", strength=0.3),
    ]
    graph = build_dag(nodes, edges)

    assert len(graph.edges) == 2
    assert graph.removed_cycle_edges[0].edge_id == "e3"
    assert _is_valid_topological_order(graph.topological_order, graph.edges)


def test_build_dag_empty_graph():
    graph = build_dag([], [])
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.topological_order == []


def test_causal_graph_json_roundtrip():
    nodes = [_node("n1", "甲"), _node("n2", "乙")]
    edges = [_edge("e1", "n1", "n2")]
    original = build_dag(nodes, edges)

    restored = CausalGraph.from_json(original.to_json())
    assert restored.to_dict() == original.to_dict()


def test_causal_graph_save_and_load(tmp_path):
    nodes = [_node("n1", "甲"), _node("n2", "乙")]
    edges = [_edge("e1", "n1", "n2")]
    graph = build_dag(nodes, edges)
    path = tmp_path / "graph.json"
    graph.save(path)

    loaded = CausalGraph.load(path)
    assert loaded.topological_order == graph.topological_order
    assert json.loads(path.read_text(encoding="utf-8"))["nodes"][0]["node_id"] == "n1"


def test_build_dag_raises_if_cycles_remain_when_break_disabled():
    nodes = [_node("n1", "甲"), _node("n2", "乙")]
    edges = [_edge("e1", "n1", "n2"), _edge("e2", "n2", "n1")]
    with pytest.raises(ValueError, match="cycles"):
        build_dag(nodes, edges, break_cycles=False)


def _is_valid_topological_order(order, edges) -> bool:
    positions = {node_id: index for index, node_id in enumerate(order)}
    return all(positions[edge.source] < positions[edge.target] for edge in edges)
