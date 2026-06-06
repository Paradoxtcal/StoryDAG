"""Tests for CGCA causal history subgraph extraction."""

from storydag.cgca import CausalHistory, extract_history, extract_history_for_scene
from storydag.cnge.graph import build_dag
from storydag.cnge.types import GraphEdge, GraphNode


def _sample_graph():
    nodes = [
        GraphNode("n1", "林远得知师妹被囚", "revelation", ["ch1_s1_n1"]),
        GraphNode("n2", "林远决定连夜救人", "intention", ["ch1_s1_n2"]),
        GraphNode("n3", "林远潜入后山", "event", ["ch2_s1_n1"]),
        GraphNode("n4", "掌柜告知令牌下落", "event", ["ch2_s1_n2"]),
        GraphNode("n5", "阿青获知令牌在当铺", "revelation", ["ch2_s1_n3"]),
    ]
    edges = [
        GraphEdge("e1", "n1", "n2", "motivates"),
        GraphEdge("e2", "n2", "n3", "motivates"),
        GraphEdge("e3", "n4", "n5", "informs"),
    ]
    return build_dag(nodes, edges)


def test_extract_history_collects_ancestors_for_character():
    graph = _sample_graph()
    history = extract_history(graph, "林远", ["n3"], scene_id="S03")

    assert isinstance(history, CausalHistory)
    assert history.scene_node_ids == ["n3"]
    assert history.ancestor_node_ids == ["n1", "n2"]
    assert len(history.motivation_edges) == 2
    assert history.knowledge_edges == []


def test_extract_history_includes_informs_edges_for_character():
    graph = _sample_graph()
    history = extract_history(graph, "阿青", ["n5"], scene_id="S05")

    assert history.scene_node_ids == ["n5"]
    assert "n4" in history.ancestor_node_ids
    assert len(history.knowledge_edges) == 1
    assert history.knowledge_edges[0].type == "informs"


def test_to_context_text_contains_sections():
    graph = _sample_graph()
    history = extract_history_for_scene(graph, "林远", "S03", ["n3"])
    text = history.to_context_text(graph)

    assert "Character: 林远" in text
    assert "Causal history" in text
    assert "林远得知师妹被囚" in text
    assert "Active motivations" in text
    assert "林远决定连夜救人" in text


def test_extract_history_empty_when_character_not_in_scene():
    graph = _sample_graph()
    history = extract_history(graph, "阿青", ["n3"], scene_id="S03")
    assert history.scene_node_ids == []
    assert history.ancestor_node_ids == []
