"""Tests for CNGE coreference resolution and node clustering."""

import numpy as np

from storydag.cnge.coref import resolve_coreferences
from storydag.cnge.types import CausalTriple, GraphEdge, GraphNode


def _triple(
    source_id: str,
    source_label: str,
    target_id: str,
    target_label: str,
    edge_type: str = "motivates",
    source_type: str = "event",
    target_type: str = "intention",
) -> CausalTriple:
    return CausalTriple(
        source_id=source_id,
        source_label=source_label,
        source_type=source_type,
        edge_type=edge_type,
        target_id=target_id,
        target_label=target_label,
        target_type=target_type,
    )


def _mock_embedder(vectors: dict[str, np.ndarray]):
    def embed(labels):
        return np.stack([vectors[label] for label in labels])

    return embed


def test_resolve_coreferences_merges_similar_labels():
    triples = [
        _triple("ch1_s1_n1", "林远得知师妹被囚", "ch1_s1_n2", "林远决定救人"),
        _triple("ch2_s1_n1", "林远得知师妹被囚", "ch2_s1_n2", "林远决定夜袭后山"),
    ]
    vectors = {
        "林远得知师妹被囚": np.array([1.0, 0.0, 0.0]),
        "林远决定救人": np.array([0.0, 1.0, 0.0]),
        "林远决定夜袭后山": np.array([0.0, 0.9, 0.1]),
    }
    nodes, edges = resolve_coreferences(
        triples,
        similarity_threshold=0.85,
        embedder=_mock_embedder(vectors),
    )

    assert len(nodes) == 2
    revelation_node = next(node for node in nodes if "ch1_s1_n1" in node.source_ids)
    assert revelation_node.node_id == "n1"
    assert revelation_node.source_ids == ["ch1_s1_n1", "ch2_s1_n1"]
    intention_node = next(node for node in nodes if revelation_node.node_id != node.node_id)
    assert intention_node.source_ids == ["ch1_s1_n2", "ch2_s1_n2"]
    assert len(edges) == 1
    assert edges[0].source == "n1" and edges[0].target == "n2"


def test_resolve_coreferences_keeps_dissimilar_labels_separate():
    triples = [
        _triple("ch1_s1_n1", "事件甲", "ch1_s1_n2", "事件乙"),
    ]
    vectors = {
        "事件甲": np.array([1.0, 0.0]),
        "事件乙": np.array([0.0, 1.0]),
    }
    nodes, edges = resolve_coreferences(
        triples,
        embedder=_mock_embedder(vectors),
    )

    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].edge_id == "e1"
    assert edges[0].source == "n1"
    assert edges[0].target == "n2"


def test_resolve_coreferences_deduplicates_edges():
    triples = [
        _triple("ch1_s1_n1", "事件甲", "ch1_s1_n2", "事件乙"),
        _triple("ch1_s2_n1", "事件甲", "ch1_s2_n2", "事件乙"),
    ]
    vectors = {
        "事件甲": np.array([1.0, 0.0]),
        "事件乙": np.array([0.0, 1.0]),
    }
    _, edges = resolve_coreferences(triples, embedder=_mock_embedder(vectors))
    assert len(edges) == 1


def test_resolve_coreferences_drops_self_loops_after_merge():
    triples = [
        _triple("ch1_s1_n1", "同一事件", "ch1_s1_n2", "同一事件"),
    ]
    vectors = {"同一事件": np.array([1.0, 0.0, 0.0])}
    nodes, edges = resolve_coreferences(triples, embedder=_mock_embedder(vectors))
    assert len(nodes) == 1
    assert edges == []


def test_resolve_coreferences_empty_input():
    nodes, edges = resolve_coreferences([])
    assert nodes == []
    assert edges == []


def test_graph_node_and_edge_types():
    triples = [_triple("ch1_s1_n1", "甲", "ch1_s1_n2", "乙")]
    vectors = {"甲": np.array([1.0, 0.0]), "乙": np.array([0.0, 1.0])}
    nodes, edges = resolve_coreferences(triples, embedder=_mock_embedder(vectors))
    assert isinstance(nodes[0], GraphNode)
    assert isinstance(edges[0], GraphEdge)
    assert edges[0].strength == 1.0
