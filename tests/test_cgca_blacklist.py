"""Tests for CGCA hard blacklist and character line generation."""

from unittest.mock import MagicMock

import numpy as np

from storydag.cgca import (
    CharacterLine,
    CombinedCausalLogitsProcessor,
    build_blacklisted_secrets,
    extract_history_for_scene,
    generate_character_line,
)
from storydag.cgca.blacklist import (
    apply_hard_blacklist,
    build_blocked_token_ids,
    find_unknown_secrets,
    text_violates_blacklist,
)
from storydag.cgca.gating import GateConfig
from storydag.cgca.logits_processor import CausalGateLogitsProcessor
from storydag.cnge.graph import build_dag
from storydag.cnge.types import GraphEdge, GraphNode
from storydag.llm import LLMClient, LLMResponse


def _mock_embedder(vectors: dict[str, np.ndarray]):
    def embed(texts):
        return np.stack([vectors[text] for text in texts])

    return embed


def _sample_graph():
    nodes = [
        GraphNode("n1", "林远得知师妹被囚", "revelation"),
        GraphNode("n2", "林远决定救人", "intention"),
        GraphNode("n3", "林远来到后山", "event"),
        GraphNode("n4", "师父才是幕后黑手", "revelation"),
    ]
    edges = [
        GraphEdge("e1", "n1", "n2", "motivates"),
        GraphEdge("e2", "n2", "n3", "motivates"),
    ]
    return build_dag(nodes, edges)


def test_find_unknown_secrets_excludes_history_nodes():
    graph = _sample_graph()
    history = extract_history_for_scene(graph, "林远", "S03", ["n3"])
    secrets = find_unknown_secrets(graph, history)
    assert [secret.node_id for secret in secrets] == ["n4"]


def test_build_blacklisted_secrets_and_token_mapping():
    graph = _sample_graph()
    history = extract_history_for_scene(graph, "林远", "S03", ["n3"])
    secrets = build_blacklisted_secrets(
        graph,
        history,
        extra_triggers={"n4": ["师父"]},
    )
    assert secrets[0].node_id == "n4"
    assert "师父" in secrets[0].trigger_tokens

    blocked = build_blocked_token_ids(secrets, {10: "师父", 11: "后山", 12: "hello"})
    assert blocked == {10}


def test_apply_hard_blacklist_sets_neg_inf():
    logits = np.array([1.0, 2.0, 3.0])
    modified = apply_hard_blacklist(logits, [1])
    assert modified[1] == float("-inf")
    assert modified[0] == 1.0


def test_text_violates_blacklist_detects_trigger():
    graph = _sample_graph()
    history = extract_history_for_scene(graph, "林远", "S03", ["n3"])
    secrets = build_blacklisted_secrets(graph, history)
    assert text_violates_blacklist("听说师父才是幕后黑手", secrets)


def test_combined_processor_applies_gate_then_blacklist():
    embedder = _mock_embedder(
        {
            "history": np.array([1.0, 0.0]),
            "<pad>": np.array([0.5, 0.5]),
            "keep": np.array([1.0, 0.0]),
            "secret": np.array([0.0, 1.0]),
        }
    )
    gate = CausalGateLogitsProcessor(
        "history",
        ["<pad>", "keep", "secret"],
        embedder=embedder,
        config=GateConfig(),
    )
    processor = CombinedCausalLogitsProcessor(gate, blocked_token_ids={2})
    scores = np.array([0.0, 2.0, 5.0], dtype=np.float32)
    modified = processor(None, scores)
    assert modified[2] == float("-inf")
    assert modified[1] > modified[0]


def test_generate_character_line_returns_backlinks():
    graph = _sample_graph()
    history = extract_history_for_scene(graph, "林远", "S03", ["n3"])
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    client.complete = MagicMock(return_value="我先去后山看看。")

    line = generate_character_line(
        graph,
        history,
        scene_setting="后山夜探",
        scene_description="夜色笼罩后山。",
        character="林远",
        client=client,
    )

    assert isinstance(line, CharacterLine)
    assert line.character == "林远"
    assert line.type == "dialogue"
    assert line.text == "我先去后山看看。"
    assert "e1" in line.causal_backlink
