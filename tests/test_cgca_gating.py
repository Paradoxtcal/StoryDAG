"""Tests for CGCA causal-history embedding and logit gating."""

import numpy as np
import pytest

from storydag.cgca.gating import (
    GateConfig,
    TokenEmbeddingCache,
    apply_gate_to_logits,
    build_token_gate_map,
    compute_gate_value,
    embed_causal_history,
    gate_for_token,
    token_compatibility,
)
from storydag.cgca.logits_processor import CausalGateLogitsProcessor


def _mock_embedder(vectors: dict[str, np.ndarray]):
    def embed(texts):
        return np.stack([vectors[text] for text in texts])

    return embed


def test_compute_gate_value_piecewise_ramp():
    config = GateConfig(low_threshold=0.15, high_threshold=0.45)
    assert compute_gate_value(0.10, config) == 0.0
    assert compute_gate_value(0.50, config) == 1.0
    assert compute_gate_value(0.30, config) == pytest.approx(0.5)


def test_token_compatibility_and_history_embedding():
    embedder = _mock_embedder(
        {
            "history": np.array([1.0, 0.0]),
            "aligned": np.array([1.0, 0.0]),
            "orthogonal": np.array([0.0, 1.0]),
        }
    )
    history = embed_causal_history("history", embedder)
    cache = TokenEmbeddingCache(embedder)

    assert token_compatibility(cache.encode("aligned"), history) == pytest.approx(1.0)
    assert token_compatibility(cache.encode("orthogonal"), history) == pytest.approx(0.0)


def test_build_token_gate_map_and_apply_to_logits():
    embedder = _mock_embedder(
        {
            "history": np.array([1.0, 0.0, 0.0]),
            "good": np.array([1.0, 0.0, 0.0]),
            "bad": np.array([0.0, 1.0, 0.0]),
        }
    )
    history = embed_causal_history("history", embedder)
    cache = TokenEmbeddingCache(embedder)
    config = GateConfig()

    gates = build_token_gate_map({1: "good", 2: "bad"}, history, cache, config)
    assert gates[1] == 1.0
    assert gates[2] == 0.0

    logits = np.array([0.0, 2.0, 2.0, 0.0])
    modified = apply_gate_to_logits(logits, gates, config)
    assert modified[1] > modified[2]


def test_gate_for_token_uses_cache():
    embedder = _mock_embedder({"history": np.array([1.0, 0.0]), "token": np.array([1.0, 0.0])})
    history = embed_causal_history("history", embedder)
    cache = TokenEmbeddingCache(embedder)
    assert gate_for_token("token", history, cache, GateConfig()) == 1.0


def test_causal_gate_logits_processor_batch_and_single():
    embedder = _mock_embedder(
        {
            "history text": np.array([1.0, 0.0]),
            "<pad>": np.array([0.5, 0.5]),
            "keep": np.array([1.0, 0.0]),
            "drop": np.array([0.0, 1.0]),
        }
    )
    processor = CausalGateLogitsProcessor(
        "history text",
        ["<pad>", "keep", "drop"],
        embedder=embedder,
    )

    single = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    modified_single = processor(None, single)
    assert modified_single[1] > modified_single[2]

    batch = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    modified_batch = processor(None, batch)
    assert modified_batch.shape == batch.shape
    assert modified_batch[0, 1] > modified_batch[0, 2]
