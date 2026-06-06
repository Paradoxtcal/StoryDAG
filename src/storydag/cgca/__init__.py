"""CGCA: Causal-Gated Character Adapter."""

from storydag.cgca.gating import (
    GateConfig,
    TokenEmbeddingCache,
    apply_gate_to_logits,
    build_token_gate_map,
    compute_gate_value,
    default_sentence_embedder,
    embed_causal_history,
    gate_for_token,
)
from storydag.cgca.history import (
    CausalHistory,
    HistoryEdge,
    extract_history,
    extract_history_for_scene,
    node_involves_character,
    scene_nodes_for_character,
)
from storydag.cgca.logits_processor import CausalGateLogitsProcessor

__all__ = [
    "CausalGateLogitsProcessor",
    "CausalHistory",
    "GateConfig",
    "TokenEmbeddingCache",
    "apply_gate_to_logits",
    "build_token_gate_map",
    "compute_gate_value",
    "default_sentence_embedder",
    "embed_causal_history",
    "gate_for_token",
    "HistoryEdge",
    "extract_history",
    "extract_history_for_scene",
    "node_involves_character",
    "scene_nodes_for_character",
]
