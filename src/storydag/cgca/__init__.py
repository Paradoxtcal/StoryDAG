"""CGCA: Causal-Gated Character Adapter."""

from storydag.cgca.blacklist import (
    BlacklistedSecret,
    apply_hard_blacklist,
    build_blacklisted_secrets,
    build_blocked_token_ids,
    find_unknown_secrets,
    text_violates_blacklist,
)
from storydag.cgca.combined_processor import CombinedCausalLogitsProcessor
from storydag.cgca.generator import CharacterLine, generate_character_line
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
    "BlacklistedSecret",
    "CharacterLine",
    "CombinedCausalLogitsProcessor",
    "CausalGateLogitsProcessor",
    "CausalHistory",
    "GateConfig",
    "TokenEmbeddingCache",
    "apply_gate_to_logits",
    "apply_hard_blacklist",
    "build_blacklisted_secrets",
    "build_blocked_token_ids",
    "build_token_gate_map",
    "compute_gate_value",
    "default_sentence_embedder",
    "embed_causal_history",
    "gate_for_token",
    "HistoryEdge",
    "find_unknown_secrets",
    "generate_character_line",
    "extract_history",
    "extract_history_for_scene",
    "node_involves_character",
    "scene_nodes_for_character",
    "text_violates_blacklist",
]
