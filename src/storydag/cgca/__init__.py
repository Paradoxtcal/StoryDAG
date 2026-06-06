"""CGCA: Causal-Gated Character Adapter."""

from storydag.cgca.history import (
    CausalHistory,
    HistoryEdge,
    extract_history,
    extract_history_for_scene,
    node_involves_character,
    scene_nodes_for_character,
)

__all__ = [
    "CausalHistory",
    "HistoryEdge",
    "extract_history",
    "extract_history_for_scene",
    "node_involves_character",
    "scene_nodes_for_character",
]
