"""Graph-driven hard blacklist for unknown revelation secrets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np

from storydag.cgca.history import CausalHistory
from storydag.cnge.graph import CausalGraph
from storydag.cnge.types import GraphNode

CHINESE_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
NEG_INF = float("-inf")


@dataclass(frozen=True)
class BlacklistedSecret:
    """A revelation the character must not disclose yet."""

    node_id: str
    label: str
    trigger_tokens: List[str] = field(default_factory=list)


def known_history_node_ids(history: CausalHistory) -> Set[str]:
    """Nodes that are part of the character's causal history at this scene."""
    return set(history.ancestor_node_ids) | set(history.scene_node_ids)


def find_unknown_secrets(graph: CausalGraph, history: CausalHistory) -> List[GraphNode]:
    """Return revelation nodes that are not yet in the character's causal history."""
    known = known_history_node_ids(history)
    return [node for node in graph.nodes if node.type == "revelation" and node.node_id not in known]


def extract_trigger_tokens(label: str, extra_tokens: Iterable[str] = ()) -> List[str]:
    """Derive trigger tokens from a secret label plus optional explicit keywords."""
    tokens = {token.strip() for token in extra_tokens if token and token.strip()}
    for match in CHINESE_TOKEN_RE.findall(label):
        tokens.add(match)
    return sorted(tokens, key=len, reverse=True)


def build_blacklisted_secrets(
    graph: CausalGraph,
    history: CausalHistory,
    *,
    extra_triggers: Mapping[str, Sequence[str]] | None = None,
) -> List[BlacklistedSecret]:
    """Build secret blacklist entries for a character at the current scene."""
    extras = extra_triggers or {}
    secrets: List[BlacklistedSecret] = []
    for node in find_unknown_secrets(graph, history):
        secrets.append(
            BlacklistedSecret(
                node_id=node.node_id,
                label=node.label,
                trigger_tokens=extract_trigger_tokens(
                    node.label,
                    extras.get(node.node_id, ()),
                ),
            )
        )
    return secrets


def map_triggers_to_token_ids(
    trigger_tokens: Iterable[str],
    token_id_to_text: Mapping[int, str],
) -> Set[int]:
    """Map trigger strings to tokenizer ids via substring matching."""
    blocked: Set[int] = set()
    triggers = [token for token in trigger_tokens if token]
    for token_id, token_text in token_id_to_text.items():
        normalized = token_text.strip()
        if not normalized:
            continue
        for trigger in triggers:
            if trigger in normalized or normalized in trigger:
                blocked.add(token_id)
    return blocked


def build_blocked_token_ids(
    secrets: Sequence[BlacklistedSecret],
    token_id_to_text: Mapping[int, str],
) -> Set[int]:
    """Collect all token ids that must be hard-blocked."""
    blocked: Set[int] = set()
    for secret in secrets:
        blocked.update(map_triggers_to_token_ids(secret.trigger_tokens, token_id_to_text))
    return blocked


def apply_hard_blacklist(logits: np.ndarray, blocked_token_ids: Iterable[int]) -> np.ndarray:
    """Set blocked token logits to ``-inf``."""
    modified = np.array(logits, dtype=np.float64, copy=True)
    for token_id in blocked_token_ids:
        if 0 <= token_id < modified.shape[-1]:
            modified[token_id] = NEG_INF
    return modified


def text_violates_blacklist(text: str, secrets: Sequence[BlacklistedSecret]) -> List[str]:
    """Return trigger tokens that appear in generated text."""
    violations: List[str] = []
    for secret in secrets:
        for trigger in secret.trigger_tokens:
            if trigger and trigger in text and trigger not in violations:
                violations.append(trigger)
    return violations
