"""Character line generation with causal gating and hard blacklist."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from storydag.cgca.blacklist import (
    BlacklistedSecret,
    build_blacklisted_secrets,
    text_violates_blacklist,
)
from storydag.cgca.history import CausalHistory
from storydag.cnge.graph import CausalGraph
from storydag.llm import LLMClient


@dataclass
class CharacterLine:
    """Generated dialogue or action for one character in a scene."""

    character: str
    type: str
    text: str
    causal_backlink: List[str] = field(default_factory=list)


def _default_causal_backlinks(history: CausalHistory) -> List[str]:
    edge_ids: List[str] = []
    for edge in [*history.knowledge_edges, *history.motivation_edges]:
        if edge.edge_id and edge.edge_id not in edge_ids:
            edge_ids.append(edge.edge_id)
    return edge_ids


def _build_generation_prompt(
    history: CausalHistory,
    graph: CausalGraph,
    scene_context: str,
    character: str,
    line_type: str,
    secrets: Sequence[BlacklistedSecret],
) -> str:
    history_text = history.to_context_text(graph)
    forbidden = sorted({trigger for secret in secrets for trigger in secret.trigger_tokens})
    forbidden_text = "、".join(forbidden) if forbidden else "(none)"

    return (
        f"Scene context:\n{scene_context.strip()}\n\n"
        f"{history_text}\n\n"
        f"Write one {line_type} line for {character}.\n"
        f"The line must be consistent with the causal history above.\n"
        f"Never mention these forbidden secret tokens: {forbidden_text}\n"
        f"Return only the line text."
    )


def generate_character_line(
    graph: CausalGraph,
    history: CausalHistory,
    scene_context: str,
    character: str,
    client: LLMClient,
    *,
    line_type: str = "dialogue",
    extra_triggers: Optional[dict[str, Sequence[str]]] = None,
    max_attempts: int = 3,
) -> CharacterLine:
    """Generate a character line constrained by causal history and secret blacklist.

    Soft logit gating is applied in local ``transformers`` generation via
    ``CombinedCausalLogitsProcessor``. API-based generation uses prompt constraints
    plus post-check against the hard blacklist.
    """
    secrets = build_blacklisted_secrets(graph, history, extra_triggers=extra_triggers)
    prompt = _build_generation_prompt(history, graph, scene_context, character, line_type, secrets)

    text = ""
    for _ in range(max_attempts):
        text = client.complete(prompt, temperature=0.0).strip()
        if not text_violates_blacklist(text, secrets):
            break

    return CharacterLine(
        character=character,
        type=line_type,
        text=text,
        causal_backlink=_default_causal_backlinks(history),
    )
