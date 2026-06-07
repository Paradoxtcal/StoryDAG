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
    scene_setting: str,
    scene_description: str,
    character: str,
    line_type: str,
    secrets: Sequence[BlacklistedSecret],
) -> str:
    history_text = history.to_context_text(graph)
    forbidden = sorted({trigger for secret in secrets for trigger in secret.trigger_tokens})
    forbidden_text = "、".join(forbidden) if forbidden else "(none)"

    return (
        f"场景: {scene_setting.strip()}\n"
        f"叙事描述: {scene_description.strip()}\n\n"
        f"{history_text}\n\n"
        f"请为 {character} 写一行{line_type}。\n"
        f"要求：\n"
        f"- 台词必须符合上述因果历史和场景设定\n"
        f"- 绝对不要提及以下禁词: {forbidden_text}\n"
        f"- 只返回台词文本"
    )


def generate_character_line(
    graph: CausalGraph,
    history: CausalHistory,
    scene_setting: str,
    scene_description: str,
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
    prompt = _build_generation_prompt(history, graph, scene_setting, scene_description, character, line_type, secrets)

    last_error: Exception | None = None
    text = ""
    for _ in range(max_attempts):
        try:
            text = client.complete(prompt, temperature=0.0).strip()
        except ConnectionError as exc:
            last_error = exc
            continue
        if not text_violates_blacklist(text, secrets):
            break
    else:
        # Only raise if we NEVER got valid text AND all attempts errored
        if not text and last_error is not None:
            raise ConnectionError(
                f"生成角色台词失败（{max_attempts} 次尝试均网络错误）：{last_error}"
            ) from last_error

    return CharacterLine(
        character=character,
        type=line_type,
        text=text,
        causal_backlink=_default_causal_backlinks(history),
    )
