"""LLM-based causal triple extraction for CNGE."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from storydag.cnge.prompts import build_extraction_messages
from storydag.cnge.segmentation import Chapter, Scene
from storydag.cnge.types import EDGE_TYPES, NODE_TYPES, CausalTriple
from storydag.llm.client import LLMClient

_REQUIRED_TRIPLE_FIELDS = (
    "source",
    "source_label",
    "source_type",
    "edge_type",
    "target",
    "target_label",
    "target_type",
)


class CausalExtractionError(ValueError):
    """Raised when LLM output cannot be parsed or validated."""


def parse_triples_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse LLM JSON output into a list of raw triple dicts."""
    text = response_text.strip()
    if not text:
        raise CausalExtractionError("LLM 返回为空")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CausalExtractionError(f"LLM 输出不是合法 JSON: {exc}") from exc

    if isinstance(payload, list):
        triples = payload
    elif isinstance(payload, dict) and isinstance(payload.get("triples"), list):
        triples = payload["triples"]
    else:
        raise CausalExtractionError("JSON 根对象必须包含 triples 数组")

    if not all(isinstance(item, dict) for item in triples):
        raise CausalExtractionError("triples 数组中的每个元素必须是对象")

    return triples


def validate_triple(raw: Mapping[str, Any], scene_id: str) -> CausalTriple:
    """Validate one raw triple dict and convert it to ``CausalTriple``."""
    missing = [field for field in _REQUIRED_TRIPLE_FIELDS if field not in raw]
    if missing:
        raise CausalExtractionError(f"三元组缺少字段: {', '.join(missing)}")

    edge_type = str(raw["edge_type"]).strip()
    source_type = str(raw["source_type"]).strip()
    target_type = str(raw["target_type"]).strip()

    if edge_type not in EDGE_TYPES:
        raise CausalExtractionError(f"非法 edge_type: {edge_type}")
    if source_type not in NODE_TYPES:
        raise CausalExtractionError(f"非法 source_type: {source_type}")
    if target_type not in NODE_TYPES:
        raise CausalExtractionError(f"非法 target_type: {target_type}")

    source_id = str(raw["source"]).strip()
    target_id = str(raw["target"]).strip()
    prefix = f"{scene_id}_"
    if not source_id.startswith(prefix):
        raise CausalExtractionError(f"source ID 必须以 {prefix} 为前缀: {source_id}")
    if not target_id.startswith(prefix):
        raise CausalExtractionError(f"target ID 必须以 {prefix} 为前缀: {target_id}")
    if source_id == target_id:
        raise CausalExtractionError("source 与 target 不能相同")

    return CausalTriple(
        source_id=source_id,
        source_label=str(raw["source_label"]).strip(),
        source_type=source_type,
        edge_type=edge_type,
        target_id=target_id,
        target_label=str(raw["target_label"]).strip(),
        target_type=target_type,
    )


def normalize_triples(raw_triples: Sequence[Mapping[str, Any]], scene_id: str) -> List[CausalTriple]:
    """Validate and convert a batch of raw triple dicts."""
    return [validate_triple(item, scene_id) for item in raw_triples]


def extract_triples(scene: Scene, client: LLMClient) -> List[CausalTriple]:
    """Extract causal triples from a single scene chunk via LLM."""
    return extract_triples_from_text(scene.scene_id, scene.text, client)


def extract_triples_from_text(scene_id: str, scene_text: str, client: LLMClient) -> List[CausalTriple]:
    """Extract causal triples from raw scene text via LLM."""
    if not scene_text.strip():
        return []

    messages = build_extraction_messages(scene_id, scene_text)
    response = client.chat(messages, temperature=0.0, json_mode=True)
    raw_triples = parse_triples_response(response.content)
    return normalize_triples(raw_triples, scene_id)


def extract_novel_triples(chapters: Iterable[Chapter], client: LLMClient) -> List[CausalTriple]:
    """Extract causal triples from every scene in a segmented novel."""
    all_triples: List[CausalTriple] = []
    for chapter in chapters:
        for scene in chapter.scenes:
            all_triples.extend(extract_triples(scene, client))
    return all_triples
