"""LLM-based causal triple extraction for CNGE."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from storydag.cnge.prompts import build_extraction_messages
from storydag.llm.types import ChatMessage
from storydag.cnge.segmentation import Chapter, Scene
from storydag.cnge.types import EDGE_TYPES, NODE_TYPES, CausalTriple
from storydag.llm.client import LLMClient
from storydag.llm.types import ChatMessage, LLMResponse

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


_RETRY_HINT = (
    "\n\n[上轮输出解析失败，请严格按以下格式返回纯 JSON："
    '\n{"triples": [{"source": "...", "source_label": "...", '
    '"source_type": "...", "edge_type": "...", '
    '"target": "...", "target_label": "...", "target_type": "..."}]}'
    "\n不要包含任何非 JSON 的说明文字。]"
)


def extract_triples_from_text(
    scene_id: str,
    scene_text: str,
    client: LLMClient,
    *,
    max_tokens: int = 8192,
    max_retries: int = 3,
) -> List[CausalTriple]:
    """Extract causal triples from raw scene text via LLM.

    **Not** passing ``json_mode=True`` for cross-provider compatibility
    (DeepSeek / Ollama / vLLM may reject ``response_format: {"type":
    "json_object"}``). The system prompt and few-shot examples already
    enforce JSON output.

    Returns ``[]`` on final failure rather than crashing the pipeline.
    """
    if not scene_text.strip():
        return []

    if len(scene_text) > 10000:
        print(
            f"  ⚠ 场景 {scene_id} 文本过长（{len(scene_text)} 字符），将被截断至 10000 字符",
            file=sys.stderr,
            flush=True,
        )
        scene_text = scene_text[:10000]

    messages = build_extraction_messages(scene_id, scene_text)
    last_response: Optional[LLMResponse] = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat(
                messages,
                temperature=0.0,
                json_mode=False,
                max_tokens=max_tokens,
            )
            last_response = response
            raw_triples = parse_triples_response(response.content)
            return normalize_triples(raw_triples, scene_id)
        except (CausalExtractionError, ConnectionError, RuntimeError) as exc:
            # Diagnose failure reason
            if isinstance(exc, CausalExtractionError) and "为空" in str(exc):
                reason = "未知"
                if last_response and hasattr(last_response.raw, "choices"):
                    ch = last_response.raw.choices[0]
                    reason = getattr(ch, "finish_reason", "unknown")
                print(
                    f"  ⚠ 场景 {scene_id} LLM 返回为空"
                    f"（文本 {len(scene_text)} 字符，finish_reason={reason}，第 {attempt + 1}/{max_retries + 1} 次）",
                    file=sys.stderr,
                    flush=True,
                )

            if attempt < max_retries:
                if isinstance(exc, CausalExtractionError):
                    last_user_idx = max(
                        i for i, m in enumerate(messages) if m.role == "user"
                    )
                    old = messages[last_user_idx]
                    if not old.content.endswith(_RETRY_HINT):
                        messages[last_user_idx] = ChatMessage(
                            role="user",
                            content=old.content + _RETRY_HINT,
                        )
                continue
            # Last attempt failed — skip so the pipeline continues
            detail = str(exc)
            if last_response and hasattr(last_response.raw, "choices"):
                ch = last_response.raw.choices[0]
                detail += f" (finish_reason={getattr(ch, 'finish_reason', 'unknown')})"
            print(
                f"  ✗ 场景 {scene_id} 因果抽取失败"
                f"（{max_retries} 次重试后仍失败: {detail}），已跳过",
                file=sys.stderr,
                flush=True,
            )
            return []


def extract_novel_triples(
    chapters: Iterable[Chapter],
    client: LLMClient,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[CausalTriple]:
    """Extract causal triples from every scene in a segmented novel.

    Scenes that fail extraction after all retries are skipped with a
    warning so that the pipeline can continue with partial results.
    """
    all_triples: List[CausalTriple] = []
    total = sum(len(c.scenes) for c in chapters)
    done = 0
    failures = 0
    for chapter in chapters:
        for scene in chapter.scenes:
            triples = extract_triples(scene, client)
            if not triples:
                failures += 1
            all_triples.extend(triples)
            done += 1
            if progress_callback:
                progress_callback(done, total)
    if failures:
        print(
            f"  警告：{failures}/{total} 个场景因果抽取失败，已跳过",
            file=sys.stderr,
            flush=True,
        )
    return all_triples
