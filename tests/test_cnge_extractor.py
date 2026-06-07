"""Tests for CNGE causal triple extraction."""

import json
from unittest.mock import MagicMock

import pytest

from storydag.cnge import (
    CausalExtractionError,
    CausalTriple,
    Scene,
    extract_novel_triples,
    extract_triples,
    extract_triples_from_text,
    normalize_triples,
    parse_triples_response,
    segment_novel,
)
from storydag.cnge.prompts import FEW_SHOT_EXAMPLES, build_extraction_messages
from storydag.llm import LLMClient, LLMResponse

VALID_PAYLOAD = {
    "triples": [
        {
            "source": "ch1_s1_n1",
            "source_label": "李明得知敌军来袭",
            "source_type": "revelation",
            "edge_type": "motivates",
            "target": "ch1_s1_n2",
            "target_label": "李明决定连夜搬救兵",
            "target_type": "intention",
        }
    ]
}


def test_parse_triples_response_accepts_wrapped_object():
    raw = parse_triples_response(json.dumps(VALID_PAYLOAD, ensure_ascii=False))
    assert len(raw) == 1
    assert raw[0]["source"] == "ch1_s1_n1"


def test_parse_triples_response_accepts_top_level_array():
    raw = parse_triples_response(json.dumps(VALID_PAYLOAD["triples"], ensure_ascii=False))
    assert len(raw) == 1


def test_parse_triples_response_rejects_invalid_json():
    with pytest.raises(CausalExtractionError, match="合法 JSON"):
        parse_triples_response("not-json")


def test_parse_triples_response_rejects_missing_triples_key():
    with pytest.raises(CausalExtractionError, match="triples"):
        parse_triples_response('{"nodes": []}')


def test_normalize_triples_validates_enums_and_ids():
    triples = normalize_triples(VALID_PAYLOAD["triples"], "ch1_s1")
    assert len(triples) == 1
    assert isinstance(triples[0], CausalTriple)
    assert triples[0].edge_type == "motivates"
    assert triples[0].source_type == "revelation"


def test_normalize_triples_rejects_bad_edge_type():
    bad = [{**VALID_PAYLOAD["triples"][0], "edge_type": "causes"}]
    with pytest.raises(CausalExtractionError, match="edge_type"):
        normalize_triples(bad, "ch1_s1")


def test_normalize_triples_rejects_id_prefix_mismatch():
    bad = [{**VALID_PAYLOAD["triples"][0], "source": "wrong_n1"}]
    with pytest.raises(CausalExtractionError, match="前缀"):
        normalize_triples(bad, "ch1_s1")


def test_build_extraction_messages_includes_scene_id_and_few_shots():
    messages = build_extraction_messages("ch2_s3", "场景正文")
    assert messages[0].role == "system"
    assert "ch2_s3" in messages[0].content
    assert len(FEW_SHOT_EXAMPLES) == 3
    assert messages[-1].role == "user"
    assert "scene_id: ch2_s3" in messages[-1].content
    assert "场景正文" in messages[-1].content


def test_extract_triples_from_text_uses_prompt_not_json_mode():
    """Cross-provider compat: we rely on the system prompt rather than
    ``response_format: {"type": "json_object"}`` because DeepSeek /
    Ollama / vLLM may reject that parameter."""
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    client._client = MagicMock()
    client._client.chat.completions.create = MagicMock(
        return_value=MagicMock(
            model="gpt-4-turbo",
            usage=None,
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=json.dumps(VALID_PAYLOAD, ensure_ascii=False)
                    ),
                    logprobs=None,
                )
            ],
        )
    )

    triples = extract_triples_from_text("ch1_s1", "李明得知敌军来袭。", client)
    assert len(triples) == 1
    assert triples[0].target_label == "李明决定连夜搬救兵"

    call_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
    # json_mode is NOT passed — cross-provider compat
    assert "response_format" not in call_kwargs


def test_extract_triples_accepts_scene_object():
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    client.chat = MagicMock(
        return_value=LLMResponse(
            content=json.dumps(VALID_PAYLOAD, ensure_ascii=False),
            model="gpt-4-turbo",
        )
    )
    scene = Scene(scene_id="ch1_s1", index=0, text="正文")
    triples = extract_triples(scene, client)
    assert len(triples) == 1


def test_extract_triples_from_text_empty_returns_empty_list():
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    client.chat = MagicMock()
    assert extract_triples_from_text("ch1_s1", "  ", client) == []
    client.chat.assert_not_called()


def test_extract_novel_triples_iterates_all_scenes():
    novel = "第一章 测试\n\n场景一。\n\n\n场景二。"
    chapters = segment_novel(novel)
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    client.chat = MagicMock(
        return_value=LLMResponse(
            content=json.dumps({"triples": []}, ensure_ascii=False),
            model="gpt-4-turbo",
        )
    )

    triples = extract_novel_triples(chapters, client)
    assert triples == []
    assert client.chat.call_count == sum(len(ch.scenes) for ch in chapters)
