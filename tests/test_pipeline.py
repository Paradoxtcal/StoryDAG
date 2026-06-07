"""Tests for the end-to-end StoryDAG pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from storydag.causopt.mcts import MCTSConfig
from storydag.causopt.models import DramaticObjectives
from storydag.cnge.graph import CausalGraph
from storydag.llm import LLMClient
from storydag.pipeline import (
    PipelineConfig,
    discover_characters,
    extract_causal_graph,
    run_pipeline,
    sanitize_output_name,
)
from storydag.serialization.metrics import compute_metrics_from_files
from storydag.serialization.yaml_reader import read_script

CHAIN_PAYLOAD = {
    "triples": [
        {
            "source": "ch1_s1_n1",
            "source_label": "林远得知秘密",
            "source_type": "revelation",
            "edge_type": "motivates",
            "target": "ch1_s1_n2",
            "target_label": "林远决定行动",
            "target_type": "intention",
        },
        {
            "source": "ch1_s1_n2",
            "source_label": "林远决定行动",
            "source_type": "intention",
            "edge_type": "motivates",
            "target": "ch1_s1_n3",
            "target_label": "林远潜入后山",
            "target_type": "event",
        },
    ]
}


def _orthogonal_embedder(labels):
    size = max(len(labels), 4)
    vectors = []
    for index in range(len(labels)):
        vector = np.zeros(size, dtype=float)
        vector[index] = 1.0
        vectors.append(vector)
    return np.stack(vectors)


def _mock_client() -> LLMClient:
    client = LLMClient(api_key="test-key", base_url="https://example.com/v1")
    client._client = MagicMock()

    def _create_completion(**kwargs):
        messages = kwargs["messages"]
        user_text = messages[-1]["content"] if messages else ""
        if "scene_id" in user_text or "triples" in str(messages[0]["content"]):
            content = json.dumps(CHAIN_PAYLOAD, ensure_ascii=False)
        else:
            content = "我都知道了。"
        return MagicMock(
            model="gpt-4-turbo",
            usage=None,
            choices=[
                MagicMock(
                    message=MagicMock(content=content),
                    logprobs=None,
                )
            ],
        )

    client._client.chat.completions.create = MagicMock(side_effect=_create_completion)
    return client


def _fast_config() -> PipelineConfig:
    return PipelineConfig(
        mcts=MCTSConfig(max_iterations=200, time_budget_sec=5.0),
        dramatic_objectives=DramaticObjectives(max_scenes=10, min_scenes_per_act=1),
    )


def test_sanitize_output_name():
    assert sanitize_output_name("测试/剧本") == "测试_剧本"
    assert sanitize_output_name("   ") == "untitled"


def test_discover_characters_from_graph():
    graph = extract_causal_graph(
        "林远得知秘密，林远决定行动，林远潜入后山。",
        _mock_client(),
        _fast_config(),
        embedder=_orthogonal_embedder,
    )
    assert discover_characters(graph) == ["林远"]


def test_discover_characters_with_explicit_list():
    graph = extract_causal_graph(
        "林远得知秘密，林远决定行动，林远潜入后山。",
        _mock_client(),
        _fast_config(),
        embedder=_orthogonal_embedder,
    )
    # explicit list overrides inference
    assert discover_characters(graph, explicit=["林远", "张三"]) == ["张三", "林远"]


def test_pipeline_config_from_env_reads_character_list():
    config = PipelineConfig.from_env({"CHARACTER_LIST": "龙皓晨, 巴尔扎, 小女孩"})
    assert config.characters == ["龙皓晨", "巴尔扎", "小女孩"]


def test_pipeline_config_from_env_empty_character_list_is_none():
    config = PipelineConfig.from_env({})
    assert config.characters is None


def test_extract_causal_graph_builds_chain_dag():
    graph = extract_causal_graph(
        "林远得知秘密，林远决定行动，林远潜入后山。",
        _mock_client(),
        _fast_config(),
        embedder=_orthogonal_embedder,
    )
    assert isinstance(graph, CausalGraph)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2


def test_pipeline_config_from_env_reads_mcts_values():
    config = PipelineConfig.from_env(
        {
            "EMBEDDING_MODEL": "test-embed",
            "MCTS_ITERATIONS": "42",
            "MCTS_TIME_BUDGET_SEC": "12.5",
            "CAUSOPT_MIN_SCENES_PER_ACT": "1",
        }
    )
    assert config.embedding_model == "test-embed"
    assert config.mcts.max_iterations == 42
    assert config.mcts.time_budget_sec == 12.5
    assert config.dramatic_objectives.min_scenes_per_act == 1


def test_run_pipeline_writes_artifacts(tmp_path: Path):
    novel_path = tmp_path / "novel.txt"
    novel_path.write_text("林远得知秘密，林远决定行动，林远潜入后山。", encoding="utf-8")
    output_dir = tmp_path / "out"

    result = run_pipeline(
        novel_path,
        "测试剧本",
        output_dir=output_dir,
        client=_mock_client(),
        config=_fast_config(),
        embedder=_orthogonal_embedder,
    )

    graph_path = output_dir / "causal_graph.json"
    script_path = output_dir / "script.yaml"
    metrics_path = output_dir / "metrics.json"

    assert graph_path.exists()
    assert script_path.exists()
    assert metrics_path.exists()
    assert result.metrics.total_edges == 2
    assert result.script.title == "测试剧本"

    script = read_script(script_path)
    metrics = compute_metrics_from_files(script_path, graph_path)
    assert metrics.title == "测试剧本"
    assert len(script.acts) >= 1
