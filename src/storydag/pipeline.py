"""End-to-end StoryDAG pipeline orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from storydag.causopt.mcts import MCTSConfig
from storydag.causopt.models import CausalDAG, DramaticObjectives
from storydag.causopt.optimize import SceneRecord, SceneSequence, optimize
from storydag.cgca.generator import CharacterLine, generate_character_line
from storydag.cgca.history import extract_history_for_scene, scene_nodes_for_character
from storydag.cnge.coref import resolve_coreferences
from storydag.cnge.extractor import extract_novel_triples
from storydag.cnge.graph import CausalGraph
from storydag.cnge.segmentation import segment_novel
from storydag.cnge.types import GraphNode
from storydag.config import get_optional_env, load_env, project_root
from storydag.llm.client import LLMClient
from storydag.serialization.metrics import MetricsReport, compute_metrics, write_metrics_report
from storydag.serialization.schema import ScriptBeat, ScriptCharacterLine, ScriptYAML
from storydag.serialization.yaml_writer import build_script, write_script

Embedder = Callable[[Sequence[str]], np.ndarray]

_CHARACTER_PREFIX = re.compile(r"^([\u4e00-\u9fff]{2})(?=[\u4e00-\u9fff])")
_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class PipelineConfig:
    """Runtime configuration loaded from ``.env`` and environment variables."""

    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    dramatic_objectives: DramaticObjectives = field(default_factory=DramaticObjectives)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "PipelineConfig":
        values = env or load_env()
        return cls(
            embedding_model=get_optional_env(values, "EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            similarity_threshold=float(
                get_optional_env(values, "COREF_SIMILARITY_THRESHOLD", "0.85")
            ),
            mcts=MCTSConfig(
                exploration_constant=float(
                    get_optional_env(values, "MCTS_EXPLORATION_CONSTANT", "0.5")
                ),
                max_iterations=int(get_optional_env(values, "MCTS_ITERATIONS", "10000")),
                time_budget_sec=float(get_optional_env(values, "MCTS_TIME_BUDGET_SEC", "300")),
                max_branching=int(get_optional_env(values, "MCTS_MAX_BRANCHING", "10")),
            ),
            dramatic_objectives=DramaticObjectives(
                max_scenes=int(get_optional_env(values, "CAUSOPT_MAX_SCENES", "50")),
                min_scenes_per_act=int(get_optional_env(values, "CAUSOPT_MIN_SCENES_PER_ACT", "3")),
            ),
        )


@dataclass
class PipelineResult:
    """Artifacts produced by a full pipeline run."""

    title: str
    output_dir: Path
    graph: CausalGraph
    sequence: SceneSequence
    script: ScriptYAML
    metrics: MetricsReport


def sanitize_output_name(title: str) -> str:
    """Convert a human title into a safe directory name."""
    cleaned = _INVALID_PATH_CHARS.sub("_", title.strip())
    return cleaned or "untitled"


def default_output_dir(title: str) -> Path:
    return project_root() / "outputs" / sanitize_output_name(title)


def discover_characters(graph: CausalGraph) -> List[str]:
    """Infer character names from CNGE node labels."""
    names: set[str] = set()
    for node in graph.nodes:
        match = _CHARACTER_PREFIX.match(node.label.strip())
        if match:
            names.add(match.group(1))
    return sorted(names)


def extract_causal_graph(
    novel_text: str,
    client: LLMClient,
    config: PipelineConfig,
    *,
    embedder: Optional[Embedder] = None,
) -> CausalGraph:
    """CNGE: segment novel, extract triples, resolve coreferences, and build a DAG."""
    chapters = segment_novel(novel_text)
    triples = extract_novel_triples(chapters, client)
    nodes, edges = resolve_coreferences(
        triples,
        similarity_threshold=config.similarity_threshold,
        embedder=embedder,
        embedding_model=config.embedding_model,
    )
    return CausalGraph.build(nodes, edges)


def _node_lookup(graph: CausalGraph) -> Dict[str, GraphNode]:
    return {node.node_id: node for node in graph.nodes}


def _scene_setting(graph: CausalGraph, scene_record: SceneRecord) -> str:
    if not scene_record.assigned_node_ids:
        return f"场景 {scene_record.scene_id}"
    node = _node_lookup(graph).get(scene_record.assigned_node_ids[0])
    return node.label if node else f"场景 {scene_record.scene_id}"


def _scene_context(graph: CausalGraph, scene_record: SceneRecord) -> str:
    lookup = _node_lookup(graph)
    labels = [
        lookup[node_id].label
        for node_id in scene_record.assigned_node_ids
        if node_id in lookup
    ]
    return "；".join(labels) if labels else scene_record.scene_id


def generate_character_beats(
    graph: CausalGraph,
    sequence: SceneSequence,
    client: LLMClient,
) -> tuple[Dict[str, str], Dict[str, List[ScriptBeat]]]:
    """CGCA: generate per-scene beats and settings from an optimized sequence."""
    characters = discover_characters(graph)
    settings: Dict[str, str] = {}
    beats_by_scene: Dict[str, List[ScriptBeat]] = {}

    for scene_record in sequence.scenes:
        settings[scene_record.scene_id] = _scene_setting(graph, scene_record)
        context = _scene_context(graph, scene_record)
        lines: List[CharacterLine] = []

        for character in characters:
            if not scene_nodes_for_character(graph, character, scene_record.assigned_node_ids):
                continue
            history = extract_history_for_scene(
                graph,
                character,
                scene_record.scene_id,
                scene_record.assigned_node_ids,
            )
            lines.append(
                generate_character_line(
                    graph,
                    history,
                    context,
                    character,
                    client,
                )
            )

        beats_by_scene[scene_record.scene_id] = [
            ScriptBeat(
                description=context,
                characters=[
                    ScriptCharacterLine(
                        character=line.character,
                        type=line.type,
                        text=line.text,
                        causal_backlink=list(line.causal_backlink),
                    )
                    for line in lines
                ],
            )
        ] if lines else []

    return settings, beats_by_scene


def run_pipeline(
    novel_path: str | Path,
    title: str,
    *,
    output_dir: str | Path | None = None,
    client: Optional[LLMClient] = None,
    config: Optional[PipelineConfig] = None,
    embedder: Optional[Embedder] = None,
) -> PipelineResult:
    """Run novel -> CNGE -> CausOpt -> CGCA -> YAML -> metrics and write artifacts."""
    novel_path = Path(novel_path)
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir(title)
    config = config or PipelineConfig.from_env()
    client = client or LLMClient.from_env()

    novel_text = novel_path.read_text(encoding="utf-8")
    graph = extract_causal_graph(novel_text, client, config, embedder=embedder)

    dag = CausalDAG.from_causal_graph(graph)
    sequence = optimize(dag, config.dramatic_objectives, config=config.mcts)

    settings, beats_by_scene = generate_character_beats(graph, sequence, client)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "causal_graph.json"
    script_path = output_dir / "script.yaml"
    metrics_path = output_dir / "metrics.json"

    graph.save(graph_path)
    script = build_script(
        title,
        sequence,
        settings=settings,
        beats_by_scene=beats_by_scene,
        source_graph=str(graph_path),
    )
    write_script(script, script_path)

    metrics = compute_metrics(graph, script)
    write_metrics_report(metrics, metrics_path)

    return PipelineResult(
        title=title,
        output_dir=output_dir,
        graph=graph,
        sequence=sequence,
        script=script,
        metrics=metrics,
    )
