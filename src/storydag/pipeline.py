"""End-to-end StoryDAG pipeline orchestration."""

from __future__ import annotations

import re
import sys
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

_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ── Print helpers ─────────────────────────────────────────────────


def _print(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def _print_header(title: str) -> None:
    print(file=sys.stderr)
    print(f"  ── {title} ", file=sys.stderr, end="")
    print("─" * max(0, 60 - len(title) - 6), file=sys.stderr, flush=True)


def _extract_progress(done: int, total: int) -> None:
    pct = done * 100 // total if total else 100
    bar_len = 24
    filled = done * bar_len // total if total else bar_len
    bar = "█" * filled + "░" * (bar_len - filled)
    total_str = str(total)
    done_pad = str(done).rjust(len(total_str))
    # \r to overwrite
    print(
        f"    场景 {done_pad}/{total_str}  {bar}  {pct}%",
        file=sys.stderr,
        end="\r" if done < total else "\n",
        flush=True,
    )


# ── Config ────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Runtime configuration loaded from ``.env`` and environment variables."""

    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85
    characters: Optional[List[str]] = None
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    dramatic_objectives: DramaticObjectives = field(default_factory=DramaticObjectives)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "PipelineConfig":
        values = env or load_env()
        raw_chars = get_optional_env(values, "CHARACTER_LIST", "")
        characters = (
            [c.strip() for c in raw_chars.split(",") if c.strip()]
            if raw_chars
            else None
        )
        return cls(
            embedding_model=get_optional_env(values, "EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            similarity_threshold=float(
                get_optional_env(values, "COREF_SIMILARITY_THRESHOLD", "0.85")
            ),
            characters=characters,
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


# ── Helpers ──────────────────────────────────────────────────────


def sanitize_output_name(title: str) -> str:
    """Convert a human title into a safe directory name."""
    cleaned = _INVALID_PATH_CHARS.sub("_", title.strip())
    return cleaned or "untitled"


def default_output_dir(title: str) -> Path:
    return project_root() / "outputs" / sanitize_output_name(title)


def discover_characters(
    graph: CausalGraph,
    explicit: Optional[Sequence[str]] = None,
    client: Optional[LLMClient] = None,
) -> List[str]:
    """Infer character names from CNGE node labels.

    Resolution order:
      1. ``explicit`` list → return directly
      2. ``client`` provided → ask LLM to extract names (NER via LLM)
      3. Fallback → data-driven longest-common-prefix across labels
    """
    if explicit:
        return sorted(explicit)

    labels = [node.label.strip() for node in graph.nodes if node.label.strip()]

    # ── LLM-based NER: ask the model directly ──────────────────
    if client is not None:
        try:
            prompt_lines = "\n".join(f"{i+1}. {label}" for i, label in enumerate(labels))
            response = client.complete(
                prompt=(
                    "从以下因果图节点标签中提取所有角色（人物）名。\n"
                    "要求：\n"
                    "- 只返回角色名，不要返回动作、情绪或事件描述\n"
                    "- 返回格式为 JSON 数组，如 [\"张三\", \"李四\"]\n"
                    "- 若某标签不含角色名则跳过\n"
                    "- 去重\n\n"
                    "节点标签：\n"
                    f"{prompt_lines}"
                ),
                json_mode=True,
                temperature=0.0,
            )
            import json as _json
            names = _json.loads(response)
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                return sorted(set(names))
        except Exception:
            pass  # fall through to LCP fallback

    # ── Fallback: longest-common-prefix across label groups ────
    groups: Dict[str, List[str]] = {}
    for label in labels:
        if len(label) < 2:
            continue
        first_two = label[:2]
        groups.setdefault(first_two, []).append(label)

    names: set[str] = set()
    for first_two, group_labels in groups.items():
        lcp = group_labels[0]
        for label in group_labels[1:]:
            while not label.startswith(lcp):
                lcp = lcp[:-1]
        name: str = first_two
        for length in range(min(4, len(lcp)), 1, -1):
            candidate = lcp[:length]
            matching = sum(1 for l in labels if l.startswith(candidate))
            if matching >= 2 or length <= 2:
                name = candidate
                break
        names.add(name)

    return sorted(names)


# ── CNGE ─────────────────────────────────────────────────────────


def extract_causal_graph(
    novel_text: str,
    client: LLMClient,
    config: PipelineConfig,
    *,
    embedder: Optional[Embedder] = None,
    env: Optional[Dict[str, str]] = None,
    quiet: bool = False,
) -> CausalGraph:
    """CNGE: segment novel, extract triples, resolve coreferences, and build a DAG."""
    if not quiet:
        _print_header("CNGE · 因果叙事图抽取")

    if not quiet:
        _print("分段中……")
    chapters = segment_novel(novel_text)
    total_scenes = sum(len(c.scenes) for c in chapters)
    if not quiet:
        _print(f"共 {total_scenes} 个场景，开始 LLM 因果抽取……")

    triples = extract_novel_triples(
        chapters,
        client,
        progress_callback=_extract_progress if not quiet else None,
    )
    if not quiet:
        _print(f"抽取完成，共 {len(triples)} 个因果三元组")

    if not quiet:
        _print("共指消解中……")

    nodes, edges = resolve_coreferences(
        triples,
        similarity_threshold=config.similarity_threshold,
        embedder=embedder,
        embedding_model=config.embedding_model,
        env=env,
    )
    if not quiet:
        _print(f"共指消解完成：{len(nodes)} 节点, {len(edges)} 边")

    graph = CausalGraph.build(nodes, edges)
    if not quiet:
        _print(f"DAG 构建完成：{len(graph.nodes)} 节点, {len(graph.edges)} 边, "
               f"移除 {len(graph.removed_cycle_edges)} 条环边")
    return graph


# ── CGCA ────────────────────────────────────────────────────────


def _node_lookup(graph: CausalGraph) -> Dict[str, GraphNode]:
    return {node.node_id: node for node in graph.nodes}


def _satisfied_edges_for_scene(
    sequence: SceneSequence, scene_id: str,
) -> List[str]:
    """Return edge IDs satisfied (i.e. settled) in a given scene."""
    return sorted(
        edge_id
        for edge_id, scene_index in sequence.satisfied_edges.items()
        if sequence.scenes[scene_index].scene_id == scene_id
    )


def _llm_generate_scene(
    graph: CausalGraph,
    scene_record: SceneRecord,
    sequence: SceneSequence,
    client: LLMClient,
) -> tuple[str, str]:
    """LLM 生成场景设置（setting）与叙事描述（beat description）。

    Returns
    -------
    (setting, description)
    """
    lookup = _node_lookup(graph)
    node_lines = "\n".join(
        f"- {nid}: {lookup[nid].label}"
        for nid in scene_record.assigned_node_ids
        if nid in lookup
    ) or "(无)"
    edge_ids = _satisfied_edges_for_scene(sequence, scene_record.scene_id)
    edge_lines = "\n".join(
        f"- {eid}  {lookup[e.source].label} → {lookup[e.target].label}"
        for eid in edge_ids
        for e in graph.edges
        if e.edge_id == eid
    ) or "(无)"

    prompt = (
        f"场景 {scene_record.scene_id}（第 {scene_record.act} 幕）的因果节点如下：\n"
        f"{node_lines}\n\n"
        f"本场景需要完成的因果边：\n"
        f"{edge_lines}\n\n"
        "请以 JSON 格式返回两项内容：\n"
        '1. "setting"：该场景的标题（15 字以内）\n'
        '2. "description"：该场景的叙事描述（50-100 字，描述发生了什么、角色做了什么、情绪基调）\n\n'
        '示例格式：{"setting": "…", "description": "…"}'
    )

    try:
        import json as _json
        response = client.complete(prompt, json_mode=True, temperature=0.7)
        data = _json.loads(response)
        setting = data.get("setting", "") or ""
        description = data.get("description", "") or ""
        if setting and description:
            return setting, description
    except Exception:
        pass

    # Fallback: first node label as setting, joined labels as description
    first_node = lookup.get(scene_record.assigned_node_ids[0]) if scene_record.assigned_node_ids else None
    setting = first_node.label if first_node else f"场景 {scene_record.scene_id}"
    desc_labels = [
        lookup[nid].label
        for nid in scene_record.assigned_node_ids
        if nid in lookup
    ]
    description = "；".join(desc_labels) if desc_labels else f"场景 {scene_record.scene_id}"
    return setting, description


def generate_character_beats(
    graph: CausalGraph,
    sequence: SceneSequence,
    client: LLMClient,
    *,
    characters: Optional[Sequence[str]] = None,
    quiet: bool = False,
) -> tuple[Dict[str, str], Dict[str, List[ScriptBeat]]]:
    """CGCA: generate per-scene beats and settings from an optimized sequence."""
    character_names = discover_characters(graph, explicit=characters, client=client)

    if not quiet:
        _print(f"发现角色: {', '.join(character_names)}")
        _print_header("CGCA · 因果门控角色台词生成")

    settings: Dict[str, str] = {}
    beats_by_scene: Dict[str, List[ScriptBeat]] = {}

    for scene_idx, scene_record in enumerate(sequence.scenes, start=1):
        if not quiet:
            _print(f"  场景 {scene_idx}/{len(sequence.scenes)}  {scene_record.scene_id}")

        # LLM generates scene setting + narrative description
        setting, description = _llm_generate_scene(graph, scene_record, sequence, client)
        settings[scene_record.scene_id] = setting

        lines: List[CharacterLine] = []
        for character in character_names:
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
                    setting,          # 传入 LLM 生成的 setting
                    description,      # 传入 LLM 生成的叙事描述作为场景上下文
                    character,
                    client,
                )
            )

        beats_by_scene[scene_record.scene_id] = [
            ScriptBeat(
                description=description,
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


# ── Full pipeline ────────────────────────────────────────────────


def run_pipeline(
    novel_path: str | Path,
    title: str,
    *,
    output_dir: str | Path | None = None,
    client: Optional[LLMClient] = None,
    config: Optional[PipelineConfig] = None,
    embedder: Optional[Embedder] = None,
    env: Optional[Dict[str, str]] = None,
    quiet: bool = False,
) -> PipelineResult:
    """Run novel -> CNGE -> CausOpt -> CGCA -> YAML -> metrics and write artifacts."""
    if not quiet:
        print(file=sys.stderr)
        print(f"  ╔═══════════════════════════════════════════════", file=sys.stderr)
        print(f"  ║  StoryDAG Pipeline  —  {title}", file=sys.stderr)
        print(f"  ╚═══════════════════════════════════════════════", file=sys.stderr)

    novel_path = Path(novel_path)
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir(title)
    resolved_env = env if env is not None else load_env()
    config = config or PipelineConfig.from_env(resolved_env)
    client = client or LLMClient.from_env(resolved_env)

    if not quiet:
        _print(f"读取小说: {novel_path}")

    novel_text = novel_path.read_text(encoding="utf-8")

    # ── Step 1: CNGE ──────────────────────────────────────────
    graph = extract_causal_graph(novel_text, client, config, embedder=embedder, env=resolved_env, quiet=quiet)

    # ── Step 2: CausOpt ───────────────────────────────────────
    if not quiet:
        _print_header("CausOpt · 硬约束场景排序")
        _print(f"MCTS 搜索中（最多 {config.mcts.max_iterations} 轮 / {config.mcts.time_budget_sec}s）……")

    dag = CausalDAG.from_causal_graph(graph)
    sequence = optimize(dag, config.dramatic_objectives, config=config.mcts)

    if not quiet:
        _print(f"排序完成：{len(sequence.scenes)} 个场景, CCR 基线 {len(sequence.satisfied_edges)}/{len(graph.edges)}")

    # ── Step 3: CGCA ──────────────────────────────────────────
    settings, beats_by_scene = generate_character_beats(
        graph, sequence, client,
        characters=config.characters,
        quiet=quiet,
    )

    # ── Step 4: Serialization ─────────────────────────────────
    if not quiet:
        _print_header("序列化 · YAML + Metrics")

    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "causal_graph.json"
    script_path = output_dir / "script.yaml"
    metrics_path = output_dir / "metrics.json"

    graph.save(graph_path)
    if not quiet:
        _print(f"因果图 → {graph_path}")

    script = build_script(
        title,
        sequence,
        settings=settings,
        beats_by_scene=beats_by_scene,
        source_graph=str(graph_path),
    )
    write_script(script, script_path)
    if not quiet:
        _print(f"剧本   → {script_path}")

    metrics = compute_metrics(graph, script)
    write_metrics_report(metrics, metrics_path)
    if not quiet:
        _print(f"指标   → {metrics_path}")
        _print(f"CCR    = {metrics.ccr:.2%}  ({metrics.satisfied_edge_count}/{metrics.total_edges})")
        print(file=sys.stderr)

    return PipelineResult(
        title=title,
        output_dir=output_dir,
        graph=graph,
        sequence=sequence,
        script=script,
        metrics=metrics,
    )
