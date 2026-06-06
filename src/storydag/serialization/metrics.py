"""Automatic causal closure and consistency metrics for script YAML."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from storydag.cnge.graph import CausalGraph
from storydag.serialization.schema import ScriptCharacterLine, ScriptScene, ScriptYAML
from storydag.serialization.yaml_reader import read_script


@dataclass
class BacklinkViolation:
    """A causal_backlink reference that fails topological validation."""

    scene_id: str
    character: str
    edge_id: str
    reason: str


@dataclass
class MetricsReport:
    """Evaluation report for a script against its source causal graph."""

    title: str
    ccr: float
    total_edges: int
    satisfied_edge_count: int
    causal_density: Dict[str, int] = field(default_factory=dict)
    character_consistency: Dict[str, bool] = field(default_factory=dict)
    plot_holes: List[str] = field(default_factory=list)
    out_of_order_edges: List[str] = field(default_factory=list)
    backlink_violations: List[BacklinkViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["backlink_violations"] = [asdict(item) for item in self.backlink_violations]
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def iter_scenes(script: ScriptYAML) -> Iterable[ScriptScene]:
    for act in script.acts:
        for scene in act.scenes:
            yield scene


def iter_character_lines(script: ScriptYAML) -> Iterable[Tuple[ScriptScene, ScriptCharacterLine]]:
    for scene in iter_scenes(script):
        for beat in scene.beats:
            for line in beat.characters:
                yield scene, line


def _build_nx_graph(graph: CausalGraph) -> nx.DiGraph:
    digraph = nx.DiGraph()
    for node in graph.nodes:
        digraph.add_node(node.node_id)
    for edge in graph.edges:
        digraph.add_edge(edge.source, edge.target)
    return digraph


def _edge_lookup(graph: CausalGraph) -> Dict[str, object]:
    return {edge.edge_id: edge for edge in graph.edges if edge.edge_id}


def _scene_order(script: ScriptYAML) -> Dict[str, int]:
    order: Dict[str, int] = {}
    for index, scene in enumerate(iter_scenes(script)):
        order[scene.scene_id] = index
    return order


def _edge_satisfaction_map(script: ScriptYAML) -> Dict[str, int]:
    """Map edge_id -> scene index where it is marked satisfied."""
    mapping: Dict[str, int] = {}
    for index, scene in enumerate(iter_scenes(script)):
        for edge_id in scene.satisfied_edges:
            mapping[edge_id] = index
    return mapping


def compute_ccr(graph: CausalGraph, script: ScriptYAML) -> Tuple[float, int, int]:
    """Return ``(ccr, satisfied_count, total_edges)``."""
    total = len(graph.edges)
    if total == 0:
        return 1.0, 0, 0

    satisfied_ids = _satisfied_edge_ids(graph, script)
    satisfied_count = len(satisfied_ids)
    return satisfied_count / total, satisfied_count, total


def _satisfied_edge_ids(graph: CausalGraph, script: ScriptYAML) -> Set[str]:
    scene_order = _scene_order(script)
    node_scene: Dict[str, int] = {}
    for scene_index, scene in enumerate(iter_scenes(script)):
        for node_id in scene.assigned_node_ids:
            node_scene[node_id] = scene_index

    satisfied: Set[str] = set()
    for edge in graph.edges:
        if not edge.edge_id:
            continue
        target_scene = node_scene.get(edge.target)
        source_scene = node_scene.get(edge.source)
        if target_scene is None:
            continue
        if source_scene is not None and source_scene >= target_scene:
            continue
        satisfied.add(edge.edge_id)
    return satisfied


def compute_causal_density(script: ScriptYAML) -> Dict[str, int]:
    """Return satisfied-edge counts per scene."""
    return {
        scene.scene_id: len(scene.satisfied_edges)
        for scene in iter_scenes(script)
    }


def detect_plot_holes(graph: CausalGraph, script: ScriptYAML) -> Tuple[List[str], List[str]]:
    """Detect unsatisfied edges and edges satisfied out of causal order."""
    node_scene: Dict[str, int] = {}
    for scene_index, scene in enumerate(iter_scenes(script)):
        for node_id in scene.assigned_node_ids:
            node_scene[node_id] = scene_index

    plot_holes: List[str] = []
    out_of_order: List[str] = []

    for edge in graph.edges:
        if not edge.edge_id:
            continue
        target_scene = node_scene.get(edge.target)
        source_scene = node_scene.get(edge.source)
        if target_scene is None:
            plot_holes.append(edge.edge_id)
            continue
        if source_scene is not None and source_scene >= target_scene:
            out_of_order.append(edge.edge_id)

    return sorted(set(plot_holes)), sorted(set(out_of_order))


def _character_edge_ids(graph: CausalGraph, character: str) -> List[str]:
    edge_ids: List[str] = []
    for edge in graph.edges:
        if not edge.edge_id:
            continue
        source = next(node for node in graph.nodes if node.node_id == edge.source)
        target = next(node for node in graph.nodes if node.node_id == edge.target)
        if character in source.label or character in target.label:
            edge_ids.append(edge.edge_id)
    return edge_ids


def compute_character_consistency(graph: CausalGraph, script: ScriptYAML) -> Dict[str, bool]:
    """Check whether each character's edge satisfaction order respects DAG topology."""
    digraph = _build_nx_graph(graph)
    topo = graph.topological_order or list(nx.topological_sort(digraph))
    topo_index = {node_id: index for index, node_id in enumerate(topo)}
    satisfaction = _edge_satisfaction_map(script)
    edges = _edge_lookup(graph)

    characters = {
        line.character
        for _, line in iter_character_lines(script)
    }

    results: Dict[str, bool] = {}
    for character in sorted(characters):
        consistent = True
        char_edges = [edges[eid] for eid in _character_edge_ids(graph, character) if eid in satisfaction]
        ordered = sorted(
            char_edges,
            key=lambda edge: satisfaction.get(edge.edge_id, -1),
        )
        for left, right in zip(ordered, ordered[1:]):
            if topo_index[left.target] > topo_index[right.target]:
                consistent = False
                break
        results[character] = consistent
    return results


def verify_backlinks(graph: CausalGraph, script: ScriptYAML) -> List[BacklinkViolation]:
    """Validate per-line ``causal_backlink`` references against graph topology."""
    edges = _edge_lookup(graph)
    digraph = _build_nx_graph(graph)
    topo = graph.topological_order or list(nx.topological_sort(digraph))
    topo_index = {node_id: index for index, node_id in enumerate(topo)}
    scene_order = _scene_order(script)
    violations: List[BacklinkViolation] = []

    for scene in iter_scenes(script):
        scene_index = scene_order[scene.scene_id]
        for beat in scene.beats:
            for line in beat.characters:
                seen_topo = -1
                for edge_id in line.causal_backlink:
                    edge = edges.get(edge_id)
                    if edge is None:
                        violations.append(
                            BacklinkViolation(
                                scene_id=scene.scene_id,
                                character=line.character,
                                edge_id=edge_id,
                                reason="unknown edge id",
                            )
                        )
                        continue

                    edge_topo = topo_index[edge.target]
                    if edge_topo < seen_topo:
                        violations.append(
                            BacklinkViolation(
                                scene_id=scene.scene_id,
                                character=line.character,
                                edge_id=edge_id,
                                reason="backlinks not topologically ordered",
                            )
                        )
                    seen_topo = max(seen_topo, edge_topo)

                    target_scene_indices = [
                        index
                        for index, candidate in enumerate(iter_scenes(script))
                        if edge.target in candidate.assigned_node_ids
                    ]
                    if target_scene_indices and scene_index < max(target_scene_indices):
                        violations.append(
                            BacklinkViolation(
                                scene_id=scene.scene_id,
                                character=line.character,
                                edge_id=edge_id,
                                reason="backlink references future edge satisfaction",
                            )
                        )
    return violations


def compute_metrics(graph: CausalGraph, script: ScriptYAML) -> MetricsReport:
    """Compute the full metrics report for a script/graph pair."""
    ccr, satisfied_count, total_edges = compute_ccr(graph, script)
    plot_holes, out_of_order = detect_plot_holes(graph, script)
    return MetricsReport(
        title=script.title,
        ccr=ccr,
        total_edges=total_edges,
        satisfied_edge_count=satisfied_count,
        causal_density=compute_causal_density(script),
        character_consistency=compute_character_consistency(graph, script),
        plot_holes=plot_holes,
        out_of_order_edges=out_of_order,
        backlink_violations=verify_backlinks(graph, script),
    )


def compute_metrics_from_files(script_path: str | Path, graph_path: str | Path) -> MetricsReport:
    """Load artifacts from disk and compute metrics."""
    script = read_script(script_path)
    graph = CausalGraph.load(graph_path)
    return compute_metrics(graph, script)


def write_metrics_report(report: MetricsReport, path: str | Path) -> None:
    Path(path).write_text(report.to_json(), encoding="utf-8")
