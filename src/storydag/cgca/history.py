"""Causal history subgraph extraction for CGCA character generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

import networkx as nx

from storydag.cnge.graph import CausalGraph
from storydag.cnge.types import GraphEdge, GraphNode


@dataclass(frozen=True)
class HistoryEdge:
    """A causal edge included in a character's history context."""

    edge_id: str
    source: str
    target: str
    type: str
    source_label: str
    target_label: str


@dataclass
class CausalHistory:
    """Causal history subgraph H(c, S) for a character at a script scene."""

    character: str
    scene_id: str
    scene_node_ids: List[str]
    ancestor_node_ids: List[str]
    knowledge_edges: List[HistoryEdge] = field(default_factory=list)
    motivation_edges: List[HistoryEdge] = field(default_factory=list)

    def to_context_text(self, graph: CausalGraph) -> str:
        """Serialize history as a character-perspective causal context block."""
        node_lookup = {node.node_id: node for node in graph.nodes}
        lines = [
            f"Character: {self.character}",
            f"Scene: {self.scene_id}",
            "",
            "Causal history (topological order):",
        ]

        if not self.ancestor_node_ids:
            lines.append("- (empty)")
        else:
            for index, node_id in enumerate(self.ancestor_node_ids, start=1):
                node = node_lookup[node_id]
                lines.append(f"{index}. [{node.type}] {node.label}")

        lines.extend(["", "Known information:"])
        if not self.knowledge_edges:
            lines.append("- (none)")
        else:
            for edge in self.knowledge_edges:
                lines.append(f"- {edge.source_label} --informs--> {edge.target_label}")

        lines.extend(["", "Active motivations:"])
        if not self.motivation_edges:
            lines.append("- (none)")
        else:
            for edge in self.motivation_edges:
                lines.append(f"- {edge.source_label} --motivates--> {edge.target_label}")

        return "\n".join(lines)


def _node_lookup(graph: CausalGraph) -> Dict[str, GraphNode]:
    return {node.node_id: node for node in graph.nodes}


def _build_nx_graph(graph: CausalGraph) -> nx.DiGraph:
    digraph = nx.DiGraph()
    for node in graph.nodes:
        digraph.add_node(node.node_id)
    for edge in graph.edges:
        digraph.add_edge(edge.source, edge.target)
    return digraph


def node_involves_character(
    node: GraphNode,
    edges: Sequence[GraphEdge],
    node_lookup: Dict[str, GraphNode],
    character: str,
) -> bool:
    """Return whether a node is tied to ``character`` via label or incident edges."""
    if character in node.label:
        return True

    for edge in edges:
        if edge.source != node.node_id and edge.target != node.node_id:
            continue
        other_id = edge.target if edge.source == node.node_id else edge.source
        other = node_lookup[other_id]
        if character in other.label:
            return True
    return False


def scene_nodes_for_character(
    graph: CausalGraph,
    character: str,
    scene_node_ids: Sequence[str],
) -> List[str]:
    """Collect scene node IDs that involve ``character``."""
    node_lookup = _node_lookup(graph)
    selected: List[str] = []
    for node_id in scene_node_ids:
        node = node_lookup[node_id]
        if node_involves_character(node, graph.edges, node_lookup, character):
            selected.append(node_id)
    return selected


def extract_history(
    graph: CausalGraph,
    character: str,
    scene_node_ids: Sequence[str],
    *,
    scene_id: str = "",
) -> CausalHistory:
    """Extract H(c, S): ancestors, knowledge, and motivations for a character."""
    node_lookup = _node_lookup(graph)
    scene_nodes = scene_nodes_for_character(graph, character, scene_node_ids)
    digraph = _build_nx_graph(graph)

    ancestor_ids: Set[str] = set()
    for node_id in scene_nodes:
        ancestor_ids.update(nx.ancestors(digraph, node_id))
    ancestor_ids -= set(scene_nodes)

    topo = graph.topological_order or list(nx.topological_sort(digraph))
    ordered_ancestors = [node_id for node_id in topo if node_id in ancestor_ids]

    history_node_ids = set(ordered_ancestors) | set(scene_nodes)
    knowledge_edges: List[HistoryEdge] = []
    motivation_edges: List[HistoryEdge] = []

    for edge in graph.edges:
        history_edge = HistoryEdge(
            edge_id=edge.edge_id,
            source=edge.source,
            target=edge.target,
            type=edge.type,
            source_label=node_lookup[edge.source].label,
            target_label=node_lookup[edge.target].label,
        )
        target_node = node_lookup[edge.target]
        source_node = node_lookup[edge.source]

        if edge.type == "informs" and node_involves_character(
            target_node, graph.edges, node_lookup, character
        ):
            if edge.source in history_node_ids or edge.target in history_node_ids:
                knowledge_edges.append(history_edge)

        if edge.type == "motivates":
            motivated_for_character = node_involves_character(
                target_node, graph.edges, node_lookup, character
            ) or node_involves_character(source_node, graph.edges, node_lookup, character)
            if motivated_for_character and edge.source in history_node_ids and edge.target in history_node_ids:
                motivation_edges.append(history_edge)

    return CausalHistory(
        character=character,
        scene_id=scene_id,
        scene_node_ids=scene_nodes,
        ancestor_node_ids=ordered_ancestors,
        knowledge_edges=knowledge_edges,
        motivation_edges=motivation_edges,
    )


def extract_history_for_scene(
    graph: CausalGraph,
    character: str,
    scene_id: str,
    assigned_node_ids: Sequence[str],
) -> CausalHistory:
    """Convenience wrapper used by the CGCA generation pipeline."""
    return extract_history(
        graph,
        character,
        assigned_node_ids,
        scene_id=scene_id,
    )
