"""DAG construction, cycle handling, and serialization for CNGE."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx

from storydag.cnge.types import GraphEdge, GraphNode


@dataclass
class RemovedCycleEdge:
    """An edge removed to break a cycle, flagged for optional human review."""

    edge_id: str
    source: str
    target: str
    type: str
    strength: float


@dataclass
class CausalGraph:
    """A directed acyclic causal narrative graph."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    topological_order: List[str] = field(default_factory=list)
    removed_cycle_edges: List[RemovedCycleEdge] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        *,
        break_cycles: bool = True,
    ) -> "CausalGraph":
        """Build a DAG from resolved nodes and edges."""
        return build_dag(nodes, edges, break_cycles=break_cycles)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "label": node.label,
                    "type": node.type,
                    "source_ids": list(node.source_ids),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "strength": edge.strength,
                }
                for edge in self.edges
            ],
            "topological_order": list(self.topological_order),
            "removed_cycle_edges": [
                {
                    "edge_id": item.edge_id,
                    "source": item.source,
                    "target": item.target,
                    "type": item.type,
                    "strength": item.strength,
                }
                for item in self.removed_cycle_edges
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
        nodes = [
            GraphNode(
                node_id=str(item["node_id"]),
                label=str(item["label"]),
                type=str(item["type"]),
                source_ids=list(item.get("source_ids", [])),
            )
            for item in data.get("nodes", [])
        ]
        edges = [
            GraphEdge(
                edge_id=str(item["edge_id"]),
                source=str(item["source"]),
                target=str(item["target"]),
                type=str(item["type"]),
                strength=float(item.get("strength", 1.0)),
            )
            for item in data.get("edges", [])
        ]
        return cls(
            nodes=nodes,
            edges=edges,
            topological_order=list(data.get("topological_order", [])),
            removed_cycle_edges=[
                RemovedCycleEdge(
                    edge_id=str(item["edge_id"]),
                    source=str(item["source"]),
                    target=str(item["target"]),
                    type=str(item.get("type", "")),
                    strength=float(item.get("strength", 1.0)),
                )
                for item in data.get("removed_cycle_edges", [])
            ],
        )

    @classmethod
    def from_json(cls, text: str) -> "CausalGraph":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CausalGraph":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _build_nx_graph(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node.node_id, label=node.label, type=node.type, source_ids=node.source_ids)
    for edge in edges:
        graph.add_edge(
            edge.source,
            edge.target,
            edge_id=edge.edge_id,
            type=edge.type,
            strength=edge.strength,
        )
    return graph


def _remove_cycle_edges(edges: List[GraphEdge]) -> tuple[List[GraphEdge], List[RemovedCycleEdge]]:
    """Iteratively remove the lowest-strength edge until the graph is acyclic."""
    working = list(edges)
    removed: List[RemovedCycleEdge] = []

    while True:
        graph = _build_nx_graph([], working)
        if nx.is_directed_acyclic_graph(graph):
            return working, removed

        cycle_edges = _edges_on_cycles(graph)
        if not cycle_edges:
            return working, removed

        weakest = min(
            cycle_edges,
            key=lambda edge: (edge.strength, edge.edge_id),
        )
        working = [edge for edge in working if edge.edge_id != weakest.edge_id]
        removed.append(
            RemovedCycleEdge(
                edge_id=weakest.edge_id,
                source=weakest.source,
                target=weakest.target,
                type=weakest.type,
                strength=weakest.strength,
            )
        )


def _edges_on_cycles(graph: nx.DiGraph) -> List[GraphEdge]:
    cycle_edge_ids = set()
    try:
        for cycle in nx.simple_cycles(graph):
            for index in range(len(cycle)):
                source = cycle[index]
                target = cycle[(index + 1) % len(cycle)]
                edge_data = graph.get_edge_data(source, target) or {}
                edge_id = edge_data.get("edge_id")
                if edge_id:
                    cycle_edge_ids.add(edge_id)
    except nx.NetworkXNoCycle:
        return []

    edge_lookup = {}
    for source, target, data in graph.edges(data=True):
        edge_id = data.get("edge_id")
        if edge_id in cycle_edge_ids:
            edge_lookup[edge_id] = GraphEdge(
                edge_id=edge_id,
                source=source,
                target=target,
                type=data.get("type", ""),
                strength=float(data.get("strength", 1.0)),
            )
    return list(edge_lookup.values())


def build_dag(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    break_cycles: bool = True,
) -> CausalGraph:
    """Construct a causal DAG, optionally breaking cycles by removing weak edges."""
    node_list = list(nodes)
    edge_list = list(edges)
    removed: List[RemovedCycleEdge] = []

    if break_cycles and edge_list:
        edge_list, removed = _remove_cycle_edges(edge_list)

    graph = _build_nx_graph(node_list, edge_list)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph still contains cycles after cycle removal")

    topo_order = list(nx.topological_sort(graph)) if graph.number_of_nodes() else []
    return CausalGraph(
        nodes=node_list,
        edges=edge_list,
        topological_order=topo_order,
        removed_cycle_edges=removed,
    )
