"""Data structures and hard constraints for CausOpt scene assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from storydag.cnge.graph import CausalGraph

SceneAssignment = List[List[str]]


@dataclass(frozen=True)
class DAGNode:
    """A node in the causal DAG consumed by CausOpt."""

    node_id: str
    label: str
    type: str


@dataclass(frozen=True)
class DAGEdge:
    """A directed causal edge between two DAG nodes."""

    source: str
    target: str
    type: str
    edge_id: Optional[str] = None
    strength: float = 1.0


@dataclass
class DramaticObjectives:
    """Dramatic quality targets for MCTS scoring (PR #8–#9)."""

    act_proportions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.25])
    max_scenes: int = 50
    min_scenes_per_act: int = 3
    pacing_curve: Optional[List[float]] = None

    def __post_init__(self) -> None:
        total = sum(self.act_proportions)
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(f"act_proportions must sum to 1.0, got {total}")
        if self.max_scenes < 1:
            raise ValueError("max_scenes must be >= 1")
        if self.min_scenes_per_act < 1:
            raise ValueError("min_scenes_per_act must be >= 1")


@dataclass
class PartialSequence:
    """A partial scene assignment state used by MCTS search."""

    scenes: List[List[str]]
    assigned_nodes: Set[str] = field(default_factory=set)
    unassigned_nodes: Set[str] = field(default_factory=set)
    blocked_nodes: Set[str] = field(default_factory=set)

    @property
    def is_complete(self) -> bool:
        return not self.unassigned_nodes


@dataclass
class CausalDAG:
    """CausOpt view of a CNGE causal graph with predecessor lookup."""

    nodes: Dict[str, DAGNode]
    edges: List[DAGEdge]
    predecessors: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def from_causal_graph(cls, graph: CausalGraph) -> "CausalDAG":
        nodes = {
            node.node_id: DAGNode(node_id=node.node_id, label=node.label, type=node.type)
            for node in graph.nodes
        }
        edges = [
            DAGEdge(
                source=edge.source,
                target=edge.target,
                type=edge.type,
                edge_id=edge.edge_id,
                strength=edge.strength,
            )
            for edge in graph.edges
        ]
        predecessors: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}
        for edge in edges:
            predecessors.setdefault(edge.target, set()).add(edge.source)
        return cls(nodes=nodes, edges=edges, predecessors=predecessors)

    @property
    def node_ids(self) -> Set[str]:
        return set(self.nodes.keys())

    def ready_nodes(self, assigned: Set[str]) -> Set[str]:
        """Return unassigned nodes whose causal antecedents are all assigned."""
        ready: Set[str] = set()
        for node_id in self.node_ids - assigned:
            preds = self.predecessors.get(node_id, set())
            if preds.issubset(assigned):
                ready.add(node_id)
        return ready

    def blocked_nodes(self, assigned: Set[str]) -> Set[str]:
        """Return unassigned nodes that cannot yet be placed."""
        unassigned = self.node_ids - assigned
        return unassigned - self.ready_nodes(assigned)


def node_scene_index(assignment: SceneAssignment) -> Dict[str, int]:
    """Map each node ID to its scene index in the assignment."""
    mapping: Dict[str, int] = {}
    for scene_index, scene_nodes in enumerate(assignment):
        for node_id in scene_nodes:
            mapping[node_id] = scene_index
    return mapping


def is_valid_assignment(dag: CausalDAG, assignment: SceneAssignment) -> bool:
    """Check hard causal precedence: scene(u) < scene(v) for every edge u→v."""
    assigned_nodes: List[str] = []
    for scene in assignment:
        assigned_nodes.extend(scene)

    if set(assigned_nodes) != dag.node_ids:
        return False
    if len(assigned_nodes) != len(set(assigned_nodes)):
        return False

    index_map = node_scene_index(assignment)
    for edge in dag.edges:
        if index_map[edge.source] >= index_map[edge.target]:
            return False
    return True


def build_partial_sequence(dag: CausalDAG, assignment: SceneAssignment) -> PartialSequence:
    """Build a ``PartialSequence`` snapshot from a (possibly partial) assignment."""
    assigned: Set[str] = set()
    scenes: List[List[str]] = []
    for scene in assignment:
        scenes.append(list(scene))
        assigned.update(scene)

    unassigned = dag.node_ids - assigned
    return PartialSequence(
        scenes=scenes,
        assigned_nodes=assigned,
        unassigned_nodes=unassigned,
        blocked_nodes=dag.blocked_nodes(assigned),
    )


def default_sigmoid_pacing(num_points: int = 100) -> List[float]:
    """Generate a default sigmoid pacing curve for ``DramaticObjectives``."""
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    midpoint = (num_points - 1) / 2.0
    return [1.0 / (1.0 + math.exp(-0.1 * (index - midpoint))) for index in range(num_points)]
