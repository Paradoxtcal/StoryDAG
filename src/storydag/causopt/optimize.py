"""CausOpt entry point: MCTS search with scored output sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from storydag.causopt.mcts import MCTSConfig, search
from storydag.causopt.models import CausalDAG, DramaticObjectives, SceneAssignment, node_scene_index
from storydag.causopt.scoring import evaluate_assignment


@dataclass
class SceneRecord:
    """One optimized script scene with act and narrative-time metadata."""

    scene_id: str
    act: int
    assigned_node_ids: List[str]
    narrative_time_clue: Dict[str, int]


@dataclass
class SceneSequence:
    """Optimized scene ordering output from CausOpt."""

    scenes: List[SceneRecord] = field(default_factory=list)
    satisfied_edges: Dict[str, int] = field(default_factory=dict)
    score: float = 0.0


def _assign_act(scene_index: int, total_scenes: int, proportions: List[float]) -> int:
    if total_scenes == 0:
        return 1
    fraction = (scene_index + 1) / total_scenes
    cumulative = 0.0
    for act_index, proportion in enumerate(proportions, start=1):
        cumulative += proportion
        if fraction <= cumulative:
            return act_index
    return len(proportions)


def _narrative_time_clue(
    dag: CausalDAG,
    node_ids: List[str],
    topo_index: Dict[str, int],
) -> Dict[str, int]:
    if not node_ids:
        return {"earliest": 0, "latest": 0}
    indices = [topo_index[node_id] for node_id in node_ids]
    return {"earliest": min(indices), "latest": max(indices)}


def _topological_index(dag: CausalDAG) -> Dict[str, int]:
    indegree = {node_id: len(dag.predecessors.get(node_id, set())) for node_id in dag.node_ids}
    queue = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: Dict[str, int] = {}
    position = 0

    while queue:
        current = queue.pop(0)
        order[current] = position
        position += 1
        for edge in dag.edges:
            if edge.source != current:
                continue
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                queue.append(edge.target)
        queue.sort()
    return order


def build_scene_sequence(
    dag: CausalDAG,
    assignment: SceneAssignment,
    objectives: DramaticObjectives,
) -> SceneSequence:
    """Convert a scene assignment into structured output with edge satisfaction."""
    topo_index = _topological_index(dag)
    node_to_scene = node_scene_index(assignment)
    total_scenes = len(assignment)

    scenes: List[SceneRecord] = []
    for index, node_ids in enumerate(assignment):
        scenes.append(
            SceneRecord(
                scene_id=f"S{index + 1:02d}",
                act=_assign_act(index, total_scenes, objectives.act_proportions),
                assigned_node_ids=list(node_ids),
                narrative_time_clue=_narrative_time_clue(dag, node_ids, topo_index),
            )
        )

    satisfied_edges: Dict[str, int] = {}
    for edge in dag.edges:
        if edge.edge_id:
            satisfied_edges[edge.edge_id] = node_to_scene[edge.target]

    score = evaluate_assignment(dag, assignment, objectives)
    return SceneSequence(scenes=scenes, satisfied_edges=satisfied_edges, score=score)


def optimize(
    dag: CausalDAG,
    objectives: Optional[DramaticObjectives] = None,
    *,
    config: Optional[MCTSConfig] = None,
) -> SceneSequence:
    """Run MCTS-HP and return the best scored scene sequence."""
    objectives = objectives or DramaticObjectives()
    assignment = search(dag, objectives, config=config)
    return build_scene_sequence(dag, assignment, objectives)
