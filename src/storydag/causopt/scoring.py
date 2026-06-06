"""Dramatic quality scoring for CausOpt scene assignments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from storydag.causopt.models import (
    CausalDAG,
    DramaticObjectives,
    SceneAssignment,
    default_sigmoid_pacing,
    is_valid_assignment,
    node_scene_index,
)

NODE_REVELATION_WEIGHT = {
    "revelation": 4,
    "intention": 2,
    "event": 1,
    "emotional_state": 1,
}

DEFAULT_LAMBDA_ACT = 1.0
DEFAULT_LAMBDA_PACING = 1.5
DEFAULT_LAMBDA_SCENE_LENGTH = 0.5

MIN_SCENE_NODES = 2
MAX_SCENE_NODES = 10


@dataclass(frozen=True)
class ScoreWeights:
    """Composite score weights (algorithm-tuned defaults)."""

    act_structure: float = DEFAULT_LAMBDA_ACT
    pacing: float = DEFAULT_LAMBDA_PACING
    scene_length: float = DEFAULT_LAMBDA_SCENE_LENGTH


def node_revelation_importance(dag: CausalDAG, node_id: str) -> int:
    node_type = dag.nodes[node_id].type
    return NODE_REVELATION_WEIGHT.get(node_type, 1)


def act_structure_match(
    assignment: SceneAssignment,
    objectives: DramaticObjectives,
) -> float:
    """Compare act scene-count distribution to targets using negative KL divergence."""
    num_scenes = len(assignment)
    if num_scenes == 0:
        return 0.0

    proportions = objectives.act_proportions
    num_acts = len(proportions)
    counts = [0] * num_acts
    boundaries = [0.0]
    cumulative = 0.0
    for proportion in proportions:
        cumulative += proportion
        boundaries.append(cumulative)

    for scene_index in range(num_scenes):
        fraction = (scene_index + 1) / num_scenes
        act_index = num_acts - 1
        for idx in range(num_acts):
            if fraction <= boundaries[idx + 1]:
                act_index = idx
                break
        counts[act_index] += 1

    actual = [count / num_scenes for count in counts]
    kl = 0.0
    for target, observed in zip(proportions, actual):
        if target <= 0.0:
            continue
        observed = max(observed, 1e-12)
        kl += target * math.log(target / observed)
    return -kl


def pacing_score(
    dag: CausalDAG,
    assignment: SceneAssignment,
    objectives: DramaticObjectives,
) -> float:
    """Measure revelation-density curve fit against desired sigmoid (negative MSE)."""
    if not assignment:
        return 0.0

    curve = objectives.pacing_curve or default_sigmoid_pacing()
    topo_index = _topological_index(dag)
    ordered_nodes = sorted(dag.node_ids, key=lambda node_id: topo_index[node_id])

    cumulative = 0.0
    max_importance = sum(node_revelation_importance(dag, node_id) for node_id in ordered_nodes)
    if max_importance == 0:
        return 0.0

    node_to_scene = node_scene_index(assignment)
    observed: List[float] = []
    for scene_index in range(len(assignment)):
        scene_nodes = assignment[scene_index]
        cumulative += sum(node_revelation_importance(dag, node_id) for node_id in scene_nodes)
        observed.append(cumulative / max_importance)

    desired = _resample_curve(curve, len(observed))
    mse = sum((left - right) ** 2 for left, right in zip(observed, desired)) / len(observed)
    return -mse


def scene_length_penalty(assignment: SceneAssignment) -> float:
    """Penalize scenes with too few (<2) or too many (>10) nodes."""
    penalty = 0.0
    for scene in assignment:
        size = len(scene)
        if size < MIN_SCENE_NODES:
            penalty += MIN_SCENE_NODES - size
        elif size > MAX_SCENE_NODES:
            penalty += size - MAX_SCENE_NODES
    return -penalty


def evaluate_assignment(
    dag: CausalDAG,
    assignment: SceneAssignment,
    objectives: DramaticObjectives,
    *,
    weights: ScoreWeights | None = None,
) -> float:
    """Composite score: λ1*Act + λ2*Pacing + λ3*SceneLength."""
    if not is_valid_assignment(dag, assignment):
        return float("-inf")

    weights = weights or ScoreWeights()
    act = act_structure_match(assignment, objectives)
    pacing = pacing_score(dag, assignment, objectives)
    length = scene_length_penalty(assignment)
    return (
        weights.act_structure * act
        + weights.pacing * pacing
        + weights.scene_length * length
    )


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


def _resample_curve(curve: Sequence[float], length: int) -> List[float]:
    if length <= 0:
        return []
    if length == 1:
        return [curve[-1]]
    last_index = len(curve) - 1
    return [curve[int(round(index * last_index / (length - 1)))] for index in range(length)]
