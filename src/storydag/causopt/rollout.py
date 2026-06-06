"""Deterministic greedy rollout for CausOpt MCTS simulation."""

from __future__ import annotations

from typing import List, Sequence, Set

from storydag.causopt.mcts import cluster_ready_nodes, dramatic_weight, topological_order
from storydag.causopt.models import CausalDAG, DramaticObjectives, SceneAssignment
from storydag.causopt.scoring import pacing_score


def local_pacing_score(
    dag: CausalDAG,
    partial_assignment: SceneAssignment,
    candidate_scene: Sequence[str],
    objectives: DramaticObjectives,
) -> float:
    """Score one rollout step by marginal pacing improvement."""
    before = pacing_score(dag, partial_assignment, objectives) if partial_assignment else 0.0
    after = pacing_score(dag, [*partial_assignment, list(candidate_scene)], objectives)
    return after - before


def greedy_rollout(
    dag: CausalDAG,
    partial_assignment: SceneAssignment,
    objectives: DramaticObjectives,
) -> SceneAssignment:
    """Complete a partial assignment using deterministic greedy pacing maximization."""
    assignment = [list(scene) for scene in partial_assignment]
    assigned: Set[str] = {node_id for scene in assignment for node_id in scene}
    topo = topological_order(dag)

    while assigned != dag.node_ids:
        ready = dag.ready_nodes(assigned)
        if not ready:
            break

        clusters = cluster_ready_nodes(dag, ready, topo)
        candidates = clusters + [[node_id] for node_id in sorted(ready)]

        best_scene = max(
            candidates,
            key=lambda scene_nodes: (
                local_pacing_score(dag, assignment, scene_nodes, objectives),
                sum(dramatic_weight(dag, node_id) for node_id in scene_nodes),
                -len(scene_nodes),
            ),
        )
        assignment.append(list(best_scene))
        assigned.update(best_scene)

    return assignment
