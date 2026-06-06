"""Tests for CausOpt dramatic quality scoring and optimization output."""

from storydag.causopt import (
    CausalDAG,
    DramaticObjectives,
    MCTSConfig,
    SceneSequence,
    build_scene_sequence,
    evaluate_assignment,
    optimize,
)
from storydag.causopt.scoring import act_structure_match, pacing_score, scene_length_penalty
from storydag.cnge.graph import build_dag
from storydag.cnge.types import GraphEdge, GraphNode


def _chain_dag() -> CausalDAG:
    graph = build_dag(
        [
            GraphNode("n1", "甲", "revelation"),
            GraphNode("n2", "乙", "intention"),
            GraphNode("n3", "丙", "event"),
        ],
        [
            GraphEdge("e1", "n1", "n2", "motivates"),
            GraphEdge("e2", "n2", "n3", "motivates"),
        ],
    )
    return CausalDAG.from_causal_graph(graph)


def test_scene_length_penalty_penalizes_tiny_and_huge_scenes():
    penalty = scene_length_penalty([["n1"], ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11"]])
    assert penalty < 0


def test_evaluate_assignment_returns_finite_score_for_valid_assignment():
    dag = _chain_dag()
    assignment = [["n1"], ["n2"], ["n3"]]
    score = evaluate_assignment(dag, assignment, DramaticObjectives())
    assert score > float("-inf")


def test_evaluate_assignment_invalid_returns_negative_inf():
    dag = _chain_dag()
    score = evaluate_assignment(dag, [["n2"], ["n1"], ["n3"]], DramaticObjectives())
    assert score == float("-inf")


def test_act_structure_and_pacing_components():
    dag = _chain_dag()
    assignment = [["n1"], ["n2"], ["n3"]]
    objectives = DramaticObjectives()
    assert act_structure_match(assignment, objectives) <= 0.0
    assert pacing_score(dag, assignment, objectives) <= 0.0


def test_build_scene_sequence_metadata():
    dag = _chain_dag()
    assignment = [["n1"], ["n2"], ["n3"]]
    sequence = build_scene_sequence(dag, assignment, DramaticObjectives())

    assert isinstance(sequence, SceneSequence)
    assert [scene.scene_id for scene in sequence.scenes] == ["S01", "S02", "S03"]
    assert sequence.scenes[0].assigned_node_ids == ["n1"]
    assert sequence.scenes[0].narrative_time_clue["earliest"] == 0
    assert sequence.satisfied_edges["e1"] == 1
    assert sequence.satisfied_edges["e2"] == 2


def test_optimize_returns_scene_sequence():
    dag = _chain_dag()
    result = optimize(
        dag,
        config=MCTSConfig(max_iterations=50, time_budget_sec=5.0),
    )
    assert len(result.scenes) == 3
    assert result.score > float("-inf")
