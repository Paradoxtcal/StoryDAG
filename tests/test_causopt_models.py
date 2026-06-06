"""Tests for CausOpt data structures and hard constraints."""

import pytest

from storydag.causopt import (
    CausalDAG,
    DramaticObjectives,
    build_partial_sequence,
    default_sigmoid_pacing,
    is_valid_assignment,
    node_scene_index,
)
from storydag.cnge.graph import build_dag
from storydag.cnge.types import GraphEdge, GraphNode


def _sample_dag() -> CausalDAG:
    nodes = [
        GraphNode("n1", "甲", "event", ["n1"]),
        GraphNode("n2", "乙", "intention", ["n2"]),
        GraphNode("n3", "丙", "event", ["n3"]),
    ]
    edges = [
        GraphEdge("e1", "n1", "n2", "motivates"),
        GraphEdge("e2", "n2", "n3", "motivates"),
    ]
    graph = build_dag(nodes, edges)
    return CausalDAG.from_causal_graph(graph)


def test_causal_dag_from_causal_graph():
    dag = _sample_dag()
    assert set(dag.node_ids) == {"n1", "n2", "n3"}
    assert dag.predecessors["n1"] == set()
    assert dag.predecessors["n2"] == {"n1"}
    assert dag.predecessors["n3"] == {"n2"}


def test_ready_nodes_and_blocked_nodes():
    dag = _sample_dag()
    assert dag.ready_nodes(set()) == {"n1"}
    assert dag.blocked_nodes(set()) == {"n2", "n3"}

    assigned = {"n1"}
    assert dag.ready_nodes(assigned) == {"n2"}
    assert dag.blocked_nodes(assigned) == {"n3"}


def test_is_valid_assignment_accepts_causal_order():
    dag = _sample_dag()
    assert is_valid_assignment(dag, [["n1"], ["n2"], ["n3"]]) is True
    # Independent nodes with no edge between them may share a scene.
    dag_parallel = CausalDAG.from_causal_graph(
        build_dag(
            [
                GraphNode("n1", "甲", "event"),
                GraphNode("n2", "乙", "event"),
            ],
            [],
        )
    )
    assert is_valid_assignment(dag_parallel, [["n1", "n2"]]) is True


def test_is_valid_assignment_rejects_precedence_violation():
    dag = _sample_dag()
    assignment = [["n2"], ["n1", "n3"]]
    assert is_valid_assignment(dag, assignment) is False


def test_is_valid_assignment_rejects_missing_or_duplicate_nodes():
    dag = _sample_dag()
    assert is_valid_assignment(dag, [["n1"], ["n2"]]) is False
    assert is_valid_assignment(dag, [["n1", "n2", "n3", "n3"]]) is False


def test_node_scene_index_mapping():
    assignment = [["n1", "n2"], ["n3"]]
    assert node_scene_index(assignment) == {"n1": 0, "n2": 0, "n3": 1}


def test_build_partial_sequence_snapshot():
    dag = _sample_dag()
    partial = build_partial_sequence(dag, [["n1"]])
    assert partial.assigned_nodes == {"n1"}
    assert partial.unassigned_nodes == {"n2", "n3"}
    assert partial.blocked_nodes == {"n3"}
    assert partial.is_complete is False


def test_dramatic_objectives_defaults_and_validation():
    objectives = DramaticObjectives()
    assert objectives.act_proportions == [0.25, 0.5, 0.25]
    assert objectives.max_scenes == 50
    assert objectives.min_scenes_per_act == 3

    with pytest.raises(ValueError, match="act_proportions"):
        DramaticObjectives(act_proportions=[0.5, 0.5, 0.5])


def test_default_sigmoid_pacing_curve():
    curve = default_sigmoid_pacing(5)
    assert len(curve) == 5
    assert curve[0] < curve[-1]
    assert all(0.0 < value < 1.0 for value in curve)
