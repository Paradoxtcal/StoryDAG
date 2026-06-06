"""Tests for CausOpt MCTS hard-pruning search."""

from storydag.causopt.mcts import (
    MCTSConfig,
    MCTSNode,
    backpropagate,
    cluster_ready_nodes,
    expand,
    generate_expansions,
    greedy_rollout,
    search,
    select,
    topological_order,
    uct_value,
)
from storydag.causopt.models import CausalDAG, DramaticObjectives, is_valid_assignment
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


def test_topological_order_and_clustering():
    dag = _chain_dag()
    topo = topological_order(dag)
    assert topo == ["n1", "n2", "n3"]

    ready = dag.ready_nodes({"n1"})
    clusters = cluster_ready_nodes(dag, ready, topo)
    assert clusters == [["n2"]]


def test_generate_expansions_only_uses_ready_nodes():
    dag = _chain_dag()
    root = MCTSNode(scenes=[])
    expansions = generate_expansions(dag, root, DramaticObjectives(), max_branching=10)
    assert expansions == [["n1"]]

    mid = MCTSNode(scenes=[["n1"]])
    expansions = generate_expansions(dag, mid, DramaticObjectives(), max_branching=10)
    assert expansions == [["n2"]]


def test_generate_expansions_respects_branching_limit():
    dag = CausalDAG.from_causal_graph(
        build_dag(
            [GraphNode(f"n{i}", f"节点{i}", "event") for i in range(1, 8)],
            [],
        )
    )
    root = MCTSNode(scenes=[])
    expansions = generate_expansions(dag, root, DramaticObjectives(), max_branching=3)
    assert len(expansions) <= 3


def test_expand_and_backpropagate_update_tree():
    dag = _chain_dag()
    root = MCTSNode(scenes=[])
    config = MCTSConfig(max_branching=10)

    child = expand(dag, root, DramaticObjectives(), config)
    assert child.scenes == [["n1"]]
    assert child.parent is root
    assert root.children == [child]

    backpropagate(child, 0.8)
    assert child.visit_count == 1
    assert child.total_score == 0.8


def test_uct_prefers_unvisited_child():
    parent = MCTSNode(scenes=[])
    parent.visit_count = 10
    visited = MCTSNode(scenes=[["n1"]], visit_count=5, total_score=2.5)
    unvisited = MCTSNode(scenes=[["n2"]], visit_count=0)
    assert uct_value(parent, unvisited, 0.5) == float("inf")
    assert uct_value(parent, visited, 0.5) < float("inf")


def test_select_stops_at_unexpanded_node():
    dag = _chain_dag()
    root = MCTSNode(scenes=[])
    leaf = select(root, dag, 0.5)
    assert leaf is root
    assert not leaf.children


def test_greedy_rollout_completes_chain():
    dag = _chain_dag()
    root = MCTSNode(scenes=[])
    assignment = greedy_rollout(dag, root)
    assert assignment == [["n1"], ["n2"], ["n3"]]
    assert is_valid_assignment(dag, assignment)


def test_search_returns_valid_assignment():
    dag = _chain_dag()
    assignment = search(
        dag,
        config=MCTSConfig(max_iterations=50, time_budget_sec=5.0),
    )
    assert is_valid_assignment(dag, assignment)
    assert assignment == [["n1"], ["n2"], ["n3"]]
