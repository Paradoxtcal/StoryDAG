"""Monte Carlo Tree Search with hard pruning for CausOpt."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from storydag.causopt.models import (
    CausalDAG,
    DramaticObjectives,
    SceneAssignment,
    build_partial_sequence,
    is_valid_assignment,
)

EvaluateFn = Callable[[CausalDAG, SceneAssignment, DramaticObjectives], float]

NODE_DRAMATIC_WEIGHT = {
    "revelation": 4,
    "intention": 2,
    "event": 1,
    "emotional_state": 1,
}

DEFAULT_EXPLORATION_CONSTANT = 0.5
DEFAULT_MAX_ITERATIONS = 10_000
DEFAULT_TIME_BUDGET_SEC = 300.0
DEFAULT_MAX_BRANCHING = 10


@dataclass
class MCTSConfig:
    """Hyper-parameters for MCTS-HP search."""

    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC
    max_branching: int = DEFAULT_MAX_BRANCHING


@dataclass
class MCTSNode:
    """A node in the MCTS search tree representing a partial scene sequence."""

    scenes: List[List[str]]
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    total_score: float = 0.0
    untried_actions: List[List[str]] = field(default_factory=list)

    @property
    def assigned_nodes(self) -> Set[str]:
        assigned: Set[str] = set()
        for scene in self.scenes:
            assigned.update(scene)
        return assigned

    def is_terminal(self, dag: CausalDAG) -> bool:
        return self.assigned_nodes == dag.node_ids

    def average_score(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_score / self.visit_count


def dramatic_weight(dag: CausalDAG, node_id: str) -> int:
    node_type = dag.nodes[node_id].type
    return NODE_DRAMATIC_WEIGHT.get(node_type, 1)


def topological_order(dag: CausalDAG) -> List[str]:
    """Return a topological order of DAG nodes (Kahn's algorithm)."""
    indegree = {node_id: len(dag.predecessors.get(node_id, set())) for node_id in dag.node_ids}
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    order: List[str] = []

    while queue:
        queue.sort()
        current = queue.pop(0)
        order.append(current)
        for edge in dag.edges:
            if edge.source != current:
                continue
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                queue.append(edge.target)

    if len(order) != len(dag.node_ids):
        raise ValueError("CausalDAG contains a cycle")
    return order


def cluster_ready_nodes(dag: CausalDAG, ready_nodes: Set[str], topo: Sequence[str]) -> List[List[str]]:
    """Group ready nodes into narrative-proximity clusters via topological order."""
    ordered = [node_id for node_id in topo if node_id in ready_nodes]
    if not ordered:
        return []

    index_map = {node_id: index for index, node_id in enumerate(topo)}
    clusters: List[List[str]] = []
    current = [ordered[0]]
    for node_id in ordered[1:]:
        if index_map[node_id] - index_map[current[-1]] <= 2:
            current.append(node_id)
        else:
            clusters.append(current)
            current = [node_id]
    clusters.append(current)
    return clusters


def expansion_heuristic(
    dag: CausalDAG,
    scene_nodes: Sequence[str],
    num_scenes_so_far: int,
    objectives: DramaticObjectives,
) -> float:
    """Rank candidate expansions: dramatic weight + act-progress bonus."""
    weight_score = sum(dramatic_weight(dag, node_id) for node_id in scene_nodes)
    act_bonus = 0.0
    if objectives.act_proportions:
        target_act1 = objectives.act_proportions[0]
        expected_scenes = max(1, int(objectives.max_scenes * target_act1))
        if num_scenes_so_far < expected_scenes:
            act_bonus = 1.0
    return weight_score + act_bonus


def generate_expansions(
    dag: CausalDAG,
    node: MCTSNode,
    objectives: DramaticObjectives,
    *,
    max_branching: int = DEFAULT_MAX_BRANCHING,
) -> List[List[str]]:
    """Generate up to ``max_branching`` valid next-scene expansions from ready nodes."""
    ready = dag.ready_nodes(node.assigned_nodes)
    if not ready:
        return []

    topo = topological_order(dag)
    clusters = cluster_ready_nodes(dag, ready, topo)

    candidates: List[List[str]] = []
    seen = set()

    def add_candidate(scene_nodes: Sequence[str]) -> None:
        key = tuple(sorted(scene_nodes))
        if key not in seen:
            seen.add(key)
            candidates.append(list(scene_nodes))

    for cluster in clusters:
        add_candidate(cluster)
    for node_id in sorted(ready, key=lambda nid: dramatic_weight(dag, nid), reverse=True):
        add_candidate([node_id])

    ranked = sorted(
        candidates,
        key=lambda scene_nodes: expansion_heuristic(
            dag, scene_nodes, len(node.scenes), objectives
        ),
        reverse=True,
    )
    return ranked[:max_branching]


def uct_value(parent: MCTSNode, child: MCTSNode, exploration_constant: float) -> float:
    """Upper Confidence Bound for Trees."""
    if child.visit_count == 0:
        return float("inf")
    exploitation = child.average_score()
    exploration = exploration_constant * math.sqrt(math.log(parent.visit_count) / child.visit_count)
    return exploitation + exploration


def select(node: MCTSNode, dag: CausalDAG, exploration_constant: float) -> MCTSNode:
    """Traverse the tree using UCT until reaching an unexpanded or terminal node."""
    current = node
    while not current.is_terminal(dag):
        if current.untried_actions or not current.children:
            return current
        current = max(
            current.children,
            key=lambda child: uct_value(current, child, exploration_constant),
        )
    return current


def expand(
    dag: CausalDAG,
    node: MCTSNode,
    objectives: DramaticObjectives,
    config: MCTSConfig,
) -> MCTSNode:
    """Expand one child from ``node`` using hard-pruned ready-node actions."""
    if node.is_terminal(dag):
        return node

    if not node.untried_actions:
        node.untried_actions = generate_expansions(
            dag,
            node,
            objectives,
            max_branching=config.max_branching,
        )

    if not node.untried_actions:
        return node

    action = node.untried_actions.pop(0)
    child = MCTSNode(scenes=[*node.scenes, action], parent=node)
    node.children.append(child)
    return child


def greedy_rollout(
    dag: CausalDAG,
    node: MCTSNode,
) -> SceneAssignment:
    """Deterministic rollout placeholder; replaced by full scorer in PR #9."""
    assignment = [list(scene) for scene in node.scenes]
    assigned = node.assigned_nodes.copy()

    while assigned != dag.node_ids:
        ready = dag.ready_nodes(assigned)
        if not ready:
            break
        next_node = max(ready, key=lambda node_id: (dramatic_weight(dag, node_id), node_id))
        assignment.append([next_node])
        assigned.add(next_node)
    return assignment


def placeholder_evaluate(
    dag: CausalDAG,
    assignment: SceneAssignment,
    objectives: DramaticObjectives,
) -> float:
    """Minimal evaluation for MCTS backprop until PR #9 scoring lands."""
    del objectives
    return 1.0 if is_valid_assignment(dag, assignment) else 0.0


def backpropagate(node: MCTSNode, score: float) -> None:
    """Update visit counts and total scores along the selection path."""
    current: Optional[MCTSNode] = node
    while current is not None:
        current.visit_count += 1
        current.total_score += score
        current = current.parent


def search(
    dag: CausalDAG,
    objectives: Optional[DramaticObjectives] = None,
    *,
    config: Optional[MCTSConfig] = None,
    evaluate_fn: Optional[EvaluateFn] = None,
) -> SceneAssignment:
    """Run MCTS-HP and return the best valid scene assignment found."""
    objectives = objectives or DramaticObjectives()
    config = config or MCTSConfig()
    evaluate = evaluate_fn or placeholder_evaluate

    root = MCTSNode(scenes=[])
    best_assignment: Optional[SceneAssignment] = None
    best_score = float("-inf")
    deadline = time.monotonic() + config.time_budget_sec

    for _ in range(config.max_iterations):
        if time.monotonic() >= deadline:
            break

        leaf = select(root, dag, config.exploration_constant)
        if leaf.is_terminal(dag):
            rollout_assignment = [list(scene) for scene in leaf.scenes]
        else:
            leaf = expand(dag, leaf, objectives, config)
            rollout_assignment = greedy_rollout(dag, leaf)

        score = evaluate(dag, rollout_assignment, objectives)
        backpropagate(leaf, score)

        if score > best_score and is_valid_assignment(dag, rollout_assignment):
            best_score = score
            best_assignment = rollout_assignment

    if best_assignment is not None:
        return best_assignment

    fallback = greedy_rollout(dag, root)
    if is_valid_assignment(dag, fallback):
        return fallback
    raise RuntimeError("MCTS failed to find a valid scene assignment")
