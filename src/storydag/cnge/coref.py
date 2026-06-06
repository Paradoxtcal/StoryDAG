"""Coreference resolution and node clustering for CNGE post-processing."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from storydag.cnge.types import CausalTriple, GraphEdge, GraphNode

Embedder = Callable[[Sequence[str]], np.ndarray]

DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _default_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(labels: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(labels), convert_to_numpy=True))

    return embed


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-12, None)
    return normalized @ normalized.T


def _cluster_indices(similarity: np.ndarray, threshold: float) -> List[List[int]]:
    """Single-linkage clustering via union-find on cosine similarity."""
    size = similarity.shape[0]
    if size == 0:
        return []
    if size == 1:
        return [[0]]

    parent = list(range(size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(size):
        for j in range(i + 1, size):
            if similarity[i, j] > threshold:
                union(i, j)

    grouped: Dict[int, List[int]] = {}
    for index in range(size):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _collect_raw_nodes(triples: Sequence[CausalTriple]) -> List[Tuple[str, str, str]]:
    """Collect unique nodes as ``(node_id, label, type)`` tuples."""
    seen: Dict[str, Tuple[str, str]] = {}
    order: List[str] = []

    for triple in triples:
        for node_id, label, node_type in (
            (triple.source_id, triple.source_label, triple.source_type),
            (triple.target_id, triple.target_label, triple.target_type),
        ):
            if node_id not in seen:
                seen[node_id] = (label, node_type)
                order.append(node_id)

    return [(node_id, seen[node_id][0], seen[node_id][1]) for node_id in order]


def _choose_representative_label(labels: Sequence[str]) -> str:
    return max(labels, key=len)


def _choose_representative_type(types: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for node_type in types:
        counts[node_type] = counts.get(node_type, 0) + 1
    return max(counts, key=lambda key: (counts[key], key))


def resolve_coreferences(
    triples: Sequence[CausalTriple],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    embedder: Optional[Embedder] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """Merge nodes with similar labels and rebuild deduplicated edges.

    Implements the CNGE post-processing step:
    1. Embed node labels with ``all-MiniLM-L6-v2`` (or injected embedder)
    2. Cluster by cosine similarity > threshold using union-find
    3. Assign canonical IDs ``n1``, ``n2``, ...
    4. Rebuild edges with merged IDs and drop self-loops / duplicates
    """
    if not triples:
        return [], []

    raw_nodes = _collect_raw_nodes(triples)
    labels = [label for _, label, _ in raw_nodes]
    embed = embedder or _default_embedder(embedding_model)
    embeddings = embed(labels)
    similarity = _cosine_similarity_matrix(embeddings)
    clusters = _cluster_indices(similarity, similarity_threshold)

    nodes: List[GraphNode] = []
    id_mapping: Dict[str, str] = {}

    for cluster_index, member_indices in enumerate(clusters, start=1):
        canonical_id = f"n{cluster_index}"
        member_nodes = [raw_nodes[index] for index in member_indices]
        for node_id, _, _ in member_nodes:
            id_mapping[node_id] = canonical_id

        nodes.append(
            GraphNode(
                node_id=canonical_id,
                label=_choose_representative_label([label for _, label, _ in member_nodes]),
                type=_choose_representative_type([node_type for _, _, node_type in member_nodes]),
                source_ids=sorted({node_id for node_id, _, _ in member_nodes}),
            )
        )

    edges: List[GraphEdge] = []
    seen_edges = set()
    edge_counter = 1

    for triple in triples:
        source = id_mapping[triple.source_id]
        target = id_mapping[triple.target_id]
        if source == target:
            continue

        key = (source, target, triple.edge_type)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        edges.append(
            GraphEdge(
                edge_id=f"e{edge_counter}",
                source=source,
                target=target,
                type=triple.edge_type,
                strength=1.0,
            )
        )
        edge_counter += 1

    return nodes, edges
