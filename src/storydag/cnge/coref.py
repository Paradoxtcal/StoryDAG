"""Coreference resolution and node clustering for CNGE post-processing."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from storydag.cnge.types import CausalTriple, GraphEdge, GraphNode

Embedder = Callable[[Sequence[str]], np.ndarray]

DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Providers with a dedicated embeddings endpoint.
# DeepSeek's OpenAI-compatible surface is chat-completions only — no embeddings.
_EMBEDDING_PROVIDERS = frozenset({"openai", "gemini"})


# ── API-based embedders ────────────────────────────────────────────


def _make_openai_compat_embedder(
    model_spec: str,
    env: Optional[Dict[str, str]] = None,
) -> Embedder:
    """OpenAI-compatible embedding API (OpenAI, etc.).

    ``model_spec`` format: ``provider/model-name``

    Credential lookup order (``.env`` / environment):
      1. ``EMBEDDING_API_KEY``  → fallback ``OPENAI_API_KEY``
      2. ``EMBEDDING_BASE_URL``  → fallback ``https://api.openai.com/v1``
    """
    from openai import OpenAI
    from openai import APIStatusError

    from storydag.config import get_optional_env, load_env

    resolved_env = env if env is not None else load_env()
    _, actual_model = model_spec.split("/", 1)

    api_key = get_optional_env(resolved_env, "EMBEDDING_API_KEY", "") or \
              get_optional_env(resolved_env, "OPENAI_API_KEY", "")
    base_url = get_optional_env(resolved_env, "EMBEDDING_BASE_URL", "") or \
               "https://api.openai.com/v1"

    if not api_key:
        raise RuntimeError(
            "OpenAI-compatible embedder 需要 API key。\n"
            "请在 .env 中设置 EMBEDDING_API_KEY=sk-...\n"
            "（若与 LLM 共用同一 key 可省略此项，自动回退到 OPENAI_API_KEY）"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(labels: Sequence[str]) -> np.ndarray:
        all_embeds: list[list[float]] = []
        batch_size = 20
        for i in range(0, len(labels), batch_size):
            batch = list(labels[i : i + batch_size])
            try:
                resp = client.embeddings.create(model=actual_model, input=batch)
            except APIStatusError as exc:
                raise RuntimeError(
                    f"Embedding API 错误 ({exc.status_code})\n"
                    f"  模型: {actual_model}\n"
                    f"  端点: {base_url}/embeddings\n"
                    f"  详情: {exc.response.text[:500]}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Embedding API 调用失败: {exc}"
                ) from exc
            all_embeds.extend([item.embedding for item in resp.data])
        return np.asarray(all_embeds, dtype=np.float64)

    return embed


def _make_gemini_embedder(
    model_spec: str,
    env: Optional[Dict[str, str]] = None,
) -> Embedder:
    """Google Gemini embedding API.

    ``model_spec`` format: ``gemini/model-name``
    Reads ``GEMINI_API_KEY`` from ``.env`` or environment.
    """
    import json
    import urllib.error
    import urllib.request

    from storydag.config import get_optional_env, load_env

    resolved_env = env if env is not None else load_env()
    _, actual_model = model_spec.split("/", 1)
    api_key = get_optional_env(resolved_env, "GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Gemini embedder 需要 GEMINI_API_KEY。\n"
            "请在 .env 中设置: GEMINI_API_KEY=your_key"
        )

    url_template = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{actual_model}:batchEmbedContents?key={api_key}"
    )

    def embed(labels: Sequence[str]) -> np.ndarray:
        all_embeds: list[list[float]] = []
        batch_size = 20
        for i in range(0, len(labels), batch_size):
            batch = list(labels[i : i + batch_size])
            payload = {
                "requests": [
                    {
                        "model": f"models/{actual_model}",
                        "content": {"parts": [{"text": t}]},
                    }
                    for t in batch
                ]
            }
            req = urllib.request.Request(
                url_template,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
                for item in data.get("embeddings", []):
                    all_embeds.append(item["values"])
            except urllib.error.HTTPError as e:
                body = e.read().decode()
                raise RuntimeError(
                    f"Gemini API 错误 ({e.code})\n"
                    f"  模型: {actual_model}\n"
                    f"  详情: {body[:500]}"
                ) from e
        return np.asarray(all_embeds, dtype=np.float64)

    return embed


# ── Embedder factory ──────────────────────────────────────────────


def _default_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    env: Optional[Dict[str, str]] = None,
) -> Embedder:
    """Dispatch to the right embedder based on ``model_name`` prefix.

    +------------------------------------------+---------------------------+
    | ``EMBEDDING_MODEL``                      | Backend                   |
    +------------------------------------------+---------------------------+
    | ``all-MiniLM-L6-v2`` (no ``/``)          | sentence-transformers     |
    | ``deepseek/<model>``                     | ✗ 不支持 (无 embeddings)  |
    | ``openai/<model>``                       | OpenAI-compatible API     |
    | ``gemini/<model>``                       | Google Gemini API         |
    +------------------------------------------+---------------------------+
    """
    if "/" in model_name:
        provider = model_name.split("/", 1)[0].lower()
        if provider == "deepseek":
            raise ValueError(
                "DeepSeek 不提供 embeddings API。\n"
                "其 OpenAI 兼容接口仅支持 chat/completions，没有 /v1/embeddings 端点。\n"
                "可用替代方案（修改 .env 中的 EMBEDDING_MODEL）:\n"
                "  all-MiniLM-L6-v2          (本地 sentence-transformers，需 HF_ENDPOINT 镜像)\n"
                "  openai/text-embedding-3-small  (需 EMBEDDING_API_KEY)\n"
                "  gemini/text-embedding-004      (需 GEMINI_API_KEY)"
            )
        elif provider in ("openai",):
            return _make_openai_compat_embedder(model_name, env=env)
        elif provider == "gemini":
            return _make_gemini_embedder(model_name, env=env)
        else:
            raise ValueError(
                f"不支持的 embedder provider: {provider!r}。\n"
                f"支持的格式:\n"
                f"  all-MiniLM-L6-v2              (本地 sentence-transformers)\n"
                f"  openai/<模型名>                (OpenAI API)\n"
                f"  gemini/<模型名>                (Google Gemini API)"
            )

    # No prefix → local sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers 未安装。请执行: pip install sentence-transformers\n"
            "或改用 API embedder，在 .env 中设置 EMBEDDING_MODEL=openai/text-embedding-3-small"
        )

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"sentence-transformers 模型 {model_name} 下载失败。\n"
            f"中国大陆用户建议设置:\n"
            f"  export HF_ENDPOINT=https://hf-mirror.com\n"
            f"或改用 API embedder:\n"
            f"  openai/text-embedding-3-small  (需 EMBEDDING_API_KEY)\n"
            f"原始错误: {exc}"
        ) from exc

    def embed(labels: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(labels), convert_to_numpy=True))

    return embed


# ── Similarity & clustering ──────────────────────────────────────


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


# ── Node/edge helpers ─────────────────────────────────────────────


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
    env: Optional[Dict[str, str]] = None,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """Merge nodes with similar labels and rebuild deduplicated edges.

    Implements the CNGE post-processing step:
    1. Embed node labels with the configured embedder
    2. Cluster by cosine similarity > threshold using union-find
    3. Assign canonical IDs ``n1``, ``n2``, ...
    4. Rebuild edges with merged IDs and drop self-loops / duplicates
    """
    if not triples:
        return [], []

    raw_nodes = _collect_raw_nodes(triples)
    labels = [label for _, label, _ in raw_nodes]
    embed = embedder or _default_embedder(embedding_model, env=env)
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
