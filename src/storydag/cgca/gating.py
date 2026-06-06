"""Causal-history embedding and logit soft gating for CGCA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

import numpy as np

Embedder = Callable[[Sequence[str]], np.ndarray]

DEFAULT_LOW_THRESHOLD = 0.15
DEFAULT_HIGH_THRESHOLD = 0.45
DEFAULT_EPSILON = 1e-8
DEFAULT_MAX_TOKEN_CACHE = 100_000


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for causal-history token gating."""

    low_threshold: float = DEFAULT_LOW_THRESHOLD
    high_threshold: float = DEFAULT_HIGH_THRESHOLD
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be < high_threshold")


class TokenEmbeddingCache:
    """Cache token-string embeddings for CGCA gating."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        max_cached: int = DEFAULT_MAX_TOKEN_CACHE,
    ) -> None:
        self._embedder = embedder
        self._max_cached = max_cached
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        if text not in self._cache:
            if len(self._cache) >= self._max_cached:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = np.asarray(self._embedder([text])[0])
        return self._cache[text]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm < 1e-12 or right_norm < 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def embed_causal_history(history_text: str, embedder: Embedder) -> np.ndarray:
    """Encode causal context text into ``h_causal``."""
    return np.asarray(embedder([history_text])[0])


def token_compatibility(token_embedding: np.ndarray, history_embedding: np.ndarray) -> float:
    """Compatibility score between one token and the causal history vector."""
    return cosine_similarity(token_embedding, history_embedding)


def compute_gate_value(compatibility: float, config: GateConfig) -> float:
    """Map compatibility to a multiplicative gate in ``[0, 1]``."""
    if compatibility < config.low_threshold:
        return 0.0
    if compatibility > config.high_threshold:
        return 1.0
    span = config.high_threshold - config.low_threshold
    return (compatibility - config.low_threshold) / span


def gate_for_token(
    token_text: str,
    history_embedding: np.ndarray,
    cache: TokenEmbeddingCache,
    config: GateConfig,
) -> float:
    """Compute the gate value for a single token string."""
    token_embedding = cache.encode(token_text)
    compatibility = token_compatibility(token_embedding, history_embedding)
    return compute_gate_value(compatibility, config)


def build_token_gate_map(
    token_texts: Mapping[int, str],
    history_embedding: np.ndarray,
    cache: TokenEmbeddingCache,
    config: GateConfig,
) -> Dict[int, float]:
    """Build token-id gate values for a candidate vocabulary subset."""
    gates: Dict[int, float] = {}
    for token_id, token_text in token_texts.items():
        if not token_text:
            continue
        gates[token_id] = gate_for_token(token_text, history_embedding, cache, config)
    return gates


def apply_gate_to_logits(
    logits: np.ndarray,
    token_gates: Mapping[int, float],
    config: GateConfig,
) -> np.ndarray:
    """Apply ``logits_new = logits + log(g + eps)`` for gated token ids."""
    modified = np.array(logits, dtype=np.float64, copy=True)
    for token_id, gate in token_gates.items():
        if token_id < 0 or token_id >= modified.shape[-1]:
            continue
        modified[token_id] += math.log(max(gate, 0.0) + config.epsilon)
    return modified


def default_sentence_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Create a sentence-transformer embedder for causal history gating."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts), convert_to_numpy=True))

    return embed
