"""HuggingFace logits processor hook for causal-history gating."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from storydag.cgca.gating import (
    Embedder,
    GateConfig,
    TokenEmbeddingCache,
    apply_gate_to_logits,
    build_token_gate_map,
    embed_causal_history,
)

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - optional dependency
    class LogitsProcessor:  # type: ignore[no-redef]
        """Fallback base class when ``transformers`` is not installed."""

        def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
            return scores


class CausalGateLogitsProcessor(LogitsProcessor):
    """Apply causal-history soft gating during local LM generation.

    Intended for use with HuggingFace ``generate(..., logits_processor=[...],
    num_beams=5)`` where beam search applies the gate at each decoding step.
    """

    def __init__(
        self,
        history_text: str,
        token_id_to_text: Sequence[str],
        *,
        embedder: Optional[Embedder] = None,
        config: Optional[GateConfig] = None,
        cache: Optional[TokenEmbeddingCache] = None,
    ) -> None:
        from storydag.cgca.gating import default_sentence_embedder

        self.config = config or GateConfig()
        self._embedder = embedder or default_sentence_embedder()
        self._cache = cache or TokenEmbeddingCache(self._embedder)
        self._history_embedding = embed_causal_history(history_text, self._embedder)
        self._token_gates = build_token_gate_map(
            {index: text for index, text in enumerate(token_id_to_text)},
            self._history_embedding,
            self._cache,
            self.config,
        )

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if scores.ndim == 1:
            modified = apply_gate_to_logits(scores, self._token_gates, self.config)
            return modified.astype(scores.dtype, copy=False)

        batch: List[np.ndarray] = []
        for row in scores:
            batch.append(apply_gate_to_logits(row, self._token_gates, self.config))
        return np.stack(batch).astype(scores.dtype, copy=False)
