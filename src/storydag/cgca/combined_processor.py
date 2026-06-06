"""Combined soft gate and hard blacklist logits processor."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from storydag.cgca.blacklist import apply_hard_blacklist
from storydag.cgca.logits_processor import CausalGateLogitsProcessor

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - optional dependency
    class LogitsProcessor:  # type: ignore[no-redef]
        def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
            return scores


class CombinedCausalLogitsProcessor(LogitsProcessor):
    """Apply PR #11 soft gating, then PR #12 hard blacklist blocking."""

    def __init__(
        self,
        gate_processor: CausalGateLogitsProcessor,
        blocked_token_ids: Iterable[int],
    ) -> None:
        self._gate_processor = gate_processor
        self._blocked_token_ids = set(blocked_token_ids)

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        gated = self._gate_processor(input_ids, scores)
        if gated.ndim == 1:
            return apply_hard_blacklist(gated, self._blocked_token_ids).astype(scores.dtype, copy=False)

        rows: List[np.ndarray] = []
        for row in gated:
            rows.append(apply_hard_blacklist(row, self._blocked_token_ids))
        return np.stack(rows).astype(scores.dtype, copy=False)
