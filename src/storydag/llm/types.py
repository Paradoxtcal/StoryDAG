"""Data types for LLM requests and responses."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

MessageContent = Union[str, List[Dict[str, Any]]]


@dataclass(frozen=True)
class ChatMessage:
    """A single chat message for the LLM API."""

    role: str
    content: MessageContent

    def to_api_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenLogprob:
    """Log probability for a single generated token."""

    token: str
    logprob: float
    top_logprobs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Normalized response from a chat completion call."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    token_logprobs: List[TokenLogprob] = field(default_factory=list)
    raw: Optional[Any] = None
