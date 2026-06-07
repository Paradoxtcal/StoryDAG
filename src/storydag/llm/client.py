"""OpenAI-compatible LLM client for CNGE extraction and CGCA generation."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI
from openai import APITimeoutError, APIConnectionError

from storydag.config import get_optional_env, get_required_env, load_env
from storydag.llm.types import ChatMessage, LLMResponse, TokenLogprob

MessageInput = Union[ChatMessage, Dict[str, Any]]


class LLMClient:
    """Thin wrapper around the OpenAI SDK for StoryDAG pipeline modules.

    CNGE uses ``temperature=0`` for deterministic causal-triple extraction.
    ``json_mode`` is **not** used by default because some providers
    (DeepSeek, Ollama, vLLM) may not support ``response_format:
    {"type": "json_object"}``; the system prompt + few-shot examples
    already enforce JSON output.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4-turbo",
        timeout: float = 60.0,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,  # we handle retries ourselves
        )

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "LLMClient":
        """Build a client from ``.env`` / process environment variables.

        Supports ``OPENAI_TIMEOUT`` (default 180s) for providers with
        slower inference (DeepSeek, local models).
        """
        values = env if env is not None else load_env()
        timeout = float(get_optional_env(values, "OPENAI_TIMEOUT", "180"))
        return cls(
            api_key=get_required_env(values, "OPENAI_API_KEY"),
            base_url=get_optional_env(values, "OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=get_optional_env(values, "LLM_MODEL", "gpt-4-turbo"),
            timeout=timeout,
        )

    def chat(
        self,
        messages: Sequence[MessageInput],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a normalized response."""
        api_messages = [self._to_api_message(m) for m in messages]
        kwargs: Dict[str, object] = {
            "model": model or self.model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except APITimeoutError:
            raise ConnectionError(
                f"LLM API 请求超时（{self.timeout}s），请检查模型可用性或增大 timeout"
            ) from None
        except APIConnectionError as exc:
            raise ConnectionError(
                f"无法连接到 LLM API（{self.base_url}），请检查网络与 BASE_URL"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"LLM API 调用失败: {exc}"
            ) from exc
        choice = completion.choices[0]
        content = choice.message.content or ""

        usage = None
        if completion.usage is not None:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        token_logprobs: List[TokenLogprob] = []
        if logprobs and choice.logprobs and choice.logprobs.content:
            for item in choice.logprobs.content:
                token_logprobs.append(
                    TokenLogprob(
                        token=item.token,
                        logprob=item.logprob,
                        top_logprobs=[
                            {"token": t.token, "logprob": t.logprob}
                            for t in (item.top_logprobs or [])
                        ],
                    )
                )

        return LLMResponse(
            content=content,
            model=completion.model,
            usage=usage,
            token_logprobs=token_logprobs,
            raw=completion,
        )

    def ping(self) -> bool:
        """Test connectivity with a minimal request.

        Returns ``True`` if the API responds, ``False`` otherwise (writes
        a diagnostic to stderr).
        """
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            return True
        except APIConnectionError as exc:
            print(
                f"  错误：无法连接到 {self.base_url}/chat/completions\n"
                f"        请检查 OPENAI_BASE_URL 和网络连接。\n"
                f"        ({exc})",
                file=sys.stderr,
            )
            return False
        except Exception as exc:
            # A 4xx/5xx response means we *did* connect, so the URL is valid.
            # This is fine — the model/key might not match but that will
            # surface during actual extraction.
            return True

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Convenience wrapper: build messages and return assistant text only."""
        messages: List[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        return self.chat(
            messages,
            temperature=temperature,
            json_mode=json_mode,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            model=model,
        ).content

    @staticmethod
    def _to_api_message(message: MessageInput) -> Dict[str, object]:
        if isinstance(message, ChatMessage):
            return message.to_api_dict()
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str):
            raise TypeError("message role must be a string")
        return {"role": role, "content": content}
