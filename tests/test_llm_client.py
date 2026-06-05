"""Tests for the OpenAI-compatible LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from storydag.llm import ChatMessage, LLMClient, LLMResponse


def _mock_completion(content: str = "hello", with_logprobs: bool = False):
    completion = MagicMock()
    completion.model = "gpt-4-turbo"
    completion.usage.prompt_tokens = 10
    completion.usage.completion_tokens = 5
    completion.usage.total_tokens = 15

    choice = MagicMock()
    choice.message.content = content

    if with_logprobs:
        logprob_item = MagicMock()
        logprob_item.token = "hello"
        logprob_item.logprob = -0.1
        top = MagicMock()
        top.token = "hello"
        top.logprob = -0.1
        logprob_item.top_logprobs = [top]
        choice.logprobs.content = [logprob_item]
    else:
        choice.logprobs = None

    completion.choices = [choice]
    return completion


def test_from_env_reads_configuration():
    env = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://example.com/v1",
        "LLM_MODEL": "gpt-4o",
    }
    client = LLMClient.from_env(env)
    assert client.api_key == "test-key"
    assert client.base_url == "https://example.com/v1"
    assert client.model == "gpt-4o"


def test_from_env_missing_api_key_exits():
    with pytest.raises(SystemExit):
        LLMClient.from_env({})


def test_chat_returns_normalized_response():
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    mock_create = MagicMock(return_value=_mock_completion('{"nodes": []}'))
    client._client = MagicMock()
    client._client.chat.completions.create = mock_create

    response = client.chat(
        [ChatMessage(role="user", content="extract graph")],
        json_mode=True,
        temperature=0.0,
    )

    assert isinstance(response, LLMResponse)
    assert response.content == '{"nodes": []}'
    assert response.model == "gpt-4-turbo"
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_chat_with_logprobs():
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    client._client = MagicMock()
    client._client.chat.completions.create = MagicMock(
        return_value=_mock_completion("hi", with_logprobs=True)
    )

    response = client.chat(
        [ChatMessage(role="user", content="say hi")],
        logprobs=True,
        top_logprobs=5,
    )

    assert len(response.token_logprobs) == 1
    assert response.token_logprobs[0].token == "hello"
    assert response.token_logprobs[0].logprob == -0.1


def test_complete_builds_system_and_user_messages():
    client = LLMClient(api_key="k", base_url="https://example.com/v1")
    with patch.object(client, "chat", return_value=LLMResponse(content="done", model="m")) as mock_chat:
        text = client.complete("prompt", system="sys", json_mode=True)

    assert text == "done"
    messages = mock_chat.call_args.args[0]
    assert messages[0].role == "system"
    assert messages[0].content == "sys"
    assert messages[1].role == "user"
    assert messages[1].content == "prompt"
    assert mock_chat.call_args.kwargs["json_mode"] is True
