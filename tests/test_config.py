"""Tests for configuration helpers."""

from storydag.config import chat_completions_url, load_env, project_root


def test_project_root_exists():
    root = project_root()
    assert (root / "pyproject.toml").exists()


def test_chat_completions_url():
    assert chat_completions_url("https://api.openai.com/v1") == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert chat_completions_url("https://api.openai.com/v1/chat/completions") == (
        "https://api.openai.com/v1/chat/completions"
    )


def test_load_env_returns_dict():
    values = load_env()
    assert isinstance(values, dict)
