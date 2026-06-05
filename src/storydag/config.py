"""Environment and API configuration helpers."""

import os
import sys
from pathlib import Path
from typing import Dict


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_env() -> Dict[str, str]:
    """Load .env values, then let real environment variables override them."""
    values: Dict[str, str] = {}
    env_path = project_root() / ".env"

    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")

    for key, value in os.environ.items():
        if value:
            values[key] = value
    return values


def get_required_env(values: Dict[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        print(f"错误：请在 .env 或环境变量中配置 {name}", file=sys.stderr)
        sys.exit(1)
    return value


def get_optional_env(values: Dict[str, str], name: str, default: str) -> str:
    return values.get(name, "").strip() or default


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"
