"""Minimal shared client for Ollama chat completions."""
from __future__ import annotations

from typing import Any

import requests

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def chat_url(base_url: str | None) -> str:
    """Normalize an Ollama base URL to its chat-completions endpoint."""
    base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")

    if base_url.endswith("/api/chat"):
        return base_url

    return f"{base_url}/api/chat"


def post_chat(
    base_url: str | None,
    payload: dict[str, Any],
    *,
    timeout: tuple[float, float],
    action: str,
) -> dict[str, Any]:
    """POST a chat request to Ollama and return the decoded JSON response."""
    url = chat_url(base_url)

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. Make sure Ollama is running."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(f"Ollama timed out while {action}.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed while {action}: {exc}") from exc

    if not response.ok:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned a non-JSON API response.") from exc
