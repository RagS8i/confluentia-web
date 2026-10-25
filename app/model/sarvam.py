"""Minimal Sarvam chat-completions client shared by CareFlow components.

The OCR organizer and the model provider both need GLM-5.3 through Sarvam's
OpenAI-compatible ``/v2/chat/completions`` endpoint.  Keeping the transport in
one module avoids two subtly different request/response parsers and gives
callers a stable, non-leaking error type.

This module never logs prompts, responses or API keys.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import httpx

DEFAULT_SARVAM_BASE_URL = "https://api.sarvam.ai"
DEFAULT_CHAT_MODEL = "glm5.3"


class SarvamChatError(RuntimeError):
    """A safe transport/schema failure from Sarvam chat completions."""

    def __init__(self, code: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _api_key(explicit: Optional[str] = None) -> str:
    key = explicit or os.getenv("SARVAM_API_KEY")
    if not key or not str(key).strip():
        raise SarvamChatError(
            "not_configured",
            "SARVAM_API_KEY is not configured.",
            retryable=False,
        )
    return str(key).strip()


def _content_from_response(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SarvamChatError(
            "malformed_response", "Sarvam returned no completion choices."
        )
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise SarvamChatError(
            "malformed_response", "Sarvam returned an invalid message envelope."
        )
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise SarvamChatError(
            "empty_response", "Sarvam returned empty completion content."
        )
    return content.strip()


def chat_completion(
    *,
    messages: Sequence[Dict[str, str]],
    model: str = DEFAULT_CHAT_MODEL,
    schema: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_SARVAM_BASE_URL,
    timeout_s: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    reasoning_effort: Optional[str] = "low",
) -> str:
    """Call Sarvam chat completions and return the assistant's text content.

    When ``schema`` is provided the request asks for strict JSON-schema output.
    Some deployments reject response formats for a given model; in that case a
    single fallback request is made without ``response_format`` and the caller's
    parser handles the JSON body.
    """
    key = _api_key(api_key)
    endpoint = f"{base_url.rstrip('/')}/v2/chat/completions"
    headers = {
        "api-subscription-key": key,
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "careflow_response",
                "schema": schema,
                "strict": True,
            },
        }

    def _post(payload: Dict[str, Any]) -> httpx.Response:
        try:
            with httpx.Client(timeout=timeout_s) as client:
                return client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise SarvamChatError(
                "timeout", "Sarvam chat request timed out."
            ) from exc
        except httpx.HTTPError as exc:
            raise SarvamChatError(
                "unavailable", "Sarvam chat service is unavailable."
            ) from exc

    response = _post(body)
    if (
        response.status_code == 400
        and schema is not None
        and any(
            marker in response.text.lower()
            for marker in ("response_format", "json_schema", "schema")
        )
    ):
        fallback = dict(body)
        fallback.pop("response_format", None)
        response = _post(fallback)

    if response.status_code >= 400:
        status = response.status_code
        if status in (401, 403):
            raise SarvamChatError(
                "auth_failed",
                "Sarvam API authentication failed.",
                retryable=False,
            )
        if status == 429:
            raise SarvamChatError(
                "rate_limited", "Sarvam chat rate limit was reached."
            )
        if status >= 500:
            raise SarvamChatError(
                "unavailable", "Sarvam chat service returned an error."
            )
        raise SarvamChatError(
            "invalid_request",
            "Sarvam rejected the chat request.",
            retryable=False,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise SarvamChatError(
            "malformed_response", "Sarvam returned invalid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise SarvamChatError(
            "malformed_response", "Sarvam returned an invalid response."
        )
    return _content_from_response(payload)
