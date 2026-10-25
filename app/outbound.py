"""Small outbound delivery adapter shared by messages and escalations.

The default mode is a local outbox so demonstrations never contact an
external service by accident. Webhook delivery is opt-in via environment
configuration. Delivery failures are surfaced to the caller and audited.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict

from .config import Settings


class DeliveryError(RuntimeError):
    pass


def deliver(settings: Settings, mode: str, payload: Dict[str, Any]) -> str:
    normalized = (mode or "outbox").strip().lower()
    if normalized == "outbox":
        return "outbox"
    if normalized == "fail":
        raise DeliveryError("simulated delivery failure")
    if normalized == "webhook":
        if not settings.outbound_webhook_url:
            raise DeliveryError("outbound webhook URL is not configured")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            settings.outbound_webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=settings.outbound_timeout_s
            ) as response:
                if response.status >= 300:
                    raise DeliveryError(f"webhook returned HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DeliveryError(f"webhook delivery failed: {type(exc).__name__}") from exc
        return "webhook"
    raise DeliveryError(f"unsupported delivery mode: {mode}")
