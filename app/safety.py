"""Input safety helpers.

Transcripts and OCR text are untrusted.  We detect common prompt-injection
phrasing for *flagging only* (the text is never executed or allowed to change
the system prompt) and serialize source content as JSON data inside prompts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List

_INJECTION_PATTERNS = [
    (r"(?i)\bignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions?\b", "ignore_instructions"),
    (r"(?i)\bdisregard\s+(the\s+)?(system|developer|previous)\b", "disregard_instructions"),
    (r"(?i)\b(system|developer)\s+prompt\b", "system_prompt_reference"),
    (r"(?i)\byou\s+are\s+now\b", "role_change"),
    (r"(?i)\b(mark|approve|classify|triage)\b.{0,40}\b(safe|routine|low.?risk|no care)\b", "force_reassurance"),
    (r"(?i)\b(no|not)\s+(need|require)\s+(for\s+)?(medical|doctor|hospital|care)\b", "force_reassurance"),
    (r"(?i)\b(reveal|show|print|leak)\b.{0,30}\b(api.?key|secret|password|prompt)\b", "secret_exfiltration"),
    (r"(?i)\b(curl|wget|fetch|http://|https://)\b", "external_action"),
    (r"(?i)\bexecute\s+(this\s+)?(command|tool|code)\b", "tool_execution"),
    (r"(?i)\bdelete\s+(all\s+)?(records?|data|case)\b", "destructive_instruction"),
    (r"(?i)(निर्देशों?\s+को\s+अनदेखा|सिस्टम\s+प्रॉम्प्ट|पिछले\s+निर्देश)", "ignore_instructions_hi"),
    (r"(?i)\b(ignore\s+karo|system\s+prompt\s+dekho)\b", "ignore_instructions_hinglish"),
]


def detect_prompt_injection(text: str) -> List[str]:
    hits: List[str] = []
    for pattern, code in _INJECTION_PATTERNS:
        if re.search(pattern, text or ""):
            if code not in hits:
                hits.append(code)
    return hits


def prompt_safe_sources(sources: Iterable[Any]) -> str:
    """Serialize sources as JSON so quotes cannot break out of a prompt section.

    STT sources keep the native transcript in ``text`` and may carry an English
    translation in ``raw_metadata.translated_text``.  The translation is
    supplied only as a reading aid; evidence quotes must come from ``text``.
    """
    payload = []
    for source in sources:
        item = {
            "source_id": source.source_id,
            "modality": source.modality.value,
            "language": source.language,
            "text": source.original_text,
        }
        translation = (getattr(source, "raw_metadata", None) or {}).get(
            "translated_text"
        )
        if isinstance(translation, str) and translation.strip():
            item["english_translation"] = translation.strip()
        payload.append(item)
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class InjectionReport:
    detected: bool = False
    codes: List[str] = field(default_factory=list)


def scan_sources(sources: Iterable[Any]) -> InjectionReport:
    codes: List[str] = []
    for source in sources:
        for code in detect_prompt_injection(source.original_text):
            if code not in codes:
                codes.append(code)
    return InjectionReport(detected=bool(codes), codes=codes)
