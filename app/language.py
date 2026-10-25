"""Lightweight language handling.

This is intentionally conservative: declared language metadata is preserved,
script heuristics are used when metadata is missing, and a mismatch is surfaced
as a warning instead of being silently overwritten.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from .config import SUPPORTED_LANGUAGES
from .schemas import LanguageStatus

_LANGUAGE_ALIASES = {
    "eng": "en",
    "en-in": "en",
    "english": "en",
    "hin": "hi",
    "hi-in": "hi",
    "hindi": "hi",
    "hi-latn": "hinglish",
    "hinglish": "hinglish",
    "en-latn": "hinglish",
    "tam": "ta",
    "tamil": "ta",
    "tel": "te",
    "telugu": "te",
    "ben": "bn",
    "bengali": "bn",
    "guj": "gu",
    "gujarati": "gu",
    "kan": "kn",
    "kannada": "kn",
    "mal": "ml",
    "malayalam": "ml",
    "pan": "pa",
    "punjabi": "pa",
    "ori": "or",
    "odia": "or",
    "oriya": "or",
    "od": "or",
    "od-in": "or",
    "or-in": "or",
    "asm": "as",
    "assamese": "as",
    "urd": "ur",
    "urdu": "ur",
    "ur-in": "ur",
    "nep": "ne",
    "nepali": "ne",
    "ne-in": "ne",
    "kok": "kok",
    "konkani": "kok",
    "kok-in": "kok",
    "kas": "ks",
    "kashmiri": "ks",
    "ks-in": "ks",
    "snd": "sd",
    "sindhi": "sd",
    "sd-in": "sd",
    "san": "sa",
    "sanskrit": "sa",
    "sa-in": "sa",
    "sat": "sat",
    "santali": "sat",
    "sat-in": "sat",
    "mni": "mni",
    "manipuri": "mni",
    "mni-in": "mni",
    "brx": "brx",
    "bodo": "brx",
    "brx-in": "brx",
    "mai": "mai",
    "maithili": "mai",
    "mai-in": "mai",
    "doi": "doi",
    "dogri": "doi",
    "doi-in": "doi",
}

_SCRIPT_RANGES = [
    ("hi", r"[\u0900-\u097F]"),  # Devanagari
    ("bn", r"[\u0980-\u09FF]"),
    ("pa", r"[\u0A00-\u0A7F]"),
    ("gu", r"[\u0A80-\u0AFF]"),
    ("or", r"[\u0B00-\u0B7F]"),
    ("ta", r"[\u0B80-\u0BFF]"),
    ("te", r"[\u0C00-\u0C7F]"),
    ("kn", r"[\u0C80-\u0CFF]"),
    ("ml", r"[\u0D00-\u0D7F]"),
]

_HINGLISH_MARKERS = {
    "hai",
    "nahi",
    "nahin",
    "dard",
    "bukhar",
    "khansi",
    "saans",
    "sans",
    "chakkar",
    "ulti",
    "pet",
    "kamzori",
    "thakaan",
    "kal",
    "aaj",
    "subah",
    "raat",
    "din",
    "se",
    "me",
    "mein",
    "ka",
    "ki",
    "ko",
    "ho",
    "raha",
    "rahi",
    "bahut",
    "thoda",
    "pata",
    "shayad",
    "doctor",
    "dawai",
    "goli",
    "bp",
    "sugar",
}


def normalize_language_code(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    code = str(value).strip().lower().replace("_", "-")
    if code in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[code]
    if code in SUPPORTED_LANGUAGES:
        return code
    primary = code.split("-")[0]
    if primary in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[primary]
    if primary in SUPPORTED_LANGUAGES:
        return primary
    return code


def heuristic_language(text: str) -> Tuple[str, float]:
    text = text or ""
    for code, pattern in _SCRIPT_RANGES:
        if re.search(pattern, text):
            return code, 0.85
    latin_words = re.findall(r"[A-Za-z]+", text.lower())
    if latin_words:
        markers = sum(1 for word in latin_words if word in _HINGLISH_MARKERS)
        ratio = markers / max(len(latin_words), 1)
        if markers >= 2 and ratio >= 0.12:
            return "hinglish", 0.65
        return "en", 0.6
    return "unknown", 0.0


def resolve_language(
    text: str, declared: Optional[str] = None
) -> Tuple[str, LanguageStatus, list[str]]:
    warnings: list[str] = []
    declared_norm = normalize_language_code(declared)
    detected, confidence = heuristic_language(text)

    if declared_norm:
        if declared_norm not in SUPPORTED_LANGUAGES:
            warnings.append(f"declared_language_not_in_supported_set:{declared_norm}")
            return declared_norm, LanguageStatus.unsupported, warnings
        if (
            detected != "unknown"
            and detected != declared_norm
            and not (
                declared_norm == "hi"
                and detected == "hi"
            )
        ):
            # Script detection is reliable for Indic scripts; Latin-script
            # Hinglish vs English is only heuristic, so mismatch is advisory.
            warnings.append(
                f"language_metadata_mismatch:declared={declared_norm}:heuristic={detected}:conf={confidence:.2f}"
            )
        return declared_norm, LanguageStatus.declared, warnings

    if detected == "unknown":
        return "unknown", LanguageStatus.unknown, ["language_undetermined"]
    return detected, LanguageStatus.heuristic, warnings
