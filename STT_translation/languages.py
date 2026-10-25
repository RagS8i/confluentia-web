"""Language metadata shared by the Sarvam speech, translation and voice paths.

Sarvam uses BCP-47 codes (``hi-IN``), while CareFlow stores compact language
codes (``hi``).  Keep the conversion and the voice-capable subset in one place
so the API, database and UI cannot silently drift apart.
"""
from __future__ import annotations

from typing import Optional

# Languages exposed by the audio intake selector and supported by Bulbul TTS.
BCP47_BY_SHORT = {
    "bn": "bn-IN",
    "en": "en-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "or": "od-IN",  # Sarvam spells Odia "od"; CareFlow historically used "or".
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}

SHORT_BY_BCP47 = {value: key for key, value in BCP47_BY_SHORT.items()}

LANGUAGE_NAMES = {
    "bn-IN": "Bengali",
    "en-IN": "English",
    "gu-IN": "Gujarati",
    "hi-IN": "Hindi",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
}

# Hinglish is a code-mixed input style, not a TTS/translation target.  The
# closest native target is Hindi, while the original Roman transcript remains
# preserved in provenance.
FOLLOWUP_LANGUAGE_CODES = frozenset(BCP47_BY_SHORT) | {"hinglish"}

_ALIASES = {
    "en": "en-IN",
    "en-in": "en-IN",
    "eng": "en-IN",
    "english": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "hin": "hi-IN",
    "hindi": "hi-IN",
    "hinglish": "hi-IN",
    "hi-latn": "hi-IN",
    "bn": "bn-IN",
    "bn-in": "bn-IN",
    "bengali": "bn-IN",
    "gu": "gu-IN",
    "gu-in": "gu-IN",
    "gujarati": "gu-IN",
    "kn": "kn-IN",
    "kn-in": "kn-IN",
    "kannada": "kn-IN",
    "ml": "ml-IN",
    "ml-in": "ml-IN",
    "malayalam": "ml-IN",
    "mr": "mr-IN",
    "mr-in": "mr-IN",
    "marathi": "mr-IN",
    "or": "od-IN",
    "or-in": "od-IN",
    "od": "od-IN",
    "od-in": "od-IN",
    "odia": "od-IN",
    "oriya": "od-IN",
    "pa": "pa-IN",
    "pa-in": "pa-IN",
    "punjabi": "pa-IN",
    "ta": "ta-IN",
    "ta-in": "ta-IN",
    "tamil": "ta-IN",
    "te": "te-IN",
    "te-in": "te-IN",
    "telugu": "te-IN",
    "as": "as-IN",
    "as-in": "as-IN",
    "assamese": "as-IN",
    "ur": "ur-IN",
    "ur-in": "ur-IN",
    "urdu": "ur-IN",
    "ne": "ne-IN",
    "ne-in": "ne-IN",
    "nepali": "ne-IN",
    "kok": "kok-IN",
    "kok-in": "kok-IN",
    "ks": "ks-IN",
    "ks-in": "ks-IN",
    "sd": "sd-IN",
    "sd-in": "sd-IN",
    "sa": "sa-IN",
    "sa-in": "sa-IN",
    "sat": "sat-IN",
    "sat-in": "sat-IN",
    "mni": "mni-IN",
    "mni-in": "mni-IN",
    "brx": "brx-IN",
    "brx-in": "brx-IN",
    "mai": "mai-IN",
    "mai-in": "mai-IN",
    "doi": "doi-IN",
    "doi-in": "doi-IN",
}

# Sarvam speech/translation accepts all of these (with ``auto`` for source).
SARVAM_TRANSLATE_LANGUAGES = frozenset(_ALIASES.values())

TTS_LANGUAGES = frozenset(BCP47_BY_SHORT.values())


def normalize_bcp47(value: Optional[str]) -> Optional[str]:
    """Return a Sarvam BCP-47 language code, or ``None`` when unrecognised."""
    if value is None:
        return None
    code = str(value).strip().replace("_", "-").lower()
    if not code:
        return None
    if code in _ALIASES:
        return _ALIASES[code]
    primary = code.split("-")[0]
    return _ALIASES.get(primary)


def short_code(value: Optional[str]) -> Optional[str]:
    """Return CareFlow's compact code (``hi``/``or``) for a language value."""
    if value is None:
        return None
    raw = str(value).strip().replace("_", "-").lower()
    if raw == "hinglish" or raw == "hi-latn":
        return "hinglish"
    bcp47 = normalize_bcp47(value)
    if bcp47 is None:
        return None
    return SHORT_BY_BCP47.get(bcp47)


def language_name(value: Optional[str]) -> str:
    bcp47 = normalize_bcp47(value)
    return LANGUAGE_NAMES.get(bcp47 or "", str(value or "unknown"))


def tts_supported(value: Optional[str]) -> bool:
    return normalize_bcp47(value) in TTS_LANGUAGES
