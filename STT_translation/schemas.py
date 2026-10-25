from enum import Enum

from pydantic import BaseModel, Field


class Language(str, Enum):
    """
    Supported audio target languages, using Sarvam BCP-47 codes.

    These are the languages CareFlow can both translate into and speak with
    Bulbul TTS.  Speech is transcribed in the detected source language
    (``saaras`` runs in ``codemix`` mode and auto-detects), then translated
    separately with ``sarvam-translate`` when the detected language differs
    from the target.  Keep this enum in sync with the ``audio-target-lang`` and
    ``patient-language`` options in ``app/static/index.html``.
    """

    BENGALI = "bn-IN"
    ENGLISH = "en-IN"
    GUJARATI = "gu-IN"
    HINDI = "hi-IN"
    KANNADA = "kn-IN"
    MALAYALAM = "ml-IN"
    MARATHI = "mr-IN"
    ODIA = "od-IN"
    PUNJABI = "pa-IN"
    TAMIL = "ta-IN"
    TELUGU = "te-IN"


class SpeechTranslationRequest(BaseModel):
    target_language: Language = Field(
        default=Language.ENGLISH,
        description="Language to translate the audio into",
    )


class SpeechTranslationInput(BaseModel):
    """Strict input contract for the STT worker."""

    filename: str = Field(min_length=1)
    target_language: Language = Language.ENGLISH


class TranscriptChunk(BaseModel):
    """One silence-aware chunk of the transcription/translation result."""

    index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    transcription: str | None = None
    translated_text: str
    language_code: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    speech_to_text_request_id: str | None = None
    translation_request_id: str | None = None
    request_id: str | None = None


class SpeechTranslationResult(BaseModel):
    """Strict output contract for the STT worker."""

    transcription: str | None = None
    translated_text: str
    target_language: str
    chunks: list[TranscriptChunk]
    document_id: str | None = None
