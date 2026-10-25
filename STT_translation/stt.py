import base64
import math
import os
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from sarvamai import SarvamAI

try:  # Imported as a package (STT_translation.stt).
    from .errors import (
        AudioDecodeError,
        LanguageDetectionError,
        TextToSpeechError,
        TranscriptionError,
        TranslationError,
        UnsupportedLanguageError,
    )
    from .languages import (
        TTS_LANGUAGES,
        language_name,
        normalize_bcp47,
        short_code,
        tts_supported,
    )
except ImportError:  # Run with --app-dir STT_translation.
    from errors import (  # type: ignore[no-redef]
        AudioDecodeError,
        LanguageDetectionError,
        TextToSpeechError,
        TranscriptionError,
        TranslationError,
        UnsupportedLanguageError,
    )
    from languages import (  # type: ignore[no-redef]
        TTS_LANGUAGES,
        language_name,
        normalize_bcp47,
        short_code,
        tts_supported,
    )

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()

_client = None


def get_sarvam_client() -> SarvamAI:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY missing from environment or .env")
    _client = SarvamAI(api_subscription_key=api_key)
    return _client

# Sarvam's REST speech-to-text endpoint accepts audio under 30 seconds.
# Aim for ~26 s chunks so cuts land on silence well before that limit.
CHUNK_TARGET_MS = 26_000
CHUNK_MAX_MS = 28_000
MIN_SILENCE_MS = 300
SILENCE_THRESHOLD_DB = 16

STT_MODEL = "saaras:v4"
STT_LANGUAGE = "unknown"  # Auto-detect the spoken language.

# sarvam-translate:v1 covers every language exposed in schemas.Language.
# Switch to "mayura:v1" if you only need the 11 core languages and want its
# extra modes, but note it rejects the newer codes (as-IN, brx-IN, ...).
TRANSLATE_MODEL = os.getenv("SARVAM_TRANSLATE_MODEL", "sarvam-translate:v1")

TTS_MODEL = os.getenv("CAREFLOW_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("CAREFLOW_TTS_SPEAKER", "shubh")
TTS_MAX_CHARS = 2_300  # Bulbul accepts at most 2500 characters per request.
TRANSLATE_MAX_CHARS = 900


def _speech_ranges(audio: AudioSegment) -> list[tuple[int, int]]:
    """Return non-silent (start_ms, end_ms) ranges in the audio."""

    duration_ms = len(audio)
    if duration_ms == 0:
        return []

    # dBFS is -inf for completely silent audio, which is not a usable
    # threshold for detect_nonsilent.
    if not math.isfinite(audio.dBFS):
        return [(0, duration_ms)]

    ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=audio.dBFS - SILENCE_THRESHOLD_DB,
    )
    return ranges or [(0, duration_ms)]


def _bounded_ranges(
    ranges: list[tuple[int, int]],
    duration_ms: int,
) -> list[tuple[int, int]]:
    """Clamp ranges and hard-split any continuous speech longer than the limit."""

    bounded: list[tuple[int, int]] = []

    for start, end in ranges:
        start = max(0, min(start, duration_ms))
        end = max(start, min(end, duration_ms))

        while end - start > CHUNK_MAX_MS:
            bounded.append((start, start + CHUNK_MAX_MS))
            start += CHUNK_MAX_MS

        if end > start:
            bounded.append((start, end))

    return bounded


def _split_on_silence(audio: AudioSegment) -> list[tuple[int, int, AudioSegment]]:
    """
    Split audio into chunks of at most CHUNK_MAX_MS.

    Chunks are cut at natural pauses near CHUNK_TARGET_MS. Only continuous
    speech longer than CHUNK_MAX_MS is split at a hard boundary, since the
    Sarvam REST endpoint cannot accept anything longer.

    Returns (start_ms, end_ms, segment) tuples.
    """

    duration_ms = len(audio)
    if duration_ms <= CHUNK_MAX_MS:
        return [(0, duration_ms, audio)]

    ranges = _bounded_ranges(_speech_ranges(audio), duration_ms)
    if not ranges:
        return [(0, duration_ms, audio)]

    groups: list[tuple[int, int]] = []
    group_start, group_end = ranges[0]

    for start, end in ranges[1:]:
        fits_limit = end - group_start <= CHUNK_MAX_MS
        needs_more_audio = group_end - group_start < CHUNK_TARGET_MS

        if fits_limit and needs_more_audio:
            group_end = end
            continue

        groups.append((group_start, group_end))
        group_start, group_end = start, end

    groups.append((group_start, group_end))

    return [
        (start, end, audio[start:end])
        for start, end in groups
        if end > start
    ]


def _export_chunk(chunk: AudioSegment, path: Path) -> None:
    """Export a chunk as 16 kHz mono 16-bit PCM WAV."""

    chunk = (
        chunk.set_frame_rate(16_000)
        .set_channels(1)
        .set_sample_width(2)
    )
    chunk.export(path, format="wav")


def _decode_audio(audio_bytes: bytes, filename: str) -> AudioSegment:
    """Decode uploaded audio with a safe error that never includes the filename."""
    if not audio_bytes:
        raise ValueError("Audio file is empty.")
    suffix = Path(filename).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="speech-audio-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        input_path.write_bytes(audio_bytes)
        try:
            return AudioSegment.from_file(input_path)
        except Exception as exc:
            # Filenames may contain patient identifiers; omit them.
            raise AudioDecodeError(
                f"Could not decode audio input (extension {suffix})."
            ) from exc


def _text_chunks(text: str, limit: int) -> list[str]:
    """Split text near sentence/discourse boundaries without losing content."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= limit:
        return [cleaned]

    sentences = re.split(r"(?<=[.!?।॥\n])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                sentence[index : index + limit]
                for index in range(0, len(sentence), limit)
            )
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def translate_text(
    text: str,
    target_language: str,
    *,
    source_language: str = "auto",
) -> dict:
    """Translate plain text with Sarvam, preserving sentence boundaries.

    Returns the translated text plus request IDs.  The caller decides how to
    cache the result; the function itself performs no storage.
    """
    cleaned = (text or "").strip()
    target = normalize_bcp47(target_language)
    if target is None:
        raise UnsupportedLanguageError(
            f"Unsupported translation target: {target_language}"
        )
    if not cleaned:
        return {
            "translated_text": "",
            "target_language": target,
            "request_ids": [],
        }

    source = normalize_bcp47(source_language) or "auto"
    if source == target:
        return {
            "translated_text": cleaned,
            "target_language": target,
            "request_ids": [],
        }

    client = get_sarvam_client()
    translated_parts: list[str] = []
    request_ids: list[str] = []
    for chunk in _text_chunks(cleaned, TRANSLATE_MAX_CHARS):
        try:
            response = client.text.translate(
                input=chunk,
                source_language_code=source,
                target_language_code=target,
                model=TRANSLATE_MODEL,
            )
        except Exception as exc:
            raise TranslationError("Sarvam translation failed.") from exc
        translated = (response.translated_text or "").strip()
        if translated:
            translated_parts.append(translated)
        if getattr(response, "request_id", None):
            request_ids.append(response.request_id)

    if not translated_parts:
        raise TranslationError("Sarvam translation returned no text.")
    return {
        "translated_text": " ".join(translated_parts),
        "target_language": target,
        "request_ids": request_ids,
    }


def detect_language(audio_bytes: bytes, filename: str = "sample.wav") -> dict:
    """Detect the spoken language from a short audio sample using Saaras v4."""
    audio = _decode_audio(audio_bytes, filename)
    client = get_sarvam_client()

    for index, (_start_ms, _end_ms, segment) in enumerate(
        _split_on_silence(audio)[:3]
    ):
        with tempfile.TemporaryDirectory(prefix="speech-detect-") as temp_dir:
            chunk_path = Path(temp_dir) / f"sample_{index:04d}.wav"
            _export_chunk(segment, chunk_path)
            try:
                with chunk_path.open("rb") as audio_file:
                    response = client.speech_to_text.transcribe(
                        file=audio_file,
                        model=STT_MODEL,
                        mode="codemix",
                        language_code=STT_LANGUAGE,
                    )
            except Exception as exc:
                raise LanguageDetectionError(
                    "Sarvam language detection failed."
                ) from exc

        transcript = (response.transcript or "").strip()
        detected = normalize_bcp47(response.language_code)
        if transcript and detected:
            return {
                "language_code": detected,
                "language": short_code(detected),
                "language_name": language_name(detected),
                "language_probability": response.language_probability,
                "tts_supported": tts_supported(detected),
                "speech_to_text_request_id": response.request_id,
            }

    raise LanguageDetectionError("No speech was detected in the audio sample.")


def text_to_speech(
    text: str,
    language_code: str,
    *,
    speaker: str | None = None,
    model: str | None = None,
) -> dict:
    """Synthesize speech with Sarvam Bulbul and return MP3 bytes.

    Long text is split below Bulbul's per-request limit and the MP3 frames are
    concatenated.  Only the 11 languages in ``languages.BCP47_BY_SHORT`` are
    advertised as voice-capable.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Text is empty.")
    target = normalize_bcp47(language_code)
    if target is None or target not in TTS_LANGUAGES:
        raise UnsupportedLanguageError(
            f"Voice playback is not available for {language_code}."
        )

    client = get_sarvam_client()
    chunks = _text_chunks(cleaned, TTS_MAX_CHARS)
    audio_parts: list[bytes] = []
    request_ids: list[str] = []
    for chunk in chunks:
        try:
            response = client.text_to_speech.convert(
                text=chunk,
                language_code=target,
                speaker=speaker or TTS_SPEAKER,
                model=model or TTS_MODEL,
                output_audio_codec="mp3",
            )
        except Exception as exc:
            raise TextToSpeechError("Sarvam text-to-speech failed.") from exc
        audios = getattr(response, "audios", None) or []
        if not audios:
            raise TextToSpeechError("Sarvam text-to-speech returned no audio.")
        for encoded in audios:
            try:
                audio_parts.append(base64.b64decode(encoded))
            except Exception as exc:
                raise TextToSpeechError(
                    "Sarvam text-to-speech returned malformed audio."
                ) from exc
        if getattr(response, "request_id", None):
            request_ids.append(response.request_id)

    return {
        "audio_bytes": b"".join(audio_parts),
        "mime": "audio/mpeg",
        "language_code": target,
        "request_ids": request_ids,
        "chunks": len(chunks),
    }


def speech_to_language(
    audio_bytes: bytes,
    target_language: str,
    filename: str = "audio.wav",
) -> dict:
    """
    Transcribe an audio file and translate it into the target language.

    Args:
        audio_bytes: Raw contents of the uploaded audio file.
        target_language: Sarvam BCP-47 target code, e.g. "hi-IN".
        filename: Original filename, used only to infer the input format.

    Returns:
        A dict with the full transcription, full translated text, target
        language, and per-chunk Sarvam outputs (detected language, language
        probability, request IDs, and timings).

    Raises:
        AudioDecodeError: the bytes could not be decoded as audio.
        TranscriptionError: Sarvam speech-to-text failed for a chunk.
        TranslationError: Sarvam translation failed for a chunk.
    """

    if not audio_bytes:
        raise ValueError("Audio file is empty.")

    suffix = Path(filename).suffix or ".wav"

    with tempfile.TemporaryDirectory(prefix="speech-translation-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / f"input{suffix}"
        input_path.write_bytes(audio_bytes)

        try:
            audio = AudioSegment.from_file(input_path)
        except Exception as exc:
            # Do not include the filename: uploads may carry patient identifiers.
            raise AudioDecodeError(
                f"Could not decode audio input (extension {suffix})."
            ) from exc

        chunk_results = []
        transcriptions = []
        translations = []

        for index, (start_ms, end_ms, segment) in enumerate(
            _split_on_silence(audio)
        ):
            chunk_path = temp_path / f"chunk_{index:04d}.wav"
            _export_chunk(segment, chunk_path)

            # Resolve the client before per-chunk work so a missing API key is
            # reported as a configuration error, not as a transcription outage.
            client = get_sarvam_client()

            # Step 1: speech -> original-language text, with autodetection.
            try:
                with chunk_path.open("rb") as audio_file:
                    stt_response = client.speech_to_text.transcribe(
                        file=audio_file,
                        model=STT_MODEL,
                        mode="codemix",
                        language_code=STT_LANGUAGE,
                    )
            except Exception as exc:
                raise TranscriptionError(
                    f"Sarvam speech-to-text failed for chunk {index}."
                ) from exc

            transcript = (stt_response.transcript or "").strip()
            detected_language = stt_response.language_code

            # Step 2: original-language text -> target language.
            translated_text = ""
            translation_request_id = None

            if transcript:
                if detected_language == target_language:
                    translated_text = transcript
                else:
                    try:
                        translation_response = client.text.translate(
                            input=transcript,
                            source_language_code=detected_language or "auto",
                            target_language_code=target_language,
                            model=TRANSLATE_MODEL,
                        )
                    except Exception as exc:
                        raise TranslationError(
                            f"Sarvam translation failed for chunk {index}."
                        ) from exc
                    translated_text = (
                        translation_response.translated_text or ""
                    ).strip()
                    translation_request_id = translation_response.request_id

            chunk_results.append(
                {
                    "index": index,
                    "start_seconds": round(start_ms / 1000, 2),
                    "end_seconds": round(end_ms / 1000, 2),
                    "transcription": transcript,
                    "translated_text": translated_text,
                    "language_code": detected_language,
                    "language_probability": stt_response.language_probability,
                    "speech_to_text_request_id": stt_response.request_id,
                    "translation_request_id": translation_request_id,
                }
            )

            if transcript:
                transcriptions.append(transcript)
            if translated_text:
                translations.append(translated_text)

        detected_language = next(
            (
                chunk["language_code"]
                for chunk in chunk_results
                if chunk.get("language_code")
            ),
            None,
        )
        probability = next(
            (
                chunk["language_probability"]
                for chunk in chunk_results
                if chunk.get("language_probability") is not None
            ),
            None,
        )
        return {
            "transcription": " ".join(transcriptions),
            "translated_text": " ".join(translations),
            "target_language": target_language,
            "language": detected_language,
            "language_code": detected_language,
            "language_probability": probability,
            "model": STT_MODEL,
            "chunks": chunk_results,
        }
