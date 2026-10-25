"""Tests for the in-process Sarvam STT integration.

Sarvam network calls are mocked; no API key or live model is required.
"""
from __future__ import annotations

import asyncio
import io
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from pydub import AudioSegment

from app.config import Settings
from app.intake import LocalSttService, SttTimeoutError, safe_intake_error
from app.main import create_app
from app.model.provider import MockProvider
from STT_translation.errors import (
    AudioDecodeError,
    TranscriptionError,
    TranslationError,
)
from STT_translation.schemas import Language
from STT_translation.stt import speech_to_language
from tests.conftest import CLINICIAN, FIXED_KEY, INTAKE, RULES, consent, new_case

AUDIO_BYTES = b"RIFF....WAVEfmt ....synthetic"
TRANSCRIPT = "मुझे दो दिन से सीने में तेज दर्द हो रहा है"
TRANSLATION = "I have been having severe chest pain for two days"


def make_test_settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings(
        data_dir=tmp_path,
        db_path=tmp_path / "careflow.db",
        rules_path=RULES,
        llm_provider="mock",
        processing_mode="sync",
        encryption_key=FIXED_KEY,
        model_backoff_s=0.0,
        max_model_retries=1,
        max_upload_bytes=1024 * 1024,
        stt_timeout_s=5.0,
    )
    return settings.with_overrides(**overrides)


def stt_payload(target_language: str = "en-IN") -> Dict[str, Any]:
    return {
        "transcription": TRANSCRIPT,
        "translated_text": TRANSLATION,
        "target_language": target_language,
        "chunks": [
            {
                "index": 0,
                "start_seconds": 0.5,
                "end_seconds": 4.2,
                "transcription": TRANSCRIPT,
                "translated_text": TRANSLATION,
                "language_code": "hi-IN",
                "language_probability": 0.96,
                "speech_to_text_request_id": "stt_req_1",
                "translation_request_id": "tr_req_1",
            }
        ],
    }


def make_client(tmp_path: Path, **settings_overrides) -> TestClient:
    app = create_app(
        settings=make_test_settings(tmp_path, **settings_overrides),
        provider=MockProvider(),
    )
    return TestClient(app)


def upload_audio(client: TestClient, case_id: str, *, target_language: str = "en-IN"):
    return client.post(
        f"/api/cases/{case_id}/intake/audio"
        f"?auto_process=false&target_language={target_language}",
        headers=CLINICIAN,
        files={"file": ("audio1.ogg", io.BytesIO(AUDIO_BYTES), "audio/ogg")},
    )


def test_intake_invokes_speech_to_language_directly(tmp_path, monkeypatch):
    """The background job calls the in-repo function with bytes and metadata."""
    calls: List[Dict[str, Any]] = []

    def fake_speech_to_language(*, audio_bytes, target_language, filename):
        calls.append(
            {
                "audio_bytes": audio_bytes,
                "target_language": target_language,
                "filename": filename,
            }
        )
        return stt_payload(target_language)

    monkeypatch.setattr(
        "STT_translation.stt.speech_to_language", fake_speech_to_language
    )

    client = make_client(tmp_path)
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_direct_01")
    consent(client, case_id, headers=INTAKE)

    resp = upload_audio(client, case_id)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    doc_id = resp.json()["document_id"]

    job = client.get(
        f"/api/cases/{case_id}/intake/jobs/{job_id}", headers=CLINICIAN
    ).json()
    assert job["stage"] == "complete"
    assert job["status"] == "complete"
    assert job["error"] is None

    assert len(calls) == 1
    assert calls[0]["audio_bytes"] == AUDIO_BYTES
    assert calls[0]["filename"] == "audio1.ogg"
    assert calls[0]["target_language"] == "en-IN"

    # Original transcript and chunk metadata survive into persisted sources.
    sources = client.app.state.db.get_sources(case_id, client.app.state.cipher)
    stt_sources = [s for s in sources if s.modality.value == "stt"]
    assert len(stt_sources) == 1
    source = stt_sources[0]
    assert source.original_text == TRANSCRIPT
    assert source.language == "hi"
    assert source.audio_start == 0.5
    assert source.audio_end == 4.2
    assert source.confidence == pytest.approx(0.96)
    assert source.document_ref == doc_id
    assert source.raw_metadata["translated_text"] == TRANSLATION
    assert source.raw_metadata["document_id"] == doc_id

    document = client.app.state.db.get_document(doc_id)
    assert document["original_name"] == "audio1.ogg"


def test_audio_intake_makes_no_http_call_to_stt_service(tmp_path, monkeypatch):
    """Regression: web audio intake must not use an HTTP client / port 8001."""

    class NoHttpClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("audio intake must not create an HTTP client")

    monkeypatch.setattr("app.intake.httpx.AsyncClient", NoHttpClient)
    monkeypatch.setattr(
        "STT_translation.stt.speech_to_language",
        lambda **kwargs: stt_payload(kwargs["target_language"]),
    )

    client = make_client(tmp_path)
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_nohttp_01")
    consent(client, case_id, headers=INTAKE)

    resp = upload_audio(client, case_id)
    assert resp.status_code == 202
    job = client.get(
        f"/api/cases/{case_id}/intake/jobs/{resp.json()['job_id']}",
        headers=CLINICIAN,
    ).json()
    assert job["status"] == "complete", job


def test_target_language_and_filename_are_passed_through(tmp_path, monkeypatch):
    calls: List[Dict[str, Any]] = []

    def fake_speech_to_language(*, audio_bytes, target_language, filename):
        calls.append({"target_language": target_language, "filename": filename})
        return stt_payload(target_language)

    monkeypatch.setattr(
        "STT_translation.stt.speech_to_language", fake_speech_to_language
    )

    client = make_client(tmp_path)
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_lang_01")
    consent(client, case_id, headers=INTAKE)

    resp = upload_audio(client, case_id, target_language="hi-IN")
    assert resp.status_code == 202
    job = client.get(
        f"/api/cases/{case_id}/intake/jobs/{resp.json()['job_id']}",
        headers=CLINICIAN,
    ).json()
    assert job["status"] == "complete", job
    assert calls == [{"target_language": "hi-IN", "filename": "audio1.ogg"}]


def test_empty_transcript_marks_job_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "STT_translation.stt.speech_to_language",
        lambda **kwargs: {
            "transcription": "",
            "translated_text": "",
            "target_language": "en-IN",
            "chunks": [],
        },
    )

    client = make_client(tmp_path)
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_empty_01")
    consent(client, case_id, headers=INTAKE)

    resp = upload_audio(client, case_id)
    assert resp.status_code == 202
    job = client.get(
        f"/api/cases/{case_id}/intake/jobs/{resp.json()['job_id']}",
        headers=CLINICIAN,
    ).json()
    assert job["stage"] == "failed"
    assert job["status"] == "failed"
    assert "no transcript text" in job["error"]

    # The encrypted original is retained for retry/inspection.
    doc_resp = client.get(
        f"/api/documents/{resp.json()['document_id']}", headers=CLINICIAN
    )
    assert doc_resp.status_code == 200


def test_sarvam_failure_is_reported_without_leaking_details(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise TranscriptionError("upstream response contained sensitive-content")

    monkeypatch.setattr("STT_translation.stt.speech_to_language", fail)

    client = make_client(tmp_path)
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_fail_01")
    consent(client, case_id, headers=INTAKE)

    resp = upload_audio(client, case_id)
    assert resp.status_code == 202
    job = client.get(
        f"/api/cases/{case_id}/intake/jobs/{resp.json()['job_id']}",
        headers=CLINICIAN,
    ).json()
    assert job["status"] == "failed"
    assert job["error"] == "Audio transcription failed: Speech transcription failed. Please retry."
    assert "sensitive-content" not in job["error"]


def test_safe_error_messages_do_not_leak_details():
    assert (
        safe_intake_error(TranscriptionError("secret upstream body"))
        == "Speech transcription failed. Please retry."
    )
    assert (
        safe_intake_error(TranslationError("secret upstream body"))
        == "Transcript translation failed. Please retry."
    )
    assert (
        safe_intake_error(AudioDecodeError("secret filename"))
        == "Audio could not be decoded. Supported formats: "
        "WAV, MP3, M4A, OGG, FLAC, WebM."
    )
    assert (
        safe_intake_error(SttTimeoutError("secret timing"))
        == "Speech processing timed out. Please retry."
    )


def test_local_stt_service_timeout_wraps_cleanly(monkeypatch):
    def slow(*args, **kwargs):
        time.sleep(0.2)
        return stt_payload()

    monkeypatch.setattr("app.intake._run_speech_to_language", slow)
    service = LocalSttService(timeout_s=0.01)

    with pytest.raises(SttTimeoutError):
        asyncio.run(service.transcribe(AUDIO_BYTES, filename="audio.ogg"))


class _FakeSttResponse:
    def __init__(self, language_code: str = "hi-IN"):
        self.transcript = TRANSCRIPT
        self.language_code = language_code
        self.language_probability = 0.97
        self.request_id = "stt_req_direct"


class _FakeTranslationResponse:
    translated_text = TRANSLATION
    request_id = "tr_req_direct"


class _FakeSpeechToText:
    def __init__(self, language_code: str = "hi-IN"):
        self.language_code = language_code
        self.calls: List[Dict[str, Any]] = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeSttResponse(self.language_code)


class _FakeText:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def translate(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeTranslationResponse()


class _FakeSarvam:
    def __init__(self, language_code: str = "hi-IN"):
        self.speech_to_text = _FakeSpeechToText(language_code)
        self.text = _FakeText()


def _wav_bytes(duration_ms: int = 800) -> bytes:
    buffer = io.BytesIO()
    AudioSegment.silent(duration=duration_ms).export(buffer, format="wav")
    return buffer.getvalue()


def test_speech_to_language_short_recording_metadata(monkeypatch):
    fake = _FakeSarvam()
    monkeypatch.setattr("STT_translation.stt.get_sarvam_client", lambda: fake)

    result = speech_to_language(_wav_bytes(), "en-IN", "short.wav")

    assert result["transcription"] == TRANSCRIPT
    assert result["translated_text"] == TRANSLATION
    assert result["target_language"] == "en-IN"
    assert len(result["chunks"]) == 1
    chunk = result["chunks"][0]
    assert chunk["index"] == 0
    assert chunk["transcription"] == TRANSCRIPT
    assert chunk["translated_text"] == TRANSLATION
    assert chunk["language_code"] == "hi-IN"
    assert chunk["language_probability"] == pytest.approx(0.97)
    assert chunk["speech_to_text_request_id"] == "stt_req_direct"
    assert chunk["translation_request_id"] == "tr_req_direct"
    assert chunk["start_seconds"] == 0.0
    assert chunk["end_seconds"] == pytest.approx(0.8, abs=0.05)

    stt_call = fake.speech_to_text.calls[0]
    assert stt_call["model"] == "saaras:v4"
    assert stt_call["mode"] == "codemix"
    assert stt_call["language_code"] == "unknown"

    translate_call = fake.text.calls[0]
    assert translate_call["input"] == TRANSCRIPT
    assert translate_call["source_language_code"] == "hi-IN"
    assert translate_call["target_language_code"] == "en-IN"


def test_speech_to_language_skips_translation_for_matching_language(monkeypatch):
    fake = _FakeSarvam(language_code="en-IN")
    monkeypatch.setattr("STT_translation.stt.get_sarvam_client", lambda: fake)

    result = speech_to_language(_wav_bytes(), "en-IN", "short.wav")

    assert result["translated_text"] == TRANSCRIPT
    assert result["chunks"][0]["translation_request_id"] is None
    assert fake.text.calls == []


def test_speech_to_language_rejects_empty_audio():
    with pytest.raises(ValueError):
        speech_to_language(b"", "en-IN", "audio1.ogg")


def test_speech_to_language_wraps_decode_error():
    with pytest.raises(AudioDecodeError):
        speech_to_language(b"this is not audio", "en-IN", "broken.ogg")


def test_ui_target_languages_match_schema():
    index_path = (
        Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
    )
    html = index_path.read_text(encoding="utf-8")
    select = re.search(
        r'<select id="audio-target-lang">(.*?)</select>', html, re.S
    )
    assert select is not None
    options = set(re.findall(r'<option value="([^"]+)"', select.group(1)))
    assert options == {language.value for language in Language}


def test_standalone_api_still_reuses_the_same_function(monkeypatch):
    from STT_translation.api import app as stt_app

    calls: List[Dict[str, Any]] = []

    def fake_speech_to_language(*, audio_bytes, target_language, filename):
        calls.append(
            {
                "audio_bytes": audio_bytes,
                "target_language": target_language,
                "filename": filename,
            }
        )
        return stt_payload(target_language)

    monkeypatch.setattr(
        "STT_translation.stt.speech_to_language", fake_speech_to_language
    )

    client = TestClient(stt_app)
    resp = client.post(
        "/speech-to-language",
        files={"audio": ("sample.wav", b"not-empty-audio", "audio/wav")},
        data={"target_language": "hi-IN"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["target_language"] == "hi-IN"
    assert calls == [
        {
            "audio_bytes": b"not-empty-audio",
            "target_language": "hi-IN",
            "filename": "sample.wav",
        }
    ]


def test_standalone_api_accepts_ui_languages():
    from STT_translation.api import app as stt_app

    client = TestClient(stt_app)
    for language in Language:
        resp = client.post(
            "/speech-to-language",
            files={"audio": ("empty.ogg", b"", "audio/ogg")},
            data={"target_language": language.value},
        )
        # Valid language gets past validation; empty audio is rejected at 400.
        assert resp.status_code == 400, (language, resp.text)

    invalid = client.post(
        "/speech-to-language",
        files={"audio": ("empty.ogg", b"", "audio/ogg")},
        data={"target_language": "xx-IN"},
    )
    assert invalid.status_code == 422
