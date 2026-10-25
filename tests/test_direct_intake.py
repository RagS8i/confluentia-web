"""Tests for direct audio and document intake with fake STT/OCR clients."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model.provider import MockProvider
from tests.conftest import ADMIN, CLINICIAN, INTAKE, FIXED_KEY, RULES, consent, new_case, patient_headers


class FakeSttClient:
    """Deterministic in-memory STT client for testing."""

    def __init__(self, should_fail: bool = False, failure_message: str = "Sarvam STT connection timed out"):
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.transcribe_calls = []

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        target_language: str = "en-IN",
    ) -> Dict[str, Any]:
        self.transcribe_calls.append({"bytes_len": len(audio_bytes), "filename": filename, "target_language": target_language})
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        return {
            "transcription": "मुझे दो दिन से सीने में तेज दर्द हो रहा है",
            "translated_text": "I have been having severe chest pain for two days",
            "target_language": target_language,
            "chunks": [
                {
                    "index": 0,
                    "start_seconds": 0.5,
                    "end_seconds": 4.2,
                    "transcription": "मुझे दो दिन से सीने में तेज दर्द हो रहा है",
                    "translated_text": "I have been having severe chest pain for two days",
                    "language_code": "hi",
                    "language_probability": 0.96,
                }
            ],
        }


class FakeOcrProcessor:
    """Deterministic in-process OCR processor for testing."""

    def __init__(self, should_fail: bool = False, failure_message: str = "PaddleOCR engine error"):
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.calls = []

    def __call__(
        self,
        document_bytes: bytes,
        filename: str,
        model: str,
        base_url: str,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "bytes_len": len(document_bytes),
                "filename": filename,
                "model": model,
                "base_url": base_url,
            }
        )
        if self.should_fail:
            raise RuntimeError(self.failure_message)
        return {
            "file_name": "temp_paddle_upload.pdf",  # Simulated temporary name to test overriding
            "is_medical": True,
            "document_type": "lab_report",
            "sources": [
                {
                    "block_id": "p1_b1",
                    "text": "Blood Pressure: 160/100 mmHg",
                    "page": 1,
                    "bbox": [20.0, 30.0, 180.0, 50.0],
                    "confidence": 0.97,
                    "unreadable": False,
                },
                {
                    "block_id": "p1_b2",
                    "text": "Troponin I: 0.12 ng/mL (High)",
                    "page": 1,
                    "bbox": [20.0, 60.0, 210.0, 80.0],
                    "confidence": 0.95,
                    "unreadable": False,
                },
            ],
        }


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
        max_upload_bytes=1024 * 1024,  # 1 MB for testing
    )
    return settings.with_overrides(**overrides)


@pytest.fixture
def intake_client_factory(tmp_path):
    clients = []

    def factory(stt_service=None, document_processor=None, **settings_overrides):
        app = create_app(
            settings=make_test_settings(tmp_path, **settings_overrides),
            provider=MockProvider(),
            stt_service=stt_service or FakeSttClient(),
            document_processor=document_processor or FakeOcrProcessor(),
        )
        client = TestClient(app)
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.close()


def test_direct_audio_intake_success_and_provenance(intake_client_factory):
    """Audio intake uploads file, runs STT, preserves original Hindi text and document_ref."""
    fake_stt = FakeSttClient()
    client = intake_client_factory(stt_service=fake_stt)

    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_audio_01")
    consent(client, case_id, headers=INTAKE)

    audio_data = b"RIFF....WAVEfmt ...."  # Synthetic WAV bytes
    files = {"file": ("patient_symptoms.wav", io.BytesIO(audio_data), "audio/wav")}

    resp = client.post(
        f"/api/cases/{case_id}/intake/audio?auto_process=true&target_language=en-IN",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 202
    data = resp.json()
    job_id = data["job_id"]
    doc_id = data["document_id"]
    assert data["kind"] == "audio"
    assert data["stage"] == "queued"
    assert data["status"] == "queued"
    assert data["poll"] == f"/api/cases/{case_id}/intake/jobs/{job_id}"

    # Poll job status
    poll_resp = client.get(f"/api/cases/{case_id}/intake/jobs/{job_id}", headers=CLINICIAN)
    assert poll_resp.status_code == 200
    job_data = poll_resp.json()
    assert job_data["stage"] == "complete"
    assert job_data["status"] == "complete"
    assert job_data["error"] is None

    # Check case detail and sources
    case_resp = client.get(f"/api/cases/{case_id}", headers=CLINICIAN)
    assert case_resp.status_code == 200
    c_data = case_resp.json()

    # Verify source was created
    sources = c_data.get("sources", [])
    assert len(sources) >= 1
    stt_src = next(s for s in sources if s["modality"] == "stt")

    # CRITICAL: Original Hindi text must be preserved, NOT the English translation!
    assert "मुझे दो दिन से सीने में तेज दर्द हो रहा है" in stt_src["original_text"]
    assert stt_src["language"] == "hi"
    assert stt_src["audio_start"] == 0.5
    assert stt_src["audio_end"] == 4.2
    assert stt_src["document_ref"] == doc_id

    # Verify original audio document can be fetched from /api/documents/{doc_id}
    doc_resp = client.get(f"/api/documents/{doc_id}", headers=CLINICIAN)
    assert doc_resp.status_code == 200
    assert doc_resp.content == audio_data
    assert "audio/wav" in doc_resp.headers["content-type"]


def test_direct_document_intake_success_and_bbox_preservation(intake_client_factory):
    """Document intake uploads file, runs OCR, preserves bbox, pages and original filename."""
    fake_ocr = FakeOcrProcessor()
    client = intake_client_factory(document_processor=fake_ocr)

    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_doc_01")
    consent(client, case_id, headers=INTAKE)

    pdf_data = b"%PDF-1.4 sample medical report content"
    files = {"file": ("official_lab_vitals.pdf", io.BytesIO(pdf_data), "application/pdf")}

    resp = client.post(
        f"/api/cases/{case_id}/intake/document?auto_process=true",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 202
    data = resp.json()
    job_id = data["job_id"]
    doc_id = data["document_id"]
    assert data["kind"] == "document"

    # Poll job status
    poll_resp = client.get(f"/api/cases/{case_id}/intake/jobs/{job_id}", headers=CLINICIAN)
    assert poll_resp.status_code == 200
    assert poll_resp.json()["stage"] == "complete"

    # Inspect sources
    case_resp = client.get(f"/api/cases/{case_id}", headers=CLINICIAN)
    c_data = case_resp.json()
    ocr_sources = [s for s in c_data.get("sources", []) if s["modality"] == "ocr"]
    assert len(ocr_sources) == 2

    bp_src = next(s for s in ocr_sources if "Blood Pressure" in s["original_text"])
    assert bp_src["page"] == 1
    assert bp_src["bbox"] == [20.0, 30.0, 180.0, 50.0]
    assert bp_src["document_ref"] == doc_id

    # Verify document download
    doc_resp = client.get(f"/api/documents/{doc_id}", headers=CLINICIAN)
    assert doc_resp.status_code == 200
    assert doc_resp.content == pdf_data


def test_direct_document_intake_reports_stage_timings(intake_client_factory):
    """A stage-aware processor can report OCR/organizer timing and progress."""

    class StageAwareOcrProcessor(FakeOcrProcessor):
        def __init__(self):
            super().__init__()
            self.stages = []

        def __call__(self, document_bytes, filename, model, base_url, on_stage=None):
            if on_stage is not None:
                self.stages.append("organizing")
                on_stage("organizing")
            result = super().__call__(document_bytes, filename, model, base_url)
            result["timings"] = {
                "ocr_seconds": 0.01,
                "organizer_seconds": 0.02,
            }
            return result

    processor = StageAwareOcrProcessor()
    client = intake_client_factory(document_processor=processor)
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_doc_timing")
    consent(client, case_id, headers=INTAKE)

    files = {"file": ("lab.pdf", io.BytesIO(b"%PDF-1.4 sample"), "application/pdf")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/document?auto_process=false",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 202
    job = client.get(
        f"/api/cases/{case_id}/intake/jobs/{resp.json()['job_id']}",
        headers=CLINICIAN,
    ).json()

    assert processor.stages == ["organizing"]
    assert job["status"] == "complete"
    assert job["result"]["timings"]["ocr_seconds"] == 0.01
    assert job["result"]["timings"]["organizer_seconds"] == 0.02
    assert job["result"]["timings"]["total_seconds"] >= 0


def test_direct_document_intake_rejects_non_medical_document(intake_client_factory):
    """The organizer's non-medical classification stops persistence."""

    class NonMedicalProcessor:
        def __call__(self, document_bytes, filename, model, base_url):
            return {
                "file_name": filename,
                "is_medical": False,
                "document_type": "non_medical",
                "sources": [],
            }

    client = intake_client_factory(document_processor=NonMedicalProcessor())
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_doc_nonmedical")
    consent(client, case_id, headers=INTAKE)

    files = {"file": ("architecture.png", io.BytesIO(b"\x89PNG\r\n"), "image/png")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/document?auto_process=false",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = client.get(
        f"/api/cases/{case_id}/intake/jobs/{job_id}", headers=CLINICIAN
    ).json()
    assert job["status"] == "failed"
    assert "clinically relevant" in job["error"]

    case_data = client.get(f"/api/cases/{case_id}", headers=CLINICIAN).json()
    assert case_data.get("sources", []) == []


def test_direct_intake_missing_consent_is_rejected(intake_client_factory):
    """Direct intake without granted consent must be rejected with 409."""
    client = intake_client_factory()
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_noconsent_01")
    # DO NOT grant consent

    files = {"file": ("audio.wav", io.BytesIO(b"RIFF...."), "audio/wav")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/audio",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 409
    assert "consent" in resp.text.lower()


def test_direct_intake_wrong_role_denied(intake_client_factory):
    """Intake endpoint checks roles and rejects unknown roles."""
    client = intake_client_factory()
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_role_01")
    consent(client, case_id, headers=INTAKE)

    bad_headers = {"X-User-Id": "guest_user", "X-Role": "guest"}
    files = {"file": ("audio.wav", io.BytesIO(b"RIFF...."), "audio/wav")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/audio",
        headers=bad_headers,
        files=files,
    )
    assert resp.status_code == 403


def test_direct_intake_cross_patient_access_denied(intake_client_factory):
    """Patient A cannot upload into patient B's case."""
    client = intake_client_factory()
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_alice")
    consent(client, case_id, headers=INTAKE)

    bob_headers = patient_headers("pt_bob")
    files = {"file": ("audio.wav", io.BytesIO(b"RIFF...."), "audio/wav")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/audio",
        headers=bob_headers,
        files=files,
    )
    assert resp.status_code == 403


def test_direct_intake_unsupported_mime_rejected(intake_client_factory):
    """Bad mime types/extensions are rejected with 415."""
    client = intake_client_factory()
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_mime_01")
    consent(client, case_id, headers=INTAKE)

    # Executable uploaded as audio
    files = {"file": ("virus.exe", io.BytesIO(b"MZ......"), "application/x-msdownload")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/audio",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 415

    # Zip uploaded as document
    files = {"file": ("archive.zip", io.BytesIO(b"PK......"), "application/zip")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/document",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 415


def test_direct_intake_oversized_upload_rejected(intake_client_factory):
    """Uploads exceeding max_upload_bytes are rejected with 413."""
    # Test setting max upload is 1 MB
    client = intake_client_factory(max_upload_bytes=1000)
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_size_01")
    consent(client, case_id, headers=INTAKE)

    large_data = b"0" * 2000
    files = {"file": ("big_audio.wav", io.BytesIO(large_data), "audio/wav")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/audio",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 413


def test_direct_intake_service_failure_preserves_original_document(intake_client_factory):
    """When STT/OCR service fails, original document is preserved and job marked failed."""
    failing_stt = FakeSttClient(should_fail=True, failure_message="Sarvam transcription unavailable")
    client = intake_client_factory(stt_service=failing_stt)

    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_fail_01")
    consent(client, case_id, headers=INTAKE)

    audio_bytes = b"RIFF..audio-to-preserve.."
    files = {"file": ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/audio",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    doc_id = resp.json()["document_id"]

    # Poll job status - should be failed
    poll_resp = client.get(f"/api/cases/{case_id}/intake/jobs/{job_id}", headers=CLINICIAN)
    assert poll_resp.status_code == 200
    job = poll_resp.json()
    assert job["stage"] == "failed"
    assert job["status"] == "failed"
    assert "Sarvam transcription unavailable" in job["error"]

    # Original document is NOT deleted and can be retrieved or retried
    doc_resp = client.get(f"/api/documents/{doc_id}", headers=CLINICIAN)
    assert doc_resp.status_code == 200
    assert doc_resp.content == audio_bytes


def test_auto_process_generates_facts_with_mock_provider(intake_client_factory):
    """When auto_process=True, successful intake triggers pipeline and produces clinical facts."""
    client = intake_client_factory()
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_autoprocess_01")
    consent(client, case_id, headers=INTAKE)

    files = {"file": ("vitals.pdf", io.BytesIO(b"%PDF sample"), "application/pdf")}
    resp = client.post(
        f"/api/cases/{case_id}/intake/document?auto_process=true",
        headers=CLINICIAN,
        files=files,
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Poll job
    poll = client.get(f"/api/cases/{case_id}/intake/jobs/{job_id}", headers=CLINICIAN)
    assert poll.json()["status"] == "complete"

    # Case should now have summary and facts from the MockProvider pipeline
    case_data = client.get(f"/api/cases/{case_id}", headers=CLINICIAN).json()
    assert case_data["case"]["status"] == "reviewed" or len(case_data.get("facts", [])) > 0


def test_legacy_json_endpoints_remain_functional(intake_client_factory):
    """Existing JSON endpoints (/sources/stt, /sources/ocr, /sources/message, /uploads) work unchanged."""
    client = intake_client_factory()
    case_id = new_case(client, headers=CLINICIAN, patient_id="pt_legacy_01")
    consent(client, case_id, headers=INTAKE)

    # 1. POST /sources/stt
    stt_resp = client.post(
        f"/api/cases/{case_id}/sources/stt",
        headers=CLINICIAN,
        json={
            "transcript_id": "legacy_stt_01",
            "language": "en",
            "segments": [{"text": "Patient has dry cough", "start": 0.0, "end": 2.0}],
        },
    )
    assert stt_resp.status_code == 200
    assert len(stt_resp.json()["inserted"]) == 1

    # 2. POST /sources/ocr
    ocr_resp = client.post(
        f"/api/cases/{case_id}/sources/ocr",
        headers=CLINICIAN,
        json={
            "file_name": "legacy_report.pdf",
            "is_medical": True,
            "document_type": "other_medical",
            "sources": [{"text": "SpO2: 98%", "page": 1}],
        },
    )
    assert ocr_resp.status_code == 200
    assert len(ocr_resp.json()["inserted"]) == 1

    # 3. POST /sources/message
    msg_resp = client.post(
        f"/api/cases/{case_id}/sources/message",
        headers=CLINICIAN,
        json={"text": "Feeling slightly dizzy since morning", "language": "en"},
    )
    assert msg_resp.status_code == 200
    assert len(msg_resp.json()["inserted"]) == 1

    # 4. POST /uploads
    upload_resp = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=CLINICIAN,
        files={"file": ("raw_test.txt", io.BytesIO(b"raw test data"), "text/plain")},
    )
    assert upload_resp.status_code == 201
    assert "document_id" in upload_resp.json()
