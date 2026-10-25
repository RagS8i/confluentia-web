from __future__ import annotations

import time

from app.model.provider import MockProvider

from .conftest import (
    CLINICIAN,
    consent,
    detail,
    ingest_message,
    ingest_ocr,
    ingest_stt,
    new_case,
    ocr_payload,
    patient_headers,
    process,
    stt_payload,
)


def _queue_ready(client, case_id, headers=CLINICIAN, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/cases/{case_id}/status", headers=headers)
        assert status.status_code == 200, status.text
        if status.json()["status"] in {"processed", "partially_processed", "processing_failed"}:
            return status.json()
        time.sleep(0.05)
    raise AssertionError("background processing did not finish")


def test_duplicate_stt_and_ocr_ingestion_are_idempotent(client):
    case_id = new_case(client)
    consent(client, case_id)
    payload = stt_payload("I have chest pain since yesterday.", transcript_id="stt_dup")
    first = ingest_stt(client, case_id, payload)
    second = ingest_stt(client, case_id, payload)
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["inserted"]
    assert second.json()["inserted"] == []
    assert second.json()["duplicates"] == first.json()["inserted"]

    ocr = ocr_payload(
        [{"page": 1, "text": "Temperature 98.6 F.", "confidence": 0.9}],
        document_id="doc_dup",
    )
    first_ocr = ingest_ocr(client, case_id, ocr)
    second_ocr = ingest_ocr(client, case_id, ocr)
    assert first_ocr.json()["inserted"]
    assert second_ocr.json()["duplicates"] == first_ocr.json()["inserted"]
    assert len(detail(client, case_id)["sources"]) == 2


def test_upload_validation_limits_and_dedup(client_factory):
    client = client_factory(max_upload_bytes=200)
    case_id = new_case(client)
    consent(client, case_id)

    content = b"%PDF-1.4 " + b"x" * 100
    first = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=CLINICIAN,
        files={"file": ("report.pdf", content, "application/pdf")},
    )
    second = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=CLINICIAN,
        files={"file": ("report.pdf", content, "application/pdf")},
    )
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first.json()["document_id"]

    too_big = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=CLINICIAN,
        files={"file": ("big.pdf", b"y" * 500, "application/pdf")},
    )
    assert too_big.status_code == 413

    wrong_type = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=CLINICIAN,
        files={"file": ("run.exe", b"MZ...", "application/x-msdownload")},
    )
    assert wrong_type.status_code == 415

    empty = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=CLINICIAN,
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert empty.status_code == 422


def test_upload_requires_consent(client):
    case_id = new_case(client)
    response = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=CLINICIAN,
        files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")},
    )
    assert response.status_code == 409


def test_empty_source_is_rejected_and_never_reassuring(client):
    case_id = new_case(client)
    consent(client, case_id)
    response = ingest_stt(client, case_id, stt_payload("   ", transcript_id="stt_empty"))
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "empty_transcript"
    assert detail(client, case_id)["sources"] == []

    process(client, case_id)
    data = detail(client, case_id)
    assert data["case"]["workflow_status"] == "UNASSESSED"
    assert data["case"]["status"] == "processing_failed"
    codes = {warning["code"] for warning in data["warnings"]}
    assert "NO_USABLE_SOURCES" in codes
    assert "NO_EVALUABLE_FACTS" in codes


def test_language_metadata_mismatch_is_flagged_not_silently_overwritten(client):
    case_id = new_case(client)
    consent(client, case_id)
    response = ingest_stt(
        client,
        case_id,
        stt_payload("मुझे बुखार है।", language="en", transcript_id="stt_mismatch"),
    )
    assert response.status_code == 200, response.text
    source = response.json()["sources"][0]
    assert source["language"] == "en"
    assert any("language_metadata_mismatch" in warning for warning in source["warnings"])
    process(client, case_id)
    data = detail(client, case_id)
    assert data["facts"]
    assert data["facts"][0]["language"] == "en"


def test_typed_message_source_works(client):
    case_id = new_case(client)
    consent(client, case_id)
    response = ingest_message(client, case_id, "I have had a headache for two days.")
    assert response.status_code == 200, response.text
    process(client, case_id)
    data = detail(client, case_id)
    assert data["sources"][0]["modality"] == "typed_message"
    assert any(fact["label"] == "headache" for fact in data["facts"])


def test_queue_mode_returns_202_and_poll_reaches_terminal_state(client_factory):
    client = client_factory(processing_mode="queue")
    case_id = new_case(client)
    consent(client, case_id)
    ingest_stt(client, case_id, stt_payload("Chest pain and shortness of breath."))
    response = client.post(f"/api/cases/{case_id}/process", headers=CLINICIAN)
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"
    state = _queue_ready(client, case_id)
    assert state["workflow_status"] == "EMERGENCY"
    data = detail(client, case_id)
    assert data["summary"]["draft_note"]
    assert data["note"]["status"] == "draft_ai"


def test_processing_without_facts_still_creates_safe_note(client):
    case_id = new_case(client)
    consent(client, case_id)
    process(client, case_id)
    data = detail(client, case_id)
    assert data["case"]["status"] == "processing_failed"
    assert data["case"]["workflow_status"] == "UNASSESSED"
    assert data["note"]["status"] == "draft_ai"
    assert "AI DRAFT" in data["note"]["text"]
