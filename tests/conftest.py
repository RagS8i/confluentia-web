from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model.provider import MockProvider

RULES = Path(__file__).resolve().parent.parent / "app" / "triage" / "rules_demo.json"
FIXED_KEY = Fernet.generate_key().decode()

CLINICIAN = {"X-User-Id": "dr_clinician", "X-Role": "clinician"}
ADMIN = {"X-User-Id": "admin_user", "X-Role": "admin"}
INTAKE = {"X-User-Id": "intake_user", "X-Role": "intake"}


def patient_headers(patient_id: str) -> Dict[str, str]:
    return {
        "X-User-Id": f"user_{patient_id}",
        "X-Role": "patient",
        "X-Patient-Id": patient_id,
    }


def make_settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings(
        data_dir=tmp_path,
        db_path=tmp_path / "careflow.db",
        rules_path=RULES,
        llm_provider="mock",
        processing_mode="sync",
        encryption_key=FIXED_KEY,
        model_backoff_s=0.0,
        max_model_retries=1,
    )
    return settings.with_overrides(**overrides)


@pytest.fixture
def client_factory(tmp_path):
    clients = []

    def factory(provider=None, **overrides):
        application = create_app(
            make_settings(tmp_path, **overrides), provider=provider or MockProvider()
        )
        client = TestClient(application)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.close()


@pytest.fixture
def client(client_factory):
    return client_factory()


def new_case(client: TestClient, headers=CLINICIAN, patient_id="patient_001") -> str:
    response = client.post("/api/cases", headers=headers, json={"patient_id": patient_id})
    assert response.status_code == 201, response.text
    return response.json()["case_id"]


def consent(client: TestClient, case_id: str, headers=INTAKE) -> None:
    response = client.post(
        f"/api/cases/{case_id}/consent", headers=headers, json={"scope": "testing"}
    )
    assert response.status_code == 200, response.text


def stt_payload(
    text: Optional[str] = None,
    *,
    segments=None,
    language="en",
    confidence=0.92,
    transcript_id="stt_test",
):
    if segments is None:
        segments = [{"text": text or "", "start": 0.0, "end": 3.0, "confidence": confidence}]
    return {
        "transcript_id": transcript_id,
        "language": language,
        "language_status": "declared",
        "segments": segments,
        "model": "synthetic-stt",
        "confidence": confidence,
    }


def ocr_payload(
    pages,
    *,
    document_id="doc_test",
    language="en",
    file_name="report.pdf",
):
    """Build the structured OCR contract consumed by /sources/ocr.

    `pages` keeps the old test-call convenience shape; this helper flattens it
    into the source-level output emitted by the organizer.
    """
    sources = []
    for page_index, page in enumerate(pages, start=1):
        page_no = page.get("page", page.get("page_number", page_index))
        blocks = page.get("blocks")
        if blocks:
            for block in blocks:
                sources.append(
                    {
                        "block_id": block.get("block_id"),
                        "text": block.get("text", ""),
                        "page": page_no,
                        "bbox": block.get("bbox") or [],
                        "confidence": block.get("confidence", page.get("confidence")),
                        "unreadable": bool(block.get("unreadable", False)),
                    }
                )
        elif page.get("text"):
            sources.append(
                {
                    "block_id": page.get("block_id"),
                    "text": page["text"],
                    "page": page_no,
                    "bbox": page.get("bbox") or [],
                    "confidence": page.get("confidence"),
                    "unreadable": bool(page.get("unreadable", False)),
                }
            )
    return {
        "document_id": document_id,
        "file_name": file_name,
        "language": language,
        "is_medical": True,
        "document_type": "other_medical",
        "sources": sources,
        "ocr_model": "synthetic-ocr",
    }


def ingest_stt(client: TestClient, case_id: str, payload, headers=CLINICIAN):
    return client.post(f"/api/cases/{case_id}/sources/stt", headers=headers, json=payload)


def ingest_ocr(client: TestClient, case_id: str, payload, headers=CLINICIAN):
    return client.post(f"/api/cases/{case_id}/sources/ocr", headers=headers, json=payload)


def ingest_message(client: TestClient, case_id: str, text, headers=CLINICIAN):
    return client.post(f"/api/cases/{case_id}/sources/message", headers=headers, json={"text": text})


def process(client: TestClient, case_id: str, headers=CLINICIAN):
    return client.post(f"/api/cases/{case_id}/process?sync=true", headers=headers)


def detail(client: TestClient, case_id: str, headers=CLINICIAN):
    response = client.get(f"/api/cases/{case_id}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def warning_codes(data: dict) -> set[str]:
    return {warning["code"] for warning in data["warnings"]}


def facts_with_label(data: dict, label: str):
    return [fact for fact in data["facts"] if fact["label"].lower() == label.lower()]
