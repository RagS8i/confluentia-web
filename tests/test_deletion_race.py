from __future__ import annotations

import threading

from app.pipeline.orchestrator import CaseUnavailable
from app.model.provider import MockProvider
from app.schemas import ExtractedFact, ExtractionResult, FactStatus, FactType

from .conftest import ADMIN, CLINICIAN, consent, ingest_stt, new_case, stt_payload


class _BlockingProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def extract_facts(self, sources, settings):
        self.entered.set()
        assert self.release.wait(timeout=10)
        return ExtractionResult(
            facts=[
                ExtractedFact(
                    fact_type=FactType.symptom,
                    label="chest pain",
                    value="chest pain",
                    status=FactStatus.present,
                    original_text="chest pain",
                    confidence=0.9,
                )
            ]
        )


def test_deletion_during_processing_does_not_resurrect_content(client_factory):
    provider = _BlockingProvider()
    client = client_factory(provider=provider)
    case_id = new_case(client)
    consent(client, case_id)
    ingest_stt(client, case_id, stt_payload("I have chest pain since yesterday."))

    errors = []

    def run():
        try:
            client.app.state.pipeline.process_case(case_id, actor="test")
        except Exception as exc:  # noqa: BLE001 - captured for assertion
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    assert provider.entered.wait(timeout=10)

    response = client.delete(
        f"/api/cases/{case_id}",
        headers=ADMIN,
        params={"reason": "race test deletion"},
    )
    assert response.status_code == 202, response.text
    assert response.json()["deletion"]["status"] == "completed"

    provider.release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert errors and isinstance(errors[0], CaseUnavailable)

    assert client.get(f"/api/cases/{case_id}", headers=CLINICIAN).status_code == 404
    deleted = client.get(f"/api/cases/{case_id}", headers=ADMIN)
    assert deleted.status_code == 200
    assert deleted.json()["case"]["status"] == "deleted"
    assert deleted.json()["facts"] == []
    assert deleted.json()["sources"] == []
    assert client.app.state.db.get_facts(case_id, client.app.state.cipher) == []
    assert client.app.state.db.get_sources(case_id, client.app.state.cipher) == []
    audit = client.get(f"/api/cases/{case_id}/audit", headers=ADMIN)
    assert audit.status_code == 200
    assert "case_deleted" in {item["action"] for item in audit.json()["items"]}
