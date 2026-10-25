from __future__ import annotations

from .conftest import (
    CLINICIAN,
    INTAKE,
    consent,
    detail,
    ingest_stt,
    new_case,
    patient_headers,
    process,
    stt_payload,
)
from app.model.provider import MockProvider


def _processed(client, text, patient_id, headers=CLINICIAN):
    case_id = new_case(client, headers, patient_id=patient_id)
    consent(client, case_id, INTAKE)
    response = ingest_stt(client, case_id, stt_payload(text), headers=headers)
    assert response.status_code == 200, response.text
    assert process(client, case_id, headers=headers).status_code == 200
    return case_id


class _FixedUrgencyProvider(MockProvider):
    """Test-only provider that returns a deterministic score per case."""

    def __init__(self):
        super().__init__()
        self.scores = {}

    def assess_urgency(self, payload, settings):
        score = self.scores.get(payload["case_id"], 50)
        return {"urgency_score": score, "rationale": f"fixed test score {score}"}


def test_pipeline_computes_advisory_urgency_and_exposes_it(client):
    case_id = _processed(
        client,
        "I have chest pain and shortness of breath.",
        "patient_urgent",
    )
    data = detail(client, case_id)
    urgency = data["case"]["urgency"]

    assert urgency is not None
    assert 0 <= urgency["score"] <= 100
    assert urgency["score"] >= 90
    assert urgency["source"] == "slm"
    assert urgency["model"] == "careflow-mock-lexicon/0.1.0"
    assert urgency["slm_score"] == urgency["score"]
    assert urgency["rationale"]
    assert data["case"]["workflow_status"] == "EMERGENCY"

    queue_item = next(
        item
        for item in client.get("/api/cases", headers=CLINICIAN).json()["items"]
        if item["case_id"] == case_id
    )
    assert queue_item["urgency_score"] == urgency["score"]
    assert queue_item["urgency_source"] == "slm"

    recommendation = next(
        item
        for item in client.get("/api/queue/intelligence", headers=CLINICIAN).json()["items"]
        if item["case_id"] == case_id
    )
    assert recommendation["urgency_score"] == urgency["score"]
    assert any("Advisory urgency" in text for text in recommendation["explanations"])


def test_queue_orders_by_urgency_within_the_same_priority_category(client_factory):
    provider = _FixedUrgencyProvider()
    client = client_factory(provider=provider)

    low_id = new_case(client, CLINICIAN, patient_id="patient_low")
    high_id = new_case(client, CLINICIAN, patient_id="patient_high")
    provider.scores[low_id] = 10
    provider.scores[high_id] = 90

    for case_id in (low_id, high_id):
        consent(client, case_id, INTAKE)
        ingest_stt(
            client,
            case_id,
            stt_payload("I have a mild headache."),
            headers=CLINICIAN,
        )
        assert process(client, case_id).status_code == 200

    low = detail(client, low_id)
    high = detail(client, high_id)
    assert low["case"]["workflow_status"] == high["case"]["workflow_status"]

    items = client.get("/api/cases", headers=CLINICIAN).json()["items"]
    ids = [item["case_id"] for item in items]
    assert ids.index(high_id) < ids.index(low_id)

    intelligence = client.get("/api/queue/intelligence", headers=CLINICIAN).json()
    intel_ids = [item["case_id"] for item in intelligence["items"]]
    assert intel_ids.index(high_id) < intel_ids.index(low_id)


def test_priority_category_still_dominates_urgency(client_factory):
    provider = _FixedUrgencyProvider()
    client = client_factory(provider=provider)

    emergency_id = new_case(client, CLINICIAN, patient_id="patient_emergency")
    routine_id = new_case(client, CLINICIAN, patient_id="patient_routine")
    provider.scores[emergency_id] = 0
    provider.scores[routine_id] = 100

    for case_id, text in (
        (emergency_id, "I have chest pain and shortness of breath."),
        (routine_id, "I have a mild headache."),
    ):
        consent(client, case_id, INTAKE)
        ingest_stt(client, case_id, stt_payload(text), headers=CLINICIAN)
        assert process(client, case_id).status_code == 200

    assert detail(client, emergency_id)["case"]["workflow_status"] == "EMERGENCY"
    assert detail(client, routine_id)["case"]["workflow_status"] == "ROUTINE_REVIEW"

    items = client.get("/api/cases", headers=CLINICIAN).json()["items"]
    ids = [item["case_id"] for item in items]
    assert ids.index(emergency_id) < ids.index(routine_id)

    intelligence = client.get("/api/queue/intelligence", headers=CLINICIAN).json()
    intel_ids = [item["case_id"] for item in intelligence["items"]]
    assert intel_ids.index(emergency_id) < intel_ids.index(routine_id)


def test_clinical_staff_can_edit_urgency_with_audit_and_workflow_unchanged(client):
    case_id = _processed(client, "I have a mild headache.", "patient_edit")
    before = detail(client, case_id)
    original_outcome = before["case"]["workflow_status"]

    response = client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=CLINICIAN,
        json={
            "score": 88,
            "reason": "Triage nurse reviewed the full record and red flags.",
            "rationale": "Clinician assessment based on the complete record.",
            "confirmed": True,
        },
    )
    assert response.status_code == 200, response.text
    urgency = response.json()["urgency"]
    assert urgency["score"] == 88
    assert urgency["source"] == "clinician"
    assert urgency["edited_by"] == "dr_clinician"
    assert urgency["slm_score"] is not None
    assert urgency["history"][-1]["previous_score"] == before["case"]["urgency"]["score"]

    after = detail(client, case_id)
    assert after["case"]["workflow_status"] == original_outcome
    assert after["case"]["urgency"]["score"] == 88
    assert after["case"]["urgency"]["source"] == "clinician"
    assert any(
        item["action"] == "edit_urgency" for item in after["reviews"]
    )

    audit = client.get(f"/api/cases/{case_id}/audit", headers=CLINICIAN).json()["items"]
    assert any(item["action"] == "urgency_edited" for item in audit)


def test_nurse_and_doctor_roles_can_edit_but_intake_and_patient_cannot(client):
    case_id = _processed(client, "I have a mild headache.", "patient_roles")
    nurse = {"X-User-Id": "nurse_1", "X-Role": "nurse"}
    doctor = {"X-User-Id": "doctor_1", "X-Role": "doctor"}

    nurse_response = client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=nurse,
        json={"score": 40, "reason": "Nurse reviewed the case.", "confirmed": True},
    )
    assert nurse_response.status_code == 200, nurse_response.text
    assert nurse_response.json()["urgency"]["edited_by"] == "nurse_1"

    doctor_response = client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=doctor,
        json={"score": 55, "reason": "Doctor reviewed the case.", "confirmed": True},
    )
    assert doctor_response.status_code == 200, doctor_response.text
    assert doctor_response.json()["urgency"]["edited_by"] == "doctor_1"
    assert client.get("/api/queue/intelligence", headers=nurse).status_code == 200

    intake_response = client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=INTAKE,
        json={"score": 30, "reason": "Intake attempt.", "confirmed": True},
    )
    assert intake_response.status_code == 403

    patient_response = client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=patient_headers("patient_roles"),
        json={"score": 30, "reason": "Patient attempt.", "confirmed": True},
    )
    assert patient_response.status_code == 403


def test_reprocessing_preserves_a_clinician_edit(client):
    case_id = _processed(client, "I have a mild headache.", "patient_preserve")
    assert client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=CLINICIAN,
        json={"score": 77, "reason": "Clinical review completed.", "confirmed": True},
    ).status_code == 200

    assert process(client, case_id).status_code == 200
    urgency = detail(client, case_id)["case"]["urgency"]
    assert urgency["score"] == 77
    assert urgency["source"] == "clinician"
    assert urgency["edited_by"] == "dr_clinician"


def test_urgency_edit_requires_confirmation_reason_and_valid_range(client):
    case_id = _processed(client, "I have a mild headache.", "patient_validation")

    assert client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=CLINICIAN,
        json={"score": 80, "reason": "No confirmation.", "confirmed": False},
    ).status_code == 422
    assert client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=CLINICIAN,
        json={"score": 101, "reason": "Out of range.", "confirmed": True},
    ).status_code == 422
    assert client.patch(
        f"/api/cases/{case_id}/urgency",
        headers=CLINICIAN,
        json={"score": 50, "reason": "", "confirmed": True},
    ).status_code == 422


def test_model_failure_falls_back_to_a_safe_deterministic_score(client_factory):
    client = client_factory(provider=MockProvider(mode="timeout"))
    case_id = _processed(client, "I have chest pain.", "patient_fallback")
    urgency = detail(client, case_id)["case"]["urgency"]

    assert urgency["source"] == "fallback"
    assert urgency["score"] == 50
    assert urgency["slm_score"] is None
    assert "fallback" in urgency["rationale"].lower()
