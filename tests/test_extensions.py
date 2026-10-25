from __future__ import annotations

from .conftest import (
    ADMIN,
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


def _case(client, text="I have chest pain and shortness of breath.", patient_id="patient_ext"):
    case_id = new_case(client, CLINICIAN, patient_id=patient_id)
    consent(client, case_id, INTAKE)
    response = ingest_stt(client, case_id, stt_payload(text), headers=CLINICIAN)
    assert response.status_code == 200, response.text
    assert process(client, case_id).status_code == 200
    return case_id


def test_communications_require_approval_and_safe_emergency_path(client):
    case_id = _case(client)

    response = client.post(
        f"/api/cases/{case_id}/communications/draft",
        headers=CLINICIAN,
        json={"intent": "status_update", "language": "en"},
    )
    assert response.status_code == 409

    response = client.post(
        f"/api/cases/{case_id}/communications/draft",
        headers=CLINICIAN,
        json={"intent": "urgent_review", "language": "en"},
    )
    assert response.status_code == 201, response.text
    message = response.json()
    assert message["status"] == "draft"
    assert "not a diagnosis" in message["body"]

    assert client.post(
        f"/api/cases/{case_id}/communications/{message['message_id']}/send",
        headers=CLINICIAN,
    ).status_code == 409

    response = client.post(
        f"/api/cases/{case_id}/communications/{message['message_id']}/approve",
        headers=CLINICIAN,
        json={},
    )
    assert response.status_code == 200, response.text
    assert response.json()["approved_by"] == "dr_clinician"

    response = client.post(
        f"/api/cases/{case_id}/communications/{message['message_id']}/send",
        headers=CLINICIAN,
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "sent"

    patient = patient_headers("patient_ext")
    data = detail(client, case_id, headers=patient)
    assert [item["message_id"] for item in data["communications"]] == [message["message_id"]]
    assert data["escalations"] == []
    assert data["queue_estimate"] is None

    assert client.post(
        f"/api/cases/{case_id}/communications/draft",
        headers=patient,
        json={"intent": "status_update", "language": "en"},
    ).status_code == 403

    assert client.post(
        f"/api/cases/{case_id}/communications/draft",
        headers=CLINICIAN,
        json={"intent": "urgent_review", "language": "ta"},
    ).status_code == 422


def test_missing_information_message_uses_curated_followups(client):
    case_id = _case(client, "I have a mild headache.")
    data = detail(client, case_id)
    assert data["case"]["workflow_status"] in {"ROUTINE_REVIEW", "PRIORITY_REVIEW"}
    assert data["followups"]

    response = client.post(
        f"/api/cases/{case_id}/communications/draft",
        headers=CLINICIAN,
        json={"intent": "missing_information", "language": "en"},
    )
    assert response.status_code == 201, response.text
    body = response.json()["body"]
    assert "clinical team needs" in body
    assert "diagnosis" in body
    assert any(item["text"] in body for item in data["followups"] if item["language"] == "en")


def test_followup_answer_is_new_source_and_reprocesses(client):
    case_id = _case(client, "I have a mild headache.", patient_id="patient_follow")
    data = detail(client, case_id)
    question = next(item for item in data["followups"] if item["language"] == "en")
    followup_id = question["followup_id"]
    patient = patient_headers("patient_follow")

    assert client.post(
        f"/api/cases/{case_id}/followups/{followup_id}/send", headers=CLINICIAN
    ).status_code == 409
    assert client.post(
        f"/api/cases/{case_id}/followups/{followup_id}/approve",
        headers=patient,
        json={},
    ).status_code == 403
    response = client.post(
        f"/api/cases/{case_id}/followups/{followup_id}/approve",
        headers=CLINICIAN,
        json={"language": "en"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "approved"
    assert client.post(
        f"/api/cases/{case_id}/followups/{followup_id}/send", headers=CLINICIAN
    ).status_code == 200

    before_sources = len(detail(client, case_id)["sources"])
    response = client.post(
        f"/api/cases/{case_id}/followups/{followup_id}/answers",
        headers=patient,
        json={"text": "It started two days ago and it is mild.", "language": "en"},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["followup"]["status"] == "answered"
    assert result["processing"] == "completed"
    assert result["answer"]["source_id"]

    after = detail(client, case_id)
    assert len(after["sources"]) == before_sources + 1
    assert any(
        source["source_id"] == result["answer"]["source_id"]
        and source["modality"] == "typed_message"
        for source in after["sources"]
    )
    assert any(
        answer["answer_id"] == result["answer"]["answer_id"]
        for answer in after["followup_answers"]
    )


def test_emergency_escalation_is_immediate_deduplicated_and_acknowledgeable(client):
    case_id = _case(client, "I have chest pain and shortness of breath.")
    response = client.get(f"/api/escalations?case_id={case_id}", headers=CLINICIAN)
    assert response.status_code == 200, response.text
    first = response.json()["items"]
    assert first
    assert any(item["status"] == "sent" for item in first)
    assert all(item["status"] != "acknowledged" for item in first)

    process(client, case_id)
    second = client.get(f"/api/escalations?case_id={case_id}", headers=CLINICIAN).json()["items"]
    assert len(second) == len(first)

    assert client.get(f"/api/escalations/{first[0]['escalation_id']}", headers=ADMIN).status_code in {404, 405}
    assert client.get(
        f"/api/escalations?case_id={case_id}", headers=patient_headers("patient_ext")
    ).status_code == 403

    response = client.post(
        f"/api/escalations/{first[0]['escalation_id']}/acknowledge",
        headers=CLINICIAN,
        json={"note": "Seen by the synthetic on-call clinician."},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "acknowledged"
    assert response.json()["acknowledged_by"] == "dr_clinician"

    data = detail(client, case_id)
    assert any(item["escalation_id"] == first[0]["escalation_id"] for item in data["escalations"])


def test_failed_escalation_is_visible_for_retry(client_factory):
    client = client_factory(escalation_delivery="fail")
    case_id = _case(client, "I have chest pain and shortness of breath.")
    items = client.get(f"/api/escalations?case_id={case_id}", headers=CLINICIAN).json()["items"]
    assert items
    assert all(item["status"] == "failed" for item in items)
    assert all(item["last_error"] for item in items)


def test_queue_intelligence_preserves_priority_and_requires_operational_sample(client):
    emergency = _case(
        client,
        "I have chest pain and shortness of breath.",
        patient_id="queue_emergency",
    )
    routine = _case(client, "I have a mild headache.", patient_id="queue_routine")

    response = client.get("/api/queue/intelligence", headers=CLINICIAN)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["items"][0]["workflow_status"] == "EMERGENCY"
    assert data["items"][0]["case_id"] == emergency
    assert data["items"][0]["wait_estimate"]["available"] is False
    assert data["bottlenecks"]["total_waiting"] >= 2

    routine_item = next(item for item in data["items"] if item["case_id"] == routine)
    assert routine_item["wait_estimate"]["available"] is False

    assert client.put(
        "/api/operations/staffing",
        headers=INTAKE,
        json={
            "clinicians_available": 2,
            "intake_available": 2,
            "average_review_minutes": 12,
            "sample_size": 8,
        },
    ).status_code == 403

    response = client.put(
        "/api/operations/staffing",
        headers=CLINICIAN,
        json={
            "clinicians_available": 2,
            "intake_available": 2,
            "average_review_minutes": 12,
            "sample_size": 8,
        },
    )
    assert response.status_code == 200, response.text
    data = client.get("/api/queue/intelligence", headers=CLINICIAN).json()
    updated = next(item for item in data["items"] if item["case_id"] == routine)
    assert updated["wait_estimate"]["available"] is True
    assert updated["wait_estimate"]["minimum_minutes"] >= 1
    assert "queue position" in updated["wait_estimate"]["disclaimer"]

    response = client.post(
        f"/api/cases/{routine}/communications/draft",
        headers=CLINICIAN,
        json={"intent": "wait_update", "language": "en"},
    )
    assert response.status_code == 201, response.text
    body = response.json()["body"]
    assert str(updated["wait_estimate"]["minimum_minutes"]) in body

    assert detail(client, emergency)["case"]["workflow_status"] == "EMERGENCY"
    assert detail(client, routine)["case"]["workflow_status"] in {
        "ROUTINE_REVIEW",
        "PRIORITY_REVIEW",
    }


def test_override_and_assignment_require_confirmation_and_warnings_survive(client):
    case_id = _case(client, "I have chest pain.")
    assert client.post(
        f"/api/cases/{case_id}/override",
        headers=CLINICIAN,
        json={"final_status": "ROUTINE_REVIEW", "reason": "Synthetic review."},
    ).status_code == 422
    assert client.post(
        f"/api/cases/{case_id}/assign",
        headers=CLINICIAN,
        json={"reviewer": "dr_next"},
    ).status_code == 422

    data = detail(client, case_id)
    warnings = {item["warning_id"]: item for item in data["warnings"]}
    warning_id = next(
        item["warning_id"] for item in data["warnings"] if item["kind"] == "rule"
    )
    assert client.post(
        f"/api/cases/{case_id}/warnings/{warning_id}/acknowledge", headers=CLINICIAN
    ).status_code == 200

    process(client, case_id)
    after = detail(client, case_id)
    persisted = {item["warning_id"]: item for item in after["warnings"]}
    assert warning_id in persisted
    assert persisted[warning_id]["status"] == "acknowledged"
    assert set(persisted).issuperset(set(warnings))


def test_extraction_schema_requires_safety_fields(tmp_path):
    from app.config import Settings
    from app.model.provider import OllamaProvider

    provider = OllamaProvider(
        Settings(data_dir=tmp_path, db_path=tmp_path / "careflow.db")
    )
    required = provider._extraction_schema["$defs"]["ExtractedFact"]["required"]
    assert "status" in required
    assert "assertion" in required
