from __future__ import annotations

from .conftest import (
    ADMIN,
    CLINICIAN,
    INTAKE,
    consent,
    detail,
    facts_with_label,
    ingest_ocr,
    ingest_stt,
    new_case,
    ocr_payload,
    patient_headers,
    process,
    stt_payload,
)


def _case_with_facts(client, headers=CLINICIAN, patient_id="patient_001"):
    case_id = new_case(client, headers, patient_id=patient_id)
    consent(client, case_id, INTAKE)
    ingest_stt(
        client,
        case_id,
        stt_payload("I have chest pain and shortness of breath since yesterday."),
        headers=headers,
    )
    process(client, case_id, headers=headers)
    return case_id


def test_14_unauthorized_access_is_denied(client):
    assert client.get("/api/cases", headers={"X-User-Id": "x"}).status_code == 401
    assert client.get("/api/cases").status_code == 401

    patient_a = patient_headers("patient_a")
    patient_b = patient_headers("patient_b")
    case_id = _case_with_facts(client, patient_a, patient_id="patient_a")

    assert client.get(f"/api/cases/{case_id}", headers=patient_b).status_code == 403
    assert client.post(
        f"/api/cases/{case_id}/note/approve", headers=patient_b
    ).status_code == 403
    assert client.get(f"/api/cases/{case_id}/audit", headers=patient_a).status_code == 403
    assert client.get(f"/api/cases/{case_id}/audit", headers=ADMIN).status_code == 200
    assert client.patch(
        f"/api/cases/{case_id}/facts/anything",
        headers=INTAKE,
        json={"action": "verify"},
    ).status_code == 403
    assert client.get(f"/api/cases/{case_id}", headers=CLINICIAN).status_code == 200


def test_15_fact_corrections_edits_and_approvals_persist(client):
    case_id = _case_with_facts(client)
    data = detail(client, case_id)
    chest = facts_with_label(data, "chest pain")[0]
    fact_id = chest["fact_id"]

    response = client.patch(
        f"/api/cases/{case_id}/facts/{fact_id}",
        headers=CLINICIAN,
        json={
            "action": "correct",
            "label": "crushing chest pain",
            "note": "patient clarified the character of the pain",
        },
    )
    assert response.status_code == 200, response.text
    corrected = response.json()
    assert corrected["verification_status"] == "corrected"
    assert corrected["label"] == "crushing chest pain"
    assert corrected["reviewed_by"] == "dr_clinician"

    process(client, case_id)
    data = detail(client, case_id)
    persisted = next(fact for fact in data["facts"] if fact["fact_id"] == fact_id)
    assert persisted["label"] == "crushing chest pain"
    assert persisted["verification_status"] == "corrected"

    edited_text = "AI DRAFT - FOR CLINICIAN REVIEW\nClinician-reviewed summary. Verify all source material."
    response = client.patch(
        f"/api/cases/{case_id}/note", headers=CLINICIAN, json={"text": edited_text}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "clinician_edited"
    assert response.json()["version"] == 2

    response = client.post(f"/api/cases/{case_id}/note/approve", headers=CLINICIAN)
    assert response.status_code == 200, response.text
    approved = response.json()
    assert approved["status"] == "approved"
    assert approved["approved_by"] == "dr_clinician"
    assert detail(client, case_id)["case"]["status"] == "reviewed"

    process(client, case_id)
    data = detail(client, case_id)
    assert data["note"]["status"] == "approved"
    assert data["note"]["text"] == edited_text
    assert data["note"]["approved_by"] == "dr_clinician"

    response = client.post(f"/api/cases/{case_id}/note/approve", headers=CLINICIAN)
    assert response.status_code == 409


def test_15b_clinician_override_requires_reason_and_is_audited(client):
    case_id = _case_with_facts(client)
    before = detail(client, case_id)

    assert client.post(
        f"/api/cases/{case_id}/override",
        headers=CLINICIAN,
        json={"final_status": "ROUTINE_REVIEW", "reason": "x"},
    ).status_code == 422

    response = client.post(
        f"/api/cases/{case_id}/override",
        headers=CLINICIAN,
        json={
            "final_status": "ROUTINE_REVIEW",
            "reason": "Synthetic test override after direct telephone assessment.",
            "confirmed": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["override"]["original_status"] == before["case"]["workflow_status"]
    assert body["override"]["final_status"] == "ROUTINE_REVIEW"
    assert body["override"]["reviewer"] == "dr_clinician"

    audit = client.get(f"/api/cases/{case_id}/audit", headers=ADMIN).json()["items"]
    override_events = [item for item in audit if item["action"] == "status_overridden"]
    assert override_events
    assert "reason_recorded" in str(override_events[-1]["detail"])

    response = client.post(
        f"/api/cases/{case_id}/assign",
        headers=CLINICIAN,
        json={"reviewer": "dr_second_reviewer", "confirmed": True},
    )
    assert response.status_code == 200
    assert detail(client, case_id)["case"]["assigned_reviewer"] == "dr_second_reviewer"


def test_16_consent_is_required_and_withdrawal_blocks_processing(client):
    patient = patient_headers("patient_consent")
    case_id = new_case(client, patient, patient_id="patient_consent")
    response = ingest_stt(client, case_id, stt_payload("I have a headache."), headers=patient)
    assert response.status_code == 409

    response = process(client, case_id, headers=patient)
    assert response.status_code == 409
    assert "consent" in response.json()["detail"].lower()

    consent(client, case_id, headers=patient)
    assert process(client, case_id, headers=patient).status_code == 200

    response = client.post(
        f"/api/cases/{case_id}/consent/withdraw",
        headers=patient,
        json={"reason": "testing withdrawal", "delete_data": False},
    )
    assert response.status_code == 200, response.text
    assert response.json()["deletion"] == "not_requested"
    data = detail(client, case_id, headers=patient)
    assert data["case"]["status"] == "withdrawn"
    assert data["case"]["workflow_status"] == "UNASSESSED"

    assert process(client, case_id, headers=patient).status_code == 409


def test_17_consent_withdrawal_deletes_clinical_content_and_retains_audit(client):
    patient = patient_headers("patient_delete")
    case_id = new_case(client, patient, patient_id="patient_delete")
    consent(client, case_id, headers=patient)
    ingest_stt(client, case_id, stt_payload("I have chest pain."), headers=patient)
    process(client, case_id, headers=patient)
    upload = client.post(
        f"/api/cases/{case_id}/uploads",
        headers=patient,
        files={"file": ("note.pdf", b"%PDF-1.4 synthetic test document", "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document_id = upload.json()["document_id"]
    assert client.get(f"/api/documents/{document_id}", headers=patient).status_code == 200

    response = client.delete(
        f"/api/cases/{case_id}",
        headers=patient,
        params={"reason": "patient requested deletion"},
    )
    assert response.status_code == 202, response.text
    job = response.json()["deletion"]
    assert job["status"] == "completed"
    assert all(step["status"] == "completed" for step in job["steps"])

    assert client.get(f"/api/cases/{case_id}", headers=patient).status_code == 404
    assert client.get(f"/api/documents/{document_id}", headers=patient).status_code == 404

    jobs = client.get("/api/deletion-jobs", headers=ADMIN).json()["items"]
    assert any(item["job_id"] == job["job_id"] and item["status"] == "completed" for item in jobs)
    audit = client.get(f"/api/cases/{case_id}/audit", headers=ADMIN)
    assert audit.status_code == 200
    actions = {item["action"] for item in audit.json()["items"]}
    assert "case_deleted" in actions
    assert "deletion_requested" in actions
    assert "I have chest pain" not in audit.text

    uploads = list((client.app.state.settings.uploads_dir).glob("*"))
    assert uploads == []


def test_clinical_text_is_encrypted_at_rest(client, tmp_path):
    case_id = _case_with_facts(client, patient_id="patient_encrypted")
    ingest_ocr(
        client,
        case_id,
        ocr_payload(
            [
                {
                    "page": 1,
                    "confidence": 0.95,
                    "blocks": [
                        {
                            "text": "Patient reports unique OCR marker for encryption.",
                            "bbox": [1, 2, 30, 40],
                            "confidence": 0.95,
                        }
                    ],
                }
            ],
            document_id="encrypted-ocr",
        ),
    )
    process(client, case_id)
    data = detail(client, case_id)
    assert any(
        "chest pain and shortness of breath" in fact["original_text"].lower()
        for fact in data["facts"]
    )
    client.post(
        f"/api/cases/{case_id}/communications/draft",
        headers=CLINICIAN,
        json={"intent": "urgent_review", "language": "en"},
    )

    db_bytes = (tmp_path / "careflow.db").read_bytes()
    wal_path = tmp_path / "careflow.db-wal"
    if wal_path.exists():
        db_bytes += wal_path.read_bytes()
    assert b"chest pain and shortness of breath" not in db_bytes
    assert b"unique OCR marker for encryption" not in db_bytes
    assert b"co-occurring chest pain" not in db_bytes
    assert b"not a diagnosis" not in db_bytes
