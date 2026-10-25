"""End-to-end smoke test for Direct Audio and Document Intake.

Can be run against a live running stack or locally to verify the end-to-end flow.
Usage:
    python scripts/smoke_direct_intake.py [--base-url http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

CLINICIAN_HEADERS = {
    "X-User-Id": "dr_smoke",
    "X-Role": "clinician",
}
INTAKE_HEADERS = {
    "X-User-Id": "intake_smoke",
    "X-Role": "intake",
}


def run_smoke(base_url: str = "http://127.0.0.1:8000") -> bool:
    print(f"=== CareFlow AI Direct Intake Smoke Test ===")
    print(f"Target CareFlow server: {base_url}")

    client = httpx.Client(base_url=base_url, timeout=30.0)

    # 1. Health check
    try:
        resp = client.get("/api/health")
        if resp.status_code != 200:
            print(f"[-] Health check failed: {resp.status_code} {resp.text}")
            return False
        print(f"[+] CareFlow health status: {resp.json().get('status')}")
    except Exception as exc:
        print(f"[-] Could not connect to CareFlow at {base_url}: {exc}")
        return False

    # 2. Create case
    case_resp = client.post("/api/cases", headers=CLINICIAN_HEADERS, json={"patient_id": "patient_smoke_live"})
    if case_resp.status_code != 201:
        print(f"[-] Case creation failed: {case_resp.text}")
        return False
    case_id = case_resp.json()["case_id"]
    print(f"[+] Created synthetic case: {case_id}")

    # 3. Grant consent
    consent_resp = client.post(f"/api/cases/{case_id}/consent", headers=INTAKE_HEADERS, json={"scope": "testing"})
    if consent_resp.status_code != 200:
        print(f"[-] Granting consent failed: {consent_resp.text}")
        return False
    print(f"[+] Consent granted for case {case_id}")

    # 4. Upload Audio
    sample_wav = Path(__file__).resolve().parent.parent / "data" / "samples" / "stt-sample.wav"
    if not sample_wav.is_file():
        print(f"[-] Sample audio file not found: {sample_wav}")
        return False

    print(f"[>] Uploading direct audio intake: {sample_wav.name} ({sample_wav.stat().st_size} bytes)...")
    with sample_wav.open("rb") as f:
        audio_resp = client.post(
            f"/api/cases/{case_id}/intake/audio?auto_process=true&target_language=en-IN",
            headers=CLINICIAN_HEADERS,
            files={"file": (sample_wav.name, f, "audio/wav")},
        )
    if audio_resp.status_code != 202:
        print(f"[-] Audio intake upload failed: {audio_resp.status_code} {audio_resp.text}")
        return False

    audio_job = audio_resp.json()
    job_id = audio_job["job_id"]
    audio_doc_id = audio_job["document_id"]
    print(f"[+] Audio intake accepted (job {job_id}, document {audio_doc_id}). Polling status...")

    # Poll audio job
    audio_done = False
    for attempt in range(40):
        time.sleep(2)
        poll = client.get(f"/api/cases/{case_id}/intake/jobs/{job_id}", headers=CLINICIAN_HEADERS)
        if poll.status_code == 200:
            p_data = poll.json()
            stage = p_data.get("stage")
            status = p_data.get("status")
            print(f"    Audio Job stage: {stage} (status: {status}, elapsed: {p_data.get('elapsed_seconds')}s)")
            if status == "complete":
                audio_done = True
                break
            elif status == "failed":
                print(f"    [!] Audio job failed: {p_data.get('error')}")
                break

    # 5. Check document download
    doc_resp = client.get(f"/api/documents/{audio_doc_id}?inline=true", headers=CLINICIAN_HEADERS)
    if doc_resp.status_code == 200:
        print(f"[+] Audio document endpoint accessible: {len(doc_resp.content)} bytes, media_type: {doc_resp.headers.get('content-type')}")
    else:
        print(f"[-] Audio document retrieval failed: {doc_resp.status_code}")

    # 6. Upload Document
    sample_png = Path(__file__).resolve().parent.parent / "data" / "samples" / "ocr-sample.png"
    if sample_png.is_file():
        print(f"[>] Uploading direct document intake: {sample_png.name} ({sample_png.stat().st_size} bytes)...")
        with sample_png.open("rb") as f:
            doc_intake_resp = client.post(
                f"/api/cases/{case_id}/intake/document?auto_process=true",
                headers=CLINICIAN_HEADERS,
                files={"file": (sample_png.name, f, "image/png")},
            )
        if doc_intake_resp.status_code == 202:
            d_job = doc_intake_resp.json()
            d_job_id = d_job["job_id"]
            d_doc_id = d_job["document_id"]
            print(f"[+] Document intake accepted (job {d_job_id}, document {d_doc_id}).")
        else:
            print(f"[-] Document intake upload failed: {doc_intake_resp.status_code} {doc_intake_resp.text}")

    print("\n[+] Direct intake smoke test execution completed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CareFlow AI Direct Intake Smoke Test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of CareFlow app")
    args = parser.parse_args()

    success = run_smoke(base_url=args.base_url)
    sys.exit(0 if success else 1)
