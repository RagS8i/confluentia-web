"""Consent-withdrawal and deletion workflow.

Deletion is recorded as a resumable job.  Each step is idempotent so an
interrupted withdrawal can be retried without leaving ciphertext or derived
clinical data behind.  Audit metadata is retained for accountability; clinical
content is not.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .db import Database
from .schemas import CaseStatus, Outcome
from .storage import DocumentStore, random_id, safe_error


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeletionService:
    def __init__(self, db: Database, documents: DocumentStore):
        self.db = db
        self.documents = documents

    def start(self, case_id: str, actor: str, reason: str = "consent withdrawal") -> Dict[str, Any]:
        job = {
            "job_id": random_id("del"),
            "case_id": case_id,
            "status": "pending",
            "steps": [
                {"name": "mark_withdrawn", "status": "pending", "error": None},
                {"name": "delete_sources", "status": "pending", "error": None},
                {"name": "delete_facts", "status": "pending", "error": None},
                {"name": "delete_derived", "status": "pending", "error": None},
                {"name": "delete_documents", "status": "pending", "error": None},
                {"name": "purge_uploads", "status": "pending", "error": None},
                {"name": "finalize", "status": "pending", "error": None},
            ],
            "created_at": _now(),
            "updated_at": _now(),
            "last_error": None,
            "actor": actor,
            "reason": reason,
        }
        self.db.save_deletion_job(job)
        self.db.audit(actor, "system", "deletion_requested", case_id, {"job_id": job["job_id"], "reason": reason})
        return self.run(job["job_id"])

    def run(self, job_id: str) -> Dict[str, Any]:
        job = self.db.get_deletion_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] == "completed":
            return job
        job["status"] = "running"
        job["updated_at"] = _now()
        self.db.save_deletion_job(job)

        for step in job["steps"]:
            if step["status"] == "completed":
                continue
            step["status"] = "running"
            self.db.save_deletion_job(job)
            try:
                self._execute(job["case_id"], step["name"])
                step["status"] = "completed"
                step["error"] = None
            except Exception as exc:  # noqa: BLE001 - persist and allow retry
                step["status"] = "failed"
                step["error"] = safe_error(exc)
                job["status"] = "partial"
                job["last_error"] = safe_error(exc)
                job["updated_at"] = _now()
                self.db.save_deletion_job(job)
                self.db.audit(
                    "system",
                    "system",
                    "deletion_step_failed",
                    job["case_id"],
                    {"job_id": job_id, "step": step["name"], "error": safe_error(exc)},
                )
                return job
            job["updated_at"] = _now()
            self.db.save_deletion_job(job)

        job["status"] = "completed"
        job["last_error"] = None
        job["updated_at"] = _now()
        self.db.save_deletion_job(job)
        self.db.audit(
            "system",
            "system",
            "case_deleted",
            job["case_id"],
            {"job_id": job_id, "steps_completed": len(job["steps"])},
        )
        return job

    def _execute(self, case_id: str, step: str) -> None:
        if step == "mark_withdrawn":
            case = self.db.get_case(case_id, include_deleted=True)
            if case is not None and case.consent_withdrawn_at is None:
                case.consent_withdrawn_at = datetime.now(timezone.utc)
            if case is not None:
                case.status = CaseStatus.withdrawn
                case.workflow_status = Outcome.unassessed
                self.db.save_case(case)
        elif step == "delete_sources":
            self.db.purge_case_sources(case_id)
        elif step == "delete_facts":
            self.db.purge_case_facts(case_id)
        elif step == "delete_derived":
            self.db.purge_case_derived(case_id)
        elif step == "delete_documents":
            for document in self.db.get_documents(case_id):
                try:
                    self.documents.delete(document["stored_name"])
                finally:
                    self.db.mark_document_deleted(document["document_id"])
        elif step == "purge_uploads":
            prefix = f"{case_id}_"
            for path in self.documents.settings.uploads_dir.glob(f"{prefix}*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
        elif step == "finalize":
            self.db.finalize_case_deletion(case_id)
        else:
            raise ValueError(f"unknown deletion step: {step}")
