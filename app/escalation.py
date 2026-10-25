"""Immediate, deduplicated emergency escalation to the clinical outbox."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

from .config import Settings
from .db import Database
from .outbound import DeliveryError, deliver
from .schemas import (
    CaseRecord,
    EscalationRecord,
    EscalationSeverity,
    EscalationStatus,
    Outcome,
    Warning,
)
from .storage import random_id, safe_error


class EscalationError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class EscalationService:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    def raise_for_case(
        self,
        case: CaseRecord,
        warnings: Sequence[Warning],
        actor: str = "system",
    ) -> List[EscalationRecord]:
        """Create escalations before optional summary or message work runs."""
        if case.workflow_status != Outcome.emergency:
            return []
        emergency_warnings = [
            warning for warning in warnings if warning.outcome == Outcome.emergency
        ]
        if not emergency_warnings:
            emergency_warnings = []
        created: List[EscalationRecord] = []
        if not emergency_warnings:
            created.append(
                self._create(
                    case,
                    trigger_code="WORKFLOW_EMERGENCY",
                    trigger_label="Emergency workflow status requires immediate human review",
                    dedupe_key=f"{case.case_id}:WORKFLOW_EMERGENCY",
                    payload={"warning_ids": []},
                    actor=actor,
                )
            )
        for warning in emergency_warnings:
            dedupe = f"{case.case_id}:{warning.warning_id}:{warning.code}"
            created.append(
                self._create(
                    case,
                    trigger_code=warning.code,
                    trigger_label=warning.label,
                    dedupe_key=dedupe,
                    payload={
                        "warning_ids": [warning.warning_id],
                        "rule_id": warning.rule_id,
                        "component": warning.component,
                    },
                    actor=actor,
                )
            )
        return [item for item in created if item is not None]

    def _create(
        self,
        case: CaseRecord,
        trigger_code: str,
        trigger_label: str,
        dedupe_key: str,
        payload: dict,
        actor: str,
    ) -> Optional[EscalationRecord]:
        existing = self.db.find_escalation(dedupe_key)
        if existing is not None:
            return existing
        now = datetime.now(timezone.utc)
        record = EscalationRecord(
            escalation_id=random_id("esc"),
            case_id=case.case_id,
            trigger_code=trigger_code,
            trigger_label=trigger_label,
            status=EscalationStatus.pending,
            dedupe_key=dedupe_key,
            created_at=now,
            updated_at=now,
            delivery=self.settings.escalation_delivery,
            payload={
                **payload,
                "workflow_status": case.workflow_status.value,
                "case_received_at": case.received_at.isoformat(),
            },
        )
        self.db.save_escalation(record)
        record.attempts += 1
        try:
            record.delivery = deliver(
                self.settings,
                self.settings.escalation_delivery,
                {
                    "type": "urgent_escalation",
                    "escalation_id": record.escalation_id,
                    "case_id": record.case_id,
                    "trigger_code": record.trigger_code,
                    "trigger_label": record.trigger_label,
                    "workflow_status": case.workflow_status.value,
                    "created_at": record.created_at.isoformat(),
                },
            )
            record.status = EscalationStatus.sent
            record.last_error = None
        except DeliveryError as exc:
            record.status = EscalationStatus.failed
            record.last_error = safe_error(exc)
        record.updated_at = datetime.now(timezone.utc)
        self.db.save_escalation(record)
        self.db.audit(
            actor,
            "system",
            "emergency_escalation_created",
            case.case_id,
            {
                "escalation_id": record.escalation_id,
                "status": record.status.value,
                "delivery": record.delivery,
                "trigger_code": trigger_code,
            },
        )
        return record

    def retry(self, escalation_id: str, actor: str, role: str) -> EscalationRecord:
        record = self.db.get_escalation(escalation_id)
        if record is None:
            raise EscalationError("not_found", "Escalation not found.", 404)
        if record.status == EscalationStatus.acknowledged:
            raise EscalationError("already_acknowledged", "Escalation is already acknowledged.")
        record.attempts += 1
        try:
            record.delivery = deliver(
                self.settings,
                self.settings.escalation_delivery,
                {
                    "type": "urgent_escalation",
                    "escalation_id": record.escalation_id,
                    "case_id": record.case_id,
                    "trigger_code": record.trigger_code,
                    "trigger_label": record.trigger_label,
                    "workflow_status": record.payload.get("workflow_status"),
                    "retry": True,
                },
            )
            record.status = EscalationStatus.sent
            record.last_error = None
        except DeliveryError as exc:
            record.status = EscalationStatus.failed
            record.last_error = safe_error(exc)
        record.updated_at = datetime.now(timezone.utc)
        self.db.save_escalation(record)
        self.db.audit(
            actor,
            role,
            "emergency_escalation_retried",
            record.case_id,
            {"escalation_id": escalation_id, "status": record.status.value},
        )
        return record

    def acknowledge(
        self, escalation_id: str, actor: str, role: str, note: Optional[str] = None
    ) -> EscalationRecord:
        record = self.db.get_escalation(escalation_id)
        if record is None:
            raise EscalationError("not_found", "Escalation not found.", 404)
        if record.status == EscalationStatus.acknowledged:
            return record
        record.status = EscalationStatus.acknowledged
        record.acknowledged_by = actor
        record.acknowledged_at = datetime.now(timezone.utc)
        record.updated_at = datetime.now(timezone.utc)
        if note:
            record.payload["acknowledgement_note"] = note
        self.db.save_escalation(record)
        self.db.audit(
            actor,
            role,
            "emergency_escalation_acknowledged",
            record.case_id,
            {"escalation_id": escalation_id, "note_recorded": bool(note)},
        )
        return record

    def list(
        self, case_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[EscalationRecord]:
        return self.db.get_escalations(case_id=case_id, status=status)
