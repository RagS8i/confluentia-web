"""Explainable queue prioritization and operational wait estimates.

Priority categories are immutable here: a score can order cases *within* a
category, but it can never move a lower category above an emergency or clear a
warning. The service is read-only with respect to cases and assignments.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Settings
from .db import Database
from .schemas import (
    BottleneckSummary,
    CaseRecord,
    CaseStatus,
    Outcome,
    QueueIntelligence,
    QueueItem,
    QueueRecommendation,
    StaffingState,
    WaitEstimate,
)
from .triage.engine import RuleEngine

_TERMINAL = {CaseStatus.reviewed, CaseStatus.withdrawn, CaseStatus.deleted}


class QueueOperationsService:
    def __init__(self, settings: Settings, db: Database, engine: RuleEngine):
        self.settings = settings
        self.db = db
        self.engine = engine

    def update_staffing(
        self,
        actor: str,
        role: str,
        clinicians_available: int,
        intake_available: int,
        average_review_minutes: Optional[float],
        sample_size: int,
        note: Optional[str] = None,
    ) -> StaffingState:
        state = StaffingState(
            clinicians_available=max(0, clinicians_available),
            intake_available=max(0, intake_available),
            average_review_minutes=average_review_minutes,
            sample_size=max(0, sample_size),
            updated_by=actor,
            updated_at=datetime.now(timezone.utc),
            note=note,
        )
        self.db.save_operations_state(state)
        self.db.audit(
            actor,
            role,
            "staffing_state_updated",
            None,
            {
                "clinicians_available": state.clinicians_available,
                "intake_available": state.intake_available,
                "average_review_minutes": state.average_review_minutes,
                "sample_size": state.sample_size,
            },
        )
        return state

    def staffing(self) -> StaffingState:
        return self.db.get_operations_state()

    def _snapshot(self, case: CaseRecord) -> Tuple[QueueItem, Dict]:
        db = self.db
        now = datetime.now(timezone.utc)
        warnings = db.get_warnings(case.case_id)
        summary = db.get_summary(case.case_id)
        discrepancies = db.get_discrepancies(case.case_id)
        sources = db.get_sources(case.case_id, _NoCipher())
        note = db.get_note(case.case_id)
        reviews = db.get_reviews(case.case_id)
        awaiting_raw = case.meta.get("awaiting_review_since")
        awaiting_minutes = None
        if awaiting_raw:
            try:
                awaiting = datetime.fromisoformat(awaiting_raw)
                if awaiting.tzinfo is None:
                    awaiting = awaiting.replace(tzinfo=timezone.utc)
                awaiting_minutes = max(0, int((now - awaiting).total_seconds() // 60))
            except ValueError:
                awaiting_minutes = None
        deadline = (case.meta.get("review_deadlines_minutes") or {}).get(
            case.workflow_status.value
        )
        overdue = bool(
            deadline is not None
            and awaiting_minutes is not None
            and awaiting_minutes > int(deadline)
        )
        failures = len(case.meta.get("processing_failures") or []) + len(
            case.meta.get("stage_failures") or []
        )
        review_status = "pending"
        if note and note.status.value == "approved":
            review_status = "approved"
        elif reviews:
            review_status = "in_review"
        consent = "granted"
        if case.consent_withdrawn_at:
            consent = "withdrawn"
        elif case.consent_granted_at is None:
            consent = "required"
        item = QueueItem(
            case_id=case.case_id,
            patient_id=case.patient_id,
            status=case.status,
            workflow_status=case.workflow_status,
            received_at=case.received_at,
            awaiting_review_since=None,
            awaiting_minutes=awaiting_minutes,
            review_deadline_minutes=int(deadline) if deadline is not None else None,
            overdue=overdue,
            source_modalities=sorted({s.modality.value for s in sources}),
            warning_labels=[w.label for w in warnings if w.kind.value == "rule"][:6],
            warning_count=len(warnings),
            missing_info_count=len(
                [
                    gap
                    for gap in (summary.gaps if summary else [])
                    if gap.code != "QUESTIONS_SUPPRESSED_EMERGENCY"
                ]
            ),
            contradiction_count=len(discrepancies),
            processing_failures=failures,
            assigned_reviewer=case.assigned_reviewer,
            review_status=review_status,
            consent=consent,
            urgency_score=case.urgency.score if case.urgency else None,
            urgency_source=case.urgency.source.value if case.urgency else None,
        )
        extra = {"warnings": warnings, "deadline": deadline}
        return item, extra

    def _score(self, item: QueueItem, deadline: Optional[int]) -> Tuple[float, Dict[str, float], List[str]]:
        factors: Dict[str, float] = {}
        explanations: List[str] = []
        if item.overdue:
            factors["overdue"] = 40.0
            explanations.append("Review deadline has passed.")
        elif deadline and item.awaiting_minutes is not None:
            ratio = min(1.0, item.awaiting_minutes / max(1, int(deadline)))
            if ratio > 0:
                factors["deadline_pressure"] = round(ratio * 25.0, 2)
                explanations.append("Case is approaching its configured review deadline.")
        if item.warning_count:
            factors["unresolved_warnings"] = float(min(16, item.warning_count * 3))
            explanations.append(f"{item.warning_count} warning(s) remain visible; none are auto-cleared.")
        if item.contradiction_count:
            factors["contradictions"] = float(min(10, item.contradiction_count * 2))
            explanations.append("Candidate contradictions need human verification.")
        if item.processing_failures:
            factors["processing_failures"] = float(min(20, item.processing_failures * 8))
            explanations.append("Processing failures may hide information; review source coverage.")
        if not item.assigned_reviewer:
            factors["unassigned"] = 5.0
            explanations.append("No clinician is assigned.")
        if item.awaiting_minutes:
            factors["waiting_time"] = float(min(15, item.awaiting_minutes / 30.0))
        if item.urgency_score is not None:
            explanations.append(
                f"Advisory urgency {item.urgency_score}/100 "
                f"({item.urgency_source or 'unknown'}); a clinician can edit it."
            )
        if not explanations:
            explanations.append("Routine prioritization within the existing workflow category.")
        return round(sum(factors.values()), 2), factors, explanations

    def _queue(self) -> List[Tuple[QueueItem, Dict]]:
        return [
            self._snapshot(case)
            for case in self.db.list_cases()
            if case.consent_withdrawn_at is None
        ]

    def wait_estimate(
        self,
        case: CaseRecord,
        queue: Optional[Sequence[Tuple[QueueItem, Dict]]] = None,
    ) -> WaitEstimate:
        if case.workflow_status == Outcome.emergency:
            return WaitEstimate(
                available=False,
                basis="Emergency cases use immediate human escalation; a wait estimate is not appropriate.",
                confidence="unavailable",
            )
        state = self.staffing()
        if (
            state.clinicians_available < 1
            or state.average_review_minutes is None
            or state.average_review_minutes <= 0
            or state.sample_size < self.settings.wait_estimate_min_sample
        ):
            return WaitEstimate(
                available=False,
                basis=(
                    "Unavailable: staffing and a sufficient operational review-time sample "
                    "are required before estimating a wait."
                ),
                confidence="unavailable",
                sample_size=state.sample_size,
            )
        queue = queue if queue is not None else self._queue()
        rank = self.engine.rank
        eligible = [
            item
            for item, _ in queue
            if item.status not in _TERMINAL
            and rank.get(item.workflow_status.value, 9)
            <= rank.get(case.workflow_status.value, 9)
        ]
        backlog = max(1, len(eligible))
        minutes = int(
            math.ceil(
                (backlog * float(state.average_review_minutes))
                / max(1, state.clinicians_available)
            )
        )
        low = max(1, int(math.floor(minutes * 0.75)))
        high = max(low, int(math.ceil(minutes * 1.25)))
        confidence = "low" if state.sample_size < 5 else ("medium" if state.sample_size < 10 else "high")
        return WaitEstimate(
            available=True,
            minimum_minutes=low,
            maximum_minutes=high,
            basis=(
                f"Operational estimate from {backlog} backlog case(s), "
                f"{state.clinicians_available} available clinician(s), and an average review time "
                f"derived from {state.sample_size} observations. It is not a queue position."
            ),
            confidence=confidence,
            sample_size=state.sample_size,
        )

    def estimate_for_case(self, case_id: str) -> WaitEstimate:
        case = self.db.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return self.wait_estimate(case)

    def intelligence(self) -> QueueIntelligence:
        queue = self._queue()
        rank = self.engine.rank
        state = self.staffing()
        recommendations: List[QueueRecommendation] = []
        for item, extra in queue:
            score, factors, explanations = self._score(item, extra.get("deadline"))
            priority_rank = rank.get(item.workflow_status.value, 9)
            if item.workflow_status == Outcome.emergency:
                action = "Escalate to a clinician now; do not wait for follow-up answers."
            elif item.overdue:
                action = "Review the case now; its configured deadline has passed."
            elif item.processing_failures:
                action = "Check processing failures and source coverage before clinical review."
            elif not item.assigned_reviewer:
                action = "Assign an authorized clinician and review warning evidence."
            else:
                action = "Continue clinician review within the current workflow category."
            recommendations.append(
                QueueRecommendation(
                    case_id=item.case_id,
                    patient_id=item.patient_id,
                    workflow_status=item.workflow_status,
                    status=item.status,
                    received_at=item.received_at,
                    awaiting_minutes=item.awaiting_minutes,
                    overdue=item.overdue,
                    priority_rank=priority_rank,
                    urgency_score=item.urgency_score,
                    urgency_source=item.urgency_source,
                    score=score,
                    factors=factors,
                    explanations=explanations,
                    recommended_action=action,
                    wait_estimate=self.wait_estimate(
                        self.db.get_case(item.case_id), queue
                    )
                    if self.db.get_case(item.case_id)
                    else None,
                )
            )
        recommendations.sort(
            key=lambda entry: (
                entry.priority_rank,
                -(entry.urgency_score if entry.urgency_score is not None else -1),
                -entry.score,
                entry.received_at,
            )
        )
        waiting = [
            item
            for item, _ in queue
            if item.status not in _TERMINAL
        ]
        bottlenecks = BottleneckSummary(
            total_waiting=len(waiting),
            overdue=sum(1 for item in waiting if item.overdue),
            unassigned=sum(1 for item in waiting if not item.assigned_reviewer),
            processing_failures=sum(1 for item in waiting if item.processing_failures),
            emergency=sum(
                1 for item in waiting if item.workflow_status == Outcome.emergency
            ),
            unassessed=sum(
                1 for item in waiting if item.workflow_status == Outcome.unassessed
            ),
            oldest_waiting_minutes=max(
                (item.awaiting_minutes or 0 for item in waiting), default=None
            )
            or None,
            pending_communications=self.db.count_communications({"draft", "approved", "failed"}),
            pending_followups=self.db.count_followups({"suggested", "approved", "sent"}),
            open_escalations=self.db.count_escalations({"pending", "sent", "failed"}),
        )
        return QueueIntelligence(
            items=recommendations,
            bottlenecks=bottlenecks,
            staffing=state,
        )


class _NoCipher:
    """Queue metadata needs only source modality, never decrypted text."""

    def decrypt_json(self, _value):
        return {"original_text": "", "normalized_text": ""}
