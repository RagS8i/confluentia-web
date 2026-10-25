"""Advisory urgency scoring.

The SLM produces a 0-100 operational urgency score after all currently
available information has been processed. The score is deliberately advisory:
it never changes the deterministic workflow status, never dismisses a warning,
and never replaces clinician review. Clinician edits are preserved across
reprocessing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Sequence

from ..config import Settings
from ..model.provider import BaseProvider
from ..schemas import (
    Discrepancy,
    Fact,
    Gap,
    Outcome,
    TimelineEntry,
    UrgencyAssessment,
    UrgencySource,
    VerificationStatus,
    Warning,
)

_FALLBACK_SCORES = {
    Outcome.emergency: 100,
    Outcome.priority_review: 75,
    Outcome.unassessed: 50,
    Outcome.routine_review: 20,
}


def _coerce_score(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("urgency_score must be a number")
    score = int(round(float(value)))
    if score < 0 or score > 100:
        raise ValueError("urgency_score out of range")
    return score


def _fallback_score(outcome: Outcome) -> int:
    return _FALLBACK_SCORES.get(outcome, 50)


def _build_payload(
    *,
    case_id: str,
    facts: Sequence[Fact],
    warnings: Sequence[Warning],
    gaps: Sequence[Gap],
    discrepancies: Sequence[Discrepancy],
    timeline: Sequence[TimelineEntry],
    outcome: Outcome,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "workflow_status": outcome.value,
        "facts": [
            {
                "fact_id": fact.fact_id,
                "fact_type": fact.fact_type.value,
                "label": fact.label,
                "status": fact.status.value,
                "assertion": fact.assertion.value,
                "value": fact.value,
                "unit": fact.unit,
                "temporal": fact.temporal.raw_expression if fact.temporal else None,
                "verification_status": fact.verification_status.value,
                "confidence": fact.confidence,
            }
            for fact in facts
            if fact.verification_status != VerificationStatus.rejected
        ],
        "warnings": [
            {
                "code": warning.code,
                "label": warning.label,
                "outcome": warning.outcome.value if warning.outcome else None,
            }
            for warning in warnings
        ],
        "gaps": [gap.code for gap in gaps],
        "discrepancies": [
            {"category": item.category.value, "summary": item.summary}
            for item in discrepancies
        ],
        "timeline": [
            {"label": entry.label, "progression": entry.progression}
            for entry in timeline
        ],
    }


def assess_urgency(
    *,
    case_id: str,
    facts: Sequence[Fact],
    warnings: Sequence[Warning],
    gaps: Sequence[Gap],
    discrepancies: Sequence[Discrepancy],
    timeline: Sequence[TimelineEntry],
    outcome: Outcome,
    provider: BaseProvider,
    settings: Settings,
    existing: UrgencyAssessment | None = None,
) -> UrgencyAssessment:
    """Return the current urgency assessment.

    A clinician-edited score is never overwritten by an automated run. Callers
    that want a fresh SLM assessment must explicitly clear or replace it.
    """
    if existing is not None and existing.source == UrgencySource.clinician:
        return existing

    payload = _build_payload(
        case_id=case_id,
        facts=facts,
        warnings=warnings,
        gaps=gaps,
        discrepancies=discrepancies,
        timeline=timeline,
        outcome=outcome,
    )
    now = datetime.now(timezone.utc)
    try:
        raw = provider.assess_urgency(payload, settings)
        if not isinstance(raw, dict):
            raise ValueError("urgency response must be an object")
        score = _coerce_score(raw.get("urgency_score"))
        rationale = str(raw.get("rationale") or "").strip()[:1000]
        return UrgencyAssessment(
            score=score,
            rationale=rationale or "No rationale was returned; review the evidence.",
            source=UrgencySource.slm,
            model=provider.model,
            assessed_at=now,
            assessed_by="system",
            slm_score=score,
        )
    except Exception as exc:  # advisory: never break the clinical pipeline
        code = getattr(exc, "code", type(exc).__name__)
        score = _fallback_score(outcome)
        return UrgencyAssessment(
            score=score,
            rationale=(
                "Deterministic fallback score because the urgency model did not "
                f"return a valid assessment ({code}). This is not an SLM "
                "assessment; review and edit it."
            ),
            source=UrgencySource.fallback,
            model=provider.model,
            assessed_at=now,
            assessed_by="system",
            model_error=str(code),
        )
