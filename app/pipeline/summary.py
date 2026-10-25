"""Stage G - evidence-linked clinical summary generation.

Structured sections are built deterministically from validated facts.  The SLM
may draft narrative text, but its output is validated and clearly labelled as
AI-generated; it can never become clinician-approved automatically.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import Settings
from ..model.provider import BaseProvider, ModelError
from ..schemas import (
    Assertion,
    ClinicalSummary,
    Discrepancy,
    DraftNote,
    Fact,
    FactStatus,
    FactType,
    FollowUpQuestion,
    Gap,
    Outcome,
    SourceRecord,
    SourceRef,
    SummarySection,
    TimelineEntry,
    VerificationStatus,
    Warning,
)
from ..storage import sha256_text

_DISCLAIMERS = [
    "CareFlow AI is a documentation and workflow-support tool. It does not diagnose patients, prescribe treatment, or approve care.",
    "Workflow statuses are demonstration rule outcomes, not a validated clinical triage scale.",
    "A qualified healthcare professional must verify evidence, resolve contradictions, and make the final clinical decision.",
    "AI-generated text must not be represented as clinician-approved until an authorized reviewer approves it.",
]

_BLOCKED_NOTE = re.compile(
    r"(?i)\b(diagnosis\s+is|you\s+have|you\s+are\s+safe|no\s+need\s+for\s+(medical|doctor|hospital)|"
    r"nothing\s+to\s+worry|not\s+serious|safe\s+to\s+(go|discharge)|confirmed\s+diagnosis|"
    r"prescribe|take\s+\d+\s*mg|discharge\s+home)\b"
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _dedupe_refs(refs: Iterable[SourceRef]) -> List[SourceRef]:
    seen = set()
    result = []
    for ref in refs:
        key = (ref.source_id, ref.segment_id, ref.start_char, ref.end_char)
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result


def _fact_refs(facts: Sequence[Fact]) -> List[SourceRef]:
    return _dedupe_refs([ref for fact in facts for ref in fact.source_refs])


def _fact_item(fact: Fact) -> str:
    label = fact.label
    value = fact.value if fact.value is not None else fact.value_normalized
    unit = f" {fact.unit}" if fact.unit else ""
    if fact.status == FactStatus.absent:
        text = f"Explicitly denied: {label}"
    elif fact.status == FactStatus.historical:
        text = f"Historical: {label}" + (f" — {value}{unit}" if value else "")
    elif fact.status == FactStatus.uncertain:
        text = f"Uncertain: {label}" + (f" — {value}{unit}" if value else "")
    elif fact.status == FactStatus.unknown:
        text = f"Status unknown: {label}"
    else:
        text = f"{label}" + (f": {value}{unit}" if value is not None else "")
    if fact.temporal and fact.temporal.raw_expression:
        text += f" (reported time: {fact.temporal.raw_expression})"
    markers = []
    if fact.assertion == Assertion.patient_reported:
        markers.append("patient-reported")
    elif fact.assertion == Assertion.document_derived:
        markers.append("document-derived")
    elif fact.assertion in (Assertion.ai_generated, Assertion.inferred_for_review):
        markers.append("AI-inferred")
    if fact.verification_status != VerificationStatus.verified:
        markers.append(fact.verification_status.value)
    if fact.ai_suggestion:
        markers.append("not an observed fact")
    if markers:
        text += " [" + ", ".join(markers) + "]"
    return text


def _section_status(facts: Sequence[Fact], default: str) -> str:
    if not facts:
        return "missing"
    statuses = {f.verification_status.value for f in facts}
    if statuses == {"verified"}:
        return "verified"
    if any(f.ai_suggestion for f in facts):
        return "ai_generated"
    if any(f.status == FactStatus.uncertain for f in facts):
        return "uncertain"
    if "verified" in statuses:
        return "mixed"
    return default


def _deterministic_note(
    facts: Sequence[Fact],
    warnings: Sequence[Warning],
    gaps: Sequence[Gap],
    outcome: Outcome,
    rule_meta: Dict[str, Any],
) -> str:
    symptoms = [f for f in facts if f.fact_type == FactType.symptom and f.status == FactStatus.present]
    traceable = [f for f in symptoms if f.source_refs and not f.ai_suggestion]
    denied = [f for f in facts if f.status == FactStatus.absent]
    history = [f for f in facts if f.fact_type == FactType.history]
    measurements = [f for f in facts if f.fact_type in (FactType.measurement, FactType.vital_sign)]
    rule_warnings = [w for w in warnings if w.kind.value == "rule"]
    complaint = traceable[0].label if traceable else "not clearly stated"

    def labels(items: Sequence[Fact]) -> str:
        return ", ".join(dict.fromkeys(item.label for item in items)) or "none extracted"

    measurement_text = ", ".join(
        f"{f.label} {f.value} {f.unit or ''}".strip() for f in measurements
    ) or "none extracted"
    missing_text = ", ".join(
        dict.fromkeys(g.code for g in gaps if g.code != "QUESTIONS_SUPPRESSED_EMERGENCY")
    ) or "none identified"
    warning_text = ", ".join(
        (w.rule_id or w.code) for w in rule_warnings
    ) or "none triggered"
    lines = [
        "AI DRAFT - FOR CLINICIAN REVIEW",
        f"Presenting complaint: {complaint}.",
        f"Patient-reported symptoms: {labels([f for f in symptoms if f.assertion == Assertion.patient_reported])}.",
        f"Explicitly denied: {labels(denied)}.",
        f"Relevant history extracted: {labels(history)}.",
        f"Measurements: {measurement_text}.",
        f"Missing information: {missing_text}.",
        f"Warning signals: {warning_text}.",
        f"Workflow status (demonstration rules {rule_meta.get('config_id')} v{rule_meta.get('config_version')}): {outcome.value}. This is not a diagnosis or clinical clearance.",
        "All findings require clinician verification against the original sources before use.",
    ]
    return "\n".join(lines)


def _valid_note(note: str, facts: Sequence[Fact]) -> tuple[bool, str]:
    if not note or len(note.strip()) < 20:
        return False, "note too short"
    if _BLOCKED_NOTE.search(note):
        return False, "note contained diagnostic, prescriptive or reassurance language"
    fact_text = " ".join(
        " ".join(
            str(x)
            for x in (
                f.label,
                f.value,
                f.value_normalized,
                f.unit,
                f.original_text,
                f.normalized_text,
            )
        )
        for f in facts
    )
    allowed_numbers = set(_NUMBER_RE.findall(fact_text))
    for number in _NUMBER_RE.findall(note):
        if number not in allowed_numbers:
            return False, f"note contained unsupported numeric value {number}"
    return True, ""


def generate_summary(
    *,
    case_id: str,
    facts: Sequence[Fact],
    sources: Sequence[SourceRecord],
    timeline: Sequence[TimelineEntry],
    discrepancies: Sequence[Discrepancy],
    warnings: Sequence[Warning],
    gaps: Sequence[Gap],
    questions: Sequence[FollowUpQuestion],
    outcome: Outcome,
    rule_meta: Dict[str, Any],
    provider: BaseProvider,
    settings: Settings,
) -> ClinicalSummary:
    visible_facts = [f for f in facts if f.verification_status != VerificationStatus.rejected]
    symptoms = [f for f in visible_facts if f.fact_type == FactType.symptom]
    patient_reported = [f for f in visible_facts if f.assertion == Assertion.patient_reported]
    document_derived = [f for f in visible_facts if f.assertion == Assertion.document_derived]
    measurements = [
        f
        for f in visible_facts
        if f.fact_type in (FactType.measurement, FactType.vital_sign)
    ]
    temporal_facts = [f for f in visible_facts if f.temporal and f.temporal.raw_expression]

    present_symptoms = [
        f
        for f in symptoms
        if f.status == FactStatus.present and f.source_refs and not f.ai_suggestion
    ]
    complaint_fact = present_symptoms[0] if present_symptoms else None
    presenting = SummarySection(
        key="presenting_complaint",
        title="Presenting complaint",
        items=[complaint_fact.label if complaint_fact else "Not clearly stated in the available sources."],
        evidence_fact_ids=[complaint_fact.fact_id] if complaint_fact else [],
        source_refs=complaint_fact.source_refs if complaint_fact else [],
        epistemic_status=(
            _section_status([complaint_fact], "patient_reported")
            if complaint_fact
            else "missing"
        ),
    )

    sections: List[SummarySection] = []

    def add_section(
        key: str,
        title: str,
        items: List[str],
        evidence: Sequence[Fact],
        status: str,
        empty_note: str = "None extracted.",
    ) -> None:
        sections.append(
            SummarySection(
                key=key,
                title=title,
                items=items or [empty_note],
                evidence_fact_ids=sorted({f.fact_id for f in evidence}),
                source_refs=_fact_refs(evidence),
                epistemic_status=status,
            )
        )

    add_section(
        "patient_reported_symptoms",
        "Patient-reported symptoms",
        [_fact_item(f) for f in patient_reported if f.fact_type in (FactType.symptom, FactType.medication, FactType.history)],
        [f for f in patient_reported if f.fact_type in (FactType.symptom, FactType.medication, FactType.history)],
        _section_status(
            [f for f in patient_reported if f.fact_type in (FactType.symptom, FactType.medication, FactType.history)],
            "patient_reported",
        ),
    )
    add_section(
        "document_findings",
        "Information explicitly extracted from supplied records",
        [_fact_item(f) for f in document_derived],
        document_derived,
        _section_status(document_derived, "document_derived"),
        "No document-derived facts were extracted.",
    )

    onset_items: List[str] = []
    onset_facts: List[Fact] = []
    for fact in temporal_facts:
        if fact.temporal is None:
            continue
        onset_facts.append(fact)
        normalized = fact.temporal.normalized_expression or "not normalized"
        onset_items.append(
            f"{fact.label}: source expression '{fact.temporal.raw_expression}' "
            f"(normalized: {normalized}; certainty: {fact.temporal.certainty})"
        )
    for entry in timeline:
        if entry.progression:
            onset_items.append(f"{entry.label}: reported progression '{entry.progression}'")
    add_section(
        "onset_and_progression",
        "Symptom onset and progression",
        onset_items,
        onset_facts,
        _section_status(onset_facts, "uncertain"),
        "No onset, duration or progression information was extracted.",
    )

    add_section(
        "measurements",
        "Important reported measurements and units",
        [
            f"{f.label}: {f.value} {f.unit or '(unit not provided)'} "
            f"[{f.assertion.value}, {f.verification_status.value}]"
            for f in measurements
        ],
        measurements,
        _section_status(measurements, "unverified"),
        "No measurements with units were extracted.",
    )

    uncertain_facts = [
        f
        for f in visible_facts
        if f.status == FactStatus.uncertain or f.verification_status != VerificationStatus.verified or not f.source_refs
    ]
    missing_items = [f"{g.code}: {g.description}" for g in gaps]
    missing_items += [
        f"Uncertain claim: {_fact_item(f)} — {f.review_note or 'review required'}"
        for f in uncertain_facts
    ]
    add_section(
        "missing_or_uncertain",
        "Missing or uncertain information",
        missing_items,
        uncertain_facts,
        "missing" if missing_items else "verified",
        "No missing or uncertain information was identified by the configured checks.",
    )

    contradiction_items = [
        f"{d.category.value}: {d.summary} — {d.explanation}" for d in discrepancies
    ]
    add_section(
        "candidate_contradictions",
        "Candidate contradictions",
        contradiction_items,
        [f for f in visible_facts if any(d.fact_ids and f.fact_id in d.fact_ids for d in discrepancies)],
        "uncertain" if discrepancies else "verified",
        "No candidate contradictions detected.",
    )

    warning_items = []
    warning_facts: List[Fact] = []
    for warning in warnings:
        rule = f" [{warning.rule_id} v{warning.rule_version}]" if warning.rule_id else f" [{warning.code}]"
        warning_items.append(f"{warning.label}{rule}: {warning.explanation}")
        warning_facts.extend(f for f in visible_facts if f.fact_id in warning.evidence_fact_ids)
    add_section(
        "warning_signals",
        "Potential warning signals and supporting evidence",
        warning_items,
        warning_facts,
        "uncertain" if warning_items else "verified",
        "No configured warning signal was triggered; this is not a clinical clearance.",
    )

    add_section(
        "follow_up_questions",
        "Suggested follow-up questions",
        [q.text for q in questions],
        [],
        "ai_generated" if questions else "missing",
        "No follow-up questions suggested.",
    )

    # Optional model narrative, validated before display.
    narrative_section = None
    draft_note_text = ""
    narrative_notes: List[str] = []
    if visible_facts:
        payload = {
            "case_id": case_id,
            "workflow_status": outcome.value,
            "facts": [
                {
                    "fact_id": f.fact_id,
                    "fact_type": f.fact_type.value,
                    "label": f.label,
                    "status": f.status.value,
                    "assertion": f.assertion.value,
                    "value": f.value,
                    "unit": f.unit,
                    "temporal": f.temporal.raw_expression if f.temporal else None,
                    "verification_status": f.verification_status.value,
                }
                for f in visible_facts
            ],
            "warnings": [
                {"rule_id": w.rule_id, "code": w.code, "label": w.label, "outcome": w.outcome.value if w.outcome else None}
                for w in warnings
            ],
            "gaps": [g.code for g in gaps],
        }
        try:
            drafted = provider.draft_summary(payload, settings)
            narrative = (drafted.get("summary_narrative") or "").strip()
            note = (drafted.get("draft_referral_note") or "").strip()
            ok, reason = _valid_note(note, visible_facts)
            if ok:
                draft_note_text = note
            elif note:
                narrative_notes.append(f"Model draft rejected: {reason}")
            if narrative:
                narrative_section = SummarySection(
                    key="ai_narrative",
                    title="AI-generated summary narrative (not clinician-approved)",
                    items=[narrative],
                    evidence_fact_ids=sorted({f.fact_id for f in visible_facts}),
                    source_refs=_fact_refs(visible_facts),
                    epistemic_status="ai_generated",
                )
        except ModelError as exc:
            narrative_notes.append(f"Model summary draft unavailable: {exc.code}")
        except Exception as exc:  # pragma: no cover - defensive
            narrative_notes.append(f"Model summary draft failed: {type(exc).__name__}")

    if not draft_note_text:
        draft_note_text = _deterministic_note(visible_facts, warnings, gaps, outcome, rule_meta)
    elif not draft_note_text.upper().startswith("AI DRAFT"):
        draft_note_text = "AI DRAFT - FOR CLINICIAN REVIEW\n" + draft_note_text
    if narrative_notes:
        draft_note_text += "\n\nDocumentation warnings: " + "; ".join(narrative_notes)

    now = datetime.now(timezone.utc)
    note = DraftNote(
        note_id="note_" + sha256_text(case_id + draft_note_text)[:16],
        text=draft_note_text,
        status="draft_ai",
        ai_generated=True,
        updated_at=now,
        history=[],
    )

    note_section = SummarySection(
        key="draft_clinician_note",
        title="Draft clinician-facing note (AI-generated, editable, not approved)",
        items=[draft_note_text],
        evidence_fact_ids=sorted({f.fact_id for f in visible_facts}),
        source_refs=_fact_refs(visible_facts),
        epistemic_status="ai_generated",
    )
    sections.append(note_section)
    if narrative_section:
        sections.insert(5, narrative_section)

    with_source = [f for f in visible_facts if f.source_refs]
    coverage = {
        "total_facts": len(visible_facts),
        "facts_with_source_reference": len(with_source),
        "facts_without_source_reference": len(visible_facts) - len(with_source),
        "verified_facts": len([f for f in visible_facts if f.verification_status == VerificationStatus.verified]),
        "unverified_facts": len([f for f in visible_facts if f.verification_status == VerificationStatus.unverified]),
        "ai_inferred_facts": len([f for f in visible_facts if f.ai_suggestion]),
        "evidence_coverage_ratio": (
            round(len(with_source) / len(visible_facts), 3) if visible_facts else 0.0
        ),
    }

    return ClinicalSummary(
        case_id=case_id,
        generated_at=now,
        pipeline_version=settings.pipeline_version,
        model=provider.model,
        presenting_complaint=presenting,
        sections=sections,
        timeline=list(timeline),
        discrepancies=list(discrepancies),
        warnings=[w for w in warnings if w.kind.value == "rule"],
        component_warnings=[w for w in warnings if w.kind.value != "rule"],
        gaps=list(gaps),
        follow_up_questions=list(questions),
        workflow_status=outcome,
        rule_config_id=rule_meta.get("config_id"),
        rule_config_version=rule_meta.get("config_version"),
        rule_config_status=rule_meta.get("config_status"),
        draft_note=note,
        evidence_coverage=coverage,
        disclaimers=_DISCLAIMERS,
    )
