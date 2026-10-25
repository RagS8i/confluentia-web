"""Typed clinical information model.

The model deliberately keeps the original wording next to every normalized
interpretation, and keeps AI inference separate from observed/patient-reported
facts.  It never represents missing information as a negative finding.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Modality(str, Enum):
    stt = "stt"
    ocr = "ocr"
    typed_message = "typed_message"
    upload = "upload"


class LanguageStatus(str, Enum):
    detected = "detected"
    declared = "declared"
    heuristic = "heuristic"
    unknown = "unknown"
    unsupported = "unsupported"


class FactType(str, Enum):
    symptom = "symptom"
    onset = "onset"
    duration = "duration"
    progression = "progression"
    measurement = "measurement"
    medication = "medication"
    medication_history = "medication_history"
    history = "history"
    vital_sign = "vital_sign"
    demographic = "demographic"
    explicit_negation = "explicit_negation"
    other = "other"


class FactStatus(str, Enum):
    present = "present"
    absent = "absent"
    historical = "historical"
    unknown = "unknown"
    uncertain = "uncertain"


class Assertion(str, Enum):
    patient_reported = "patient_reported"
    document_derived = "document_derived"
    inferred_for_review = "inferred_for_review"
    unknown = "unknown"
    ai_generated = "ai_generated"


class VerificationStatus(str, Enum):
    unverified = "unverified"
    verified = "verified"
    corrected = "corrected"
    disputed = "disputed"
    rejected = "rejected"


class DiscrepancyCategory(str, Enum):
    possible_direct_contradiction = "possible_direct_contradiction"
    different_reference_time = "different_reference_time"
    unit_mismatch = "unit_mismatch"
    potential_duplicate = "potential_duplicate"
    ambiguous_interpretation = "ambiguous_interpretation"
    unable_to_compare = "unable_to_compare"
    requires_human_verification = "requires_human_verification"


class Severity(str, Enum):
    info = "info"
    review = "review"
    high = "high"


class Outcome(str, Enum):
    emergency = "EMERGENCY"
    priority_review = "PRIORITY_REVIEW"
    routine_review = "ROUTINE_REVIEW"
    unassessed = "UNASSESSED"


class WarningKind(str, Enum):
    rule = "rule"
    component = "component"
    missing_information = "missing_information"


class WarningStatus(str, Enum):
    active = "active"
    acknowledged = "acknowledged"
    resolved = "resolved"


class NoteStatus(str, Enum):
    draft_ai = "draft_ai"
    clinician_edited = "clinician_edited"
    approved = "approved"


class CaseStatus(str, Enum):
    received = "received"
    consent_pending = "consent_pending"
    queued = "queued"
    processing = "processing"
    processed = "processed"
    partially_processed = "partially_processed"
    processing_failed = "processing_failed"
    awaiting_review = "awaiting_review"
    reviewed = "reviewed"
    withdrawn = "withdrawn"
    deleted = "deleted"


class ReviewerAction(str, Enum):
    verify = "verify"
    correct = "correct"
    reject = "reject"
    approve_note = "approve_note"
    edit_note = "edit_note"
    override_status = "override_status"
    assign = "assign"
    edit_urgency = "edit_urgency"
    acknowledge_warning = "acknowledge_warning"


class CommunicationIntent(str, Enum):
    status_update = "status_update"
    missing_information = "missing_information"
    document_request = "document_request"
    wait_update = "wait_update"
    urgent_review = "urgent_review"


class CommunicationStatus(str, Enum):
    draft = "draft"
    approved = "approved"
    sent = "sent"
    failed = "failed"
    dismissed = "dismissed"


class FollowUpStatus(str, Enum):
    suggested = "suggested"
    approved = "approved"
    sent = "sent"
    answered = "answered"
    dismissed = "dismissed"


class EscalationStatus(str, Enum):
    pending = "pending"
    sent = "sent"
    failed = "failed"
    acknowledged = "acknowledged"


class EscalationSeverity(str, Enum):
    emergency = "emergency"
    priority = "priority"


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    segment_id: str
    quote: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None


class Temporal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_expression: Optional[str] = None
    normalized_expression: Optional[str] = None
    offset_hours: Optional[float] = None
    estimated_start_at: Optional[datetime] = None
    precision: Optional[str] = None  # hour | day | week | month | unknown
    reference_basis: Optional[str] = None  # source_received_at | explicit_date | unknown
    certainty: str = "uncertain"  # certain | approximate | uncertain
    notes: Optional[str] = None


class ExtractionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    model: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    extracted_at: datetime = Field(default_factory=utcnow)
    retries: int = 0


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    case_id: str
    fact_type: FactType
    label: str
    value: Any = None
    value_normalized: Optional[str] = None
    unit: Optional[str] = None
    status: FactStatus = FactStatus.unknown
    assertion: Assertion = Assertion.unknown
    language: str = "unknown"
    language_status: LanguageStatus = LanguageStatus.unknown
    original_text: str = ""
    normalized_text: Optional[str] = None
    source_refs: List[SourceRef] = Field(default_factory=list)
    temporal: Optional[Temporal] = None
    extraction: ExtractionMeta
    verification_status: VerificationStatus = VerificationStatus.unverified
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    ai_suggestion: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_note: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


class ExtractedFact(BaseModel):
    """Shape returned by the language model.  `original_text` must be an exact
    quote from one of the supplied sources; attribution is verified afterward."""

    model_config = ConfigDict(extra="forbid")

    fact_type: FactType
    label: str
    value: Any = None
    status: FactStatus = FactStatus.unknown
    assertion: Assertion = Assertion.patient_reported
    original_text: str
    normalized_text: Optional[str] = None
    unit: Optional[str] = None
    symptom_onset: Optional[str] = None
    duration: Optional[str] = None
    progression: Optional[str] = None
    confidence: Optional[float] = None
    notes: Optional[str] = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: List[ExtractedFact] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    interpretation_notes: List[str] = Field(default_factory=list)
    summary_narrative: Optional[str] = None
    draft_referral_note: Optional[str] = None


class SourceRecord(BaseModel):
    """Normalized source segment.  Original text is stored encrypted; this
    model is only serialized with decrypted text in memory."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    parent_source_id: Optional[str] = None
    case_id: str
    modality: Modality
    segment_id: str
    original_text: str
    normalized_text: str
    language: str = "unknown"
    language_status: LanguageStatus = LanguageStatus.unknown
    confidence: Optional[float] = None
    content_hash: str = ""
    document_ref: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None
    captured_at: Optional[datetime] = None
    received_at: datetime = Field(default_factory=utcnow)
    usable: bool = True
    warnings: List[str] = Field(default_factory=list)
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class Warning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warning_id: str
    case_id: str
    kind: WarningKind
    code: str
    label: str
    explanation: str = ""
    outcome: Optional[Outcome] = None
    rule_id: Optional[str] = None
    rule_version: Optional[str] = None
    evidence_fact_ids: List[str] = Field(default_factory=list)
    source_refs: List[SourceRef] = Field(default_factory=list)
    status: WarningStatus = WarningStatus.active
    component: str = "pipeline"
    created_at: datetime = Field(default_factory=utcnow)


class Discrepancy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discrepancy_id: str
    case_id: str
    category: DiscrepancyCategory
    severity: Severity = Severity.review
    summary: str
    explanation: str = ""
    fact_ids: List[str] = Field(default_factory=list)
    source_refs: List[SourceRef] = Field(default_factory=list)
    requires_human_verification: bool = True
    review_status: str = "open"
    created_at: datetime = Field(default_factory=utcnow)


class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    fact_id: str
    label: str
    raw_time: Optional[str] = None
    normalized_time: Optional[str] = None
    offset_hours: Optional[float] = None
    precision: Optional[str] = None
    reference_basis: Optional[str] = None
    estimated_start_at: Optional[datetime] = None
    certainty: str = "uncertain"
    progression: Optional[str] = None
    source_refs: List[SourceRef] = Field(default_factory=list)


class Gap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    description: str
    priority: str = "review"  # high | review | info
    related_fact_ids: List[str] = Field(default_factory=list)


class FollowUpQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    text: str
    language: str = "en"
    addresses: List[str] = Field(default_factory=list)
    rationale: str = ""
    status: str = "suggested"


class SummarySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    items: List[str] = Field(default_factory=list)
    evidence_fact_ids: List[str] = Field(default_factory=list)
    source_refs: List[SourceRef] = Field(default_factory=list)
    epistemic_status: str = "unverified"
    # observed | patient_reported | document_derived | ai_generated | missing | uncertain


class DraftNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_id: str
    text: str
    status: NoteStatus = NoteStatus.draft_ai
    version: int = 1
    ai_generated: bool = True
    edited_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=utcnow)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class ClinicalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    pipeline_version: str
    model: Optional[str] = None
    presenting_complaint: Optional[SummarySection] = None
    sections: List[SummarySection] = Field(default_factory=list)
    timeline: List[TimelineEntry] = Field(default_factory=list)
    discrepancies: List[Discrepancy] = Field(default_factory=list)
    warnings: List[Warning] = Field(default_factory=list)
    component_warnings: List[Warning] = Field(default_factory=list)
    gaps: List[Gap] = Field(default_factory=list)
    follow_up_questions: List[FollowUpQuestion] = Field(default_factory=list)
    workflow_status: Outcome = Outcome.unassessed
    rule_config_id: Optional[str] = None
    rule_config_version: Optional[str] = None
    rule_config_status: Optional[str] = None
    draft_note: Optional[DraftNote] = None
    evidence_coverage: Dict[str, Any] = Field(default_factory=dict)
    disclaimers: List[str] = Field(default_factory=list)


class UrgencySource(str, Enum):
    slm = "slm"
    clinician = "clinician"
    fallback = "fallback"


class UrgencyEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=100)
    previous_score: Optional[int] = Field(default=None, ge=0, le=100)
    edited_by: str
    role: str
    reason: str
    edited_at: datetime = Field(default_factory=utcnow)


class UrgencyAssessment(BaseModel):
    """Advisory 0-100 urgency. Never a diagnosis, clearance or workflow status."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=100)
    rationale: str = Field(default="", max_length=1000)
    source: UrgencySource = UrgencySource.slm
    model: Optional[str] = None
    assessed_at: datetime = Field(default_factory=utcnow)
    assessed_by: Optional[str] = None
    slm_score: Optional[int] = Field(default=None, ge=0, le=100)
    edited_by: Optional[str] = None
    edited_at: Optional[datetime] = None
    edit_reason: Optional[str] = None
    model_error: Optional[str] = None
    history: List[UrgencyEdit] = Field(default_factory=list)


class CaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    patient_id: str
    created_at: datetime = Field(default_factory=utcnow)
    received_at: datetime = Field(default_factory=utcnow)
    status: CaseStatus = CaseStatus.received
    workflow_status: Outcome = Outcome.unassessed
    assigned_reviewer: Optional[str] = None
    consent_granted_at: Optional[datetime] = None
    consent_scope: Optional[str] = None
    consent_withdrawn_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    language_hint: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    urgency: Optional[UrgencyAssessment] = None

    @property
    def consent_granted(self) -> bool:
        return self.consent_granted_at is not None and self.consent_withdrawn_at is None

    @property
    def deleted(self) -> bool:
        return self.deleted_at is not None


class ProcessingRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    stage: str
    status: str  # pending | running | succeeded | failed | skipped
    attempt: int = 1
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    case_id: str
    action: ReviewerAction
    actor: str
    role: str
    original_status: Optional[Outcome] = None
    final_status: Optional[Outcome] = None
    reason: Optional[str] = None
    detail: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class QueueItem(BaseModel):
    case_id: str
    patient_id: str
    status: CaseStatus
    workflow_status: Outcome
    received_at: datetime
    awaiting_review_since: Optional[datetime] = None
    awaiting_minutes: Optional[int] = None
    review_deadline_minutes: Optional[int] = None
    overdue: bool = False
    source_modalities: List[str] = Field(default_factory=list)
    warning_labels: List[str] = Field(default_factory=list)
    warning_count: int = 0
    missing_info_count: int = 0
    contradiction_count: int = 0
    processing_failures: int = 0
    assigned_reviewer: Optional[str] = None
    review_status: str = "pending"
    consent: str = "unknown"
    urgency_score: Optional[int] = None
    urgency_source: Optional[str] = None


class PatientMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    case_id: str
    intent: CommunicationIntent
    language: str
    template_id: str
    body: str
    status: CommunicationStatus = CommunicationStatus.draft
    ai_generated: bool = True
    generated_by: str
    created_at: datetime = Field(default_factory=utcnow)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivery: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FollowUpRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    followup_id: str
    case_id: str
    question_id: str
    text: str
    language: str = "en"
    addresses: List[str] = Field(default_factory=list)
    rationale: str = ""
    status: FollowUpStatus = FollowUpStatus.suggested
    origin: str = "configured_gap_check"
    created_by: str = "system"
    created_at: datetime = Field(default_factory=utcnow)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    dismissed_by: Optional[str] = None
    dismissed_at: Optional[datetime] = None
    delivery: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FollowUpAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str
    followup_id: str
    case_id: str
    text: str
    language: str = "unknown"
    source_id: Optional[str] = None
    answered_by: str
    role: str
    created_at: datetime = Field(default_factory=utcnow)
    # Audio answers keep the native transcript in ``text``, the English
    # translation here (encrypted at rest), and a link to the stored recording.
    translated_text: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EscalationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escalation_id: str
    case_id: str
    trigger_code: str
    trigger_label: str
    severity: EscalationSeverity = EscalationSeverity.emergency
    status: EscalationStatus = EscalationStatus.pending
    dedupe_key: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    attempts: int = 0
    delivery: str = "outbox"
    last_error: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class StaffingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinicians_available: int = 0
    intake_available: int = 0
    average_review_minutes: Optional[float] = None
    sample_size: int = 0
    updated_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    note: Optional[str] = None


class WaitEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    minimum_minutes: Optional[int] = None
    maximum_minutes: Optional[int] = None
    basis: str
    confidence: str = "unavailable"
    sample_size: int = 0
    disclaimer: str = (
        "Operational estimate only. It is not a queue position, a clinical assessment, "
        "or a guarantee of when a clinician will review the case."
    )


class QueueRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    patient_id: str
    workflow_status: Outcome
    status: CaseStatus
    received_at: datetime
    awaiting_minutes: Optional[int] = None
    overdue: bool = False
    priority_rank: int
    urgency_score: Optional[int] = None
    urgency_source: Optional[str] = None
    score: float
    factors: Dict[str, float] = Field(default_factory=dict)
    explanations: List[str] = Field(default_factory=list)
    recommended_action: str
    wait_estimate: Optional[WaitEstimate] = None


class BottleneckSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_waiting: int = 0
    overdue: int = 0
    unassigned: int = 0
    processing_failures: int = 0
    emergency: int = 0
    unassessed: int = 0
    oldest_waiting_minutes: Optional[int] = None
    pending_communications: int = 0
    pending_followups: int = 0
    open_escalations: int = 0


class QueueIntelligence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=utcnow)
    items: List[QueueRecommendation] = Field(default_factory=list)
    bottlenecks: BottleneckSummary = Field(default_factory=BottleneckSummary)
    staffing: StaffingState = Field(default_factory=StaffingState)
    disclaimer: str = (
        "Recommendations support human prioritization. They never change workflow status, "
        "dismiss warnings, or reassign a case automatically."
    )
