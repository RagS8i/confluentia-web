"""CareFlow AI FastAPI application."""
from __future__ import annotations

import logging
import mimetypes
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from OCR1.model import SUPPORTED_EXTENSIONS

from .communications import (
    SUPPORTED_MESSAGE_LANGUAGES,
    CommunicationError,
    CommunicationService,
)
from .config import Settings, load_settings
from .db import Database
from .deletion import DeletionService
from .escalation import EscalationError, EscalationService
from .followups import FollowUpError, FollowUpService
from .intake import (
    DocumentProcessor,
    IntakeService,
    SttService,
    persist_sources,
)
from .model.provider import BaseProvider, build_provider
from .patient_voice import PatientVoiceError, PatientVoiceService
from .pipeline.normalize import (
    InputError,
    adapt_stt,
    adapt_typed,
    sources_from_ocr_pipeline,
)
from .pipeline.orchestrator import (
    CaseUnavailable,
    ConsentRequired,
    PipelineService,
    ProcessingConflict,
)
from .queue_ops import QueueOperationsService
from .schemas import (
    CaseRecord,
    CaseStatus,
    ClinicalSummary,
    CommunicationIntent,
    CommunicationStatus,
    Fact,
    FollowUpStatus,
    Outcome,
    QueueItem,
    ReviewerAction,
    ReviewEvent,
    UrgencyAssessment,
    UrgencyEdit,
    UrgencySource,
    VerificationStatus,
)
from .security import (
    Principal,
    current_principal,
    ensure_case_access,
    ensure_roles,
)
from .storage import Cipher, DocumentStore, random_id, safe_error
from .triage.engine import RuleEngine

logger = logging.getLogger("careflow")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

STATIC_DIR = Path(__file__).resolve().parent / "static"
router = APIRouter(prefix="/api")


# --------------------------------------------------------------------- models
class CreateCaseRequest(BaseModel):
    patient_id: Optional[str] = None
    language_hint: Optional[str] = None


class ConsentRequest(BaseModel):
    scope: str = "careflow_processing_and_review"


class WithdrawRequest(BaseModel):
    reason: str = "consent withdrawn"
    delete_data: bool = True


class MessageRequest(BaseModel):
    text: str
    language: Optional[str] = None
    captured_at: Optional[str] = None


class FactReviewRequest(BaseModel):
    action: Literal["verify", "correct", "reject", "dispute"]
    value: Optional[Any] = None
    label: Optional[str] = None
    unit: Optional[str] = None
    note: Optional[str] = None


class NoteEditRequest(BaseModel):
    text: str = Field(min_length=1)


class OverrideRequest(BaseModel):
    final_status: Outcome
    reason: str = Field(min_length=3)
    confirmed: bool = False


class AssignRequest(BaseModel):
    reviewer: str = Field(min_length=1)
    confirmed: bool = False


class UrgencyEditRequest(BaseModel):
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=3, max_length=1000)
    rationale: Optional[str] = Field(default=None, max_length=1000)
    confirmed: bool = False


class CommunicationDraftRequest(BaseModel):
    intent: CommunicationIntent
    language: str = "en"


class CommunicationApproveRequest(BaseModel):
    text: Optional[str] = None


class CommunicationDismissRequest(BaseModel):
    reason: str = Field(min_length=3)


class FollowUpApproveRequest(BaseModel):
    text: Optional[str] = None
    language: Optional[str] = None


class FollowUpAnswerRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Optional[str] = None
    reprocess: bool = True


class FollowUpDismissRequest(BaseModel):
    reason: str = Field(min_length=3)


class PatientVoiceRequest(BaseModel):
    language: Optional[str] = None
    include_audio: bool = True


class StaffingUpdateRequest(BaseModel):
    clinicians_available: int = Field(ge=0, le=500)
    intake_available: int = Field(ge=0, le=500)
    average_review_minutes: Optional[float] = Field(default=None, gt=0, le=10080)
    sample_size: int = Field(default=0, ge=0, le=1000000)
    note: Optional[str] = Field(default=None, max_length=500)


class EscalationAcknowledgeRequest(BaseModel):
    note: Optional[str] = Field(default=None, max_length=1000)


# -------------------------------------------------------------------- helpers
def _db(request: Request) -> Database:
    return request.app.state.db


def _cipher(request: Request) -> Cipher:
    return request.app.state.cipher


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _pipeline(request: Request) -> PipelineService:
    return request.app.state.pipeline


def _deletion(request: Request) -> DeletionService:
    return request.app.state.deletion


def _documents(request: Request) -> DocumentStore:
    return request.app.state.documents


def _engine(request: Request) -> RuleEngine:
    return request.app.state.engine


def _communications(request: Request) -> CommunicationService:
    return request.app.state.communications


def _followups(request: Request) -> FollowUpService:
    return request.app.state.followups


def _patient_voice(request: Request) -> PatientVoiceService:
    return request.app.state.patient_voice


def _escalations(request: Request) -> EscalationService:
    return request.app.state.escalations


def _queue_ops(request: Request) -> QueueOperationsService:
    return request.app.state.queue_ops


def _intake(request: Request) -> IntakeService:
    return request.app.state.intake


def _raise_service_error(exc: Exception):
    if isinstance(exc, (CommunicationError, FollowUpError, EscalationError, PatientVoiceError)):
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        )
    raise exc


def _load_case(
    request: Request,
    case_id: str,
    principal: Principal,
    include_deleted: bool = False,
) -> CaseRecord:
    case = _db(request).get_case(case_id, include_deleted=include_deleted)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    ensure_case_access(_db(request), principal, case)
    return case


def _require_consent(case: CaseRecord) -> None:
    if case.deleted_at is not None:
        raise HTTPException(status_code=410, detail="case has been deleted")
    if case.consent_withdrawn_at is not None:
        raise HTTPException(status_code=409, detail="consent has been withdrawn")
    if case.consent_granted_at is None:
        raise HTTPException(status_code=409, detail="explicit patient consent is required before processing")


def _source_public(source, include_text: bool = False) -> Dict[str, Any]:
    payload = {
        "source_id": source.source_id,
        "parent_source_id": source.parent_source_id,
        "modality": source.modality.value,
        "segment_id": source.segment_id,
        "language": source.language,
        "language_status": source.language_status.value,
        "confidence": source.confidence,
        "document_ref": source.document_ref,
        "page": source.page,
        "bbox": source.bbox,
        "audio_start": source.audio_start,
        "audio_end": source.audio_end,
        "captured_at": source.captured_at.isoformat() if source.captured_at else None,
        "received_at": source.received_at.isoformat(),
        "usable": source.usable,
        "warnings": source.warnings,
    }
    if include_text:
        payload["original_text"] = source.original_text
        payload["normalized_text"] = source.normalized_text
    return payload


def _ingest(request: Request, case: CaseRecord, sources, notes, principal: Principal, kind: str):
    _require_consent(case)
    res = persist_sources(
        db=_db(request),
        cipher=_cipher(request),
        case=case,
        sources=sources,
        notes=notes,
        user_id=principal.user_id,
        role=principal.role,
        kind=kind,
    )
    return {
        "case_id": case.case_id,
        "inserted": res["inserted"],
        "duplicates": res["duplicates"],
        "notes": notes,
        "sources": [_source_public(s) for s in sources],
    }


# --------------------------------------------------------------------- routes
@router.get("/me")
def me(principal: Principal = Depends(current_principal)):
    return {"user_id": principal.user_id, "role": principal.role, "patient_id": principal.patient_id}


@router.get("/health")
def health(request: Request):
    db_ok = True
    try:
        _db(request).list_cases()
    except Exception:
        db_ok = False
    provider = request.app.state.provider.health()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "provider": provider,
        "rules": {
            "config_id": _engine(request).config.get("config_id"),
            "version": _engine(request).config.get("version"),
            "status": _engine(request).config.get("status"),
        },
    }


@router.get("/rules")
def rules(request: Request, principal: Principal = Depends(current_principal)):
    ensure_roles(principal, {"clinician", "admin"})
    return _engine(request).config


@router.post("/cases", status_code=201)
def create_case(
    payload: CreateCaseRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    db = _db(request)
    if principal.role == "patient":
        patient_id = principal.patient_id
    else:
        patient_id = payload.patient_id or ""
    if not patient_id:
        raise HTTPException(status_code=422, detail="patient_id is required")
    case = CaseRecord(
        case_id=random_id("case"),
        patient_id=patient_id,
        language_hint=payload.language_hint,
    )
    db.save_case(case)
    db.audit(principal.user_id, principal.role, "case_created", case.case_id, {})
    return case.model_dump(mode="json")


@router.get("/cases")
def list_cases(
    request: Request,
    principal: Principal = Depends(current_principal),
    include_deleted: bool = Query(default=False),
):
    if include_deleted:
        ensure_roles(principal, {"admin"})
    db, cipher = _db(request), _cipher(request)
    cases = db.list_cases(include_deleted=include_deleted)
    if principal.role == "patient":
        cases = [c for c in cases if c.patient_id == principal.patient_id]
    engine = _engine(request)
    rank = engine.rank
    items: List[QueueItem] = []
    now = datetime.now(timezone.utc)
    for case in cases:
        sources = db.get_sources(case.case_id, cipher)
        warnings = db.get_warnings(case.case_id)
        discrepancies = db.get_discrepancies(case.case_id)
        summary = db.get_summary(case.case_id)
        note = db.get_note(case.case_id)
        reviews = db.get_reviews(case.case_id)
        awaiting_raw = case.meta.get("awaiting_review_since")
        awaiting = None
        awaiting_minutes = None
        if awaiting_raw:
            try:
                awaiting = datetime.fromisoformat(awaiting_raw)
                if awaiting.tzinfo is None:
                    awaiting = awaiting.replace(tzinfo=timezone.utc)
                awaiting_minutes = int((now - awaiting).total_seconds() // 60)
            except ValueError:
                awaiting = None
        deadline = (case.meta.get("review_deadlines_minutes") or {}).get(case.workflow_status.value)
        overdue = bool(
            deadline is not None and awaiting_minutes is not None and awaiting_minutes > int(deadline)
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
        failures = len(case.meta.get("processing_failures") or []) + len(
            case.meta.get("stage_failures") or []
        )
        items.append(
            QueueItem(
                case_id=case.case_id,
                patient_id=case.patient_id,
                status=case.status,
                workflow_status=case.workflow_status,
                received_at=case.received_at,
                awaiting_review_since=awaiting,
                awaiting_minutes=awaiting_minutes,
                review_deadline_minutes=int(deadline) if deadline is not None else None,
                overdue=overdue,
                source_modalities=sorted({s.modality.value for s in sources}),
                warning_labels=[w.label for w in warnings if w.kind.value == "rule"][:6],
                warning_count=len(warnings),
                missing_info_count=len([g for g in (summary.gaps if summary else []) if g.code != "QUESTIONS_SUPPRESSED_EMERGENCY"]),
                contradiction_count=len(discrepancies),
                processing_failures=failures,
                assigned_reviewer=case.assigned_reviewer,
                review_status=review_status,
                consent=consent,
                urgency_score=case.urgency.score if case.urgency else None,
                urgency_source=case.urgency.source.value if case.urgency else None,
            )
        )
    items.sort(
        key=lambda item: (
            rank.get(item.workflow_status.value, 9),
            -(item.urgency_score if item.urgency_score is not None else -1),
            item.received_at,
        )
    )
    return {"items": [item.model_dump(mode="json") for item in items], "count": len(items)}


@router.get("/cases/{case_id}")
def case_detail(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    db, cipher = _db(request), _cipher(request)
    case = _load_case(request, case_id, principal, include_deleted=principal.role == "admin")
    if principal.role == "patient":
        # Patients receive only their own communication and follow-up data.
        # Clinician notes, warnings, discrepancies, raw sources, extracted facts,
        # escalations and internal processing metadata stay staff-only.
        communications = [
            item
            for item in _communications(request).list_for_case(case_id)
            if item.status == CommunicationStatus.sent
        ]
        followups = [
            item
            for item in _followups(request).list_for_case(case_id)
            if item.status in {FollowUpStatus.sent, FollowUpStatus.answered}
        ]
        answers = [
            item
            for item in _followups(request).answers_for_case(case_id)
            if item.role == "patient"
        ]
        return {
            "case": case.model_dump(mode="json"),
            "sources": [],
            "facts": [],
            "warnings": [],
            "discrepancies": [],
            "summary": None,
            "note": None,
            "reviews": [],
            "processing_runs": [],
            "documents": [],
            "communications": [item.model_dump(mode="json") for item in communications],
            "followups": [item.model_dump(mode="json") for item in followups],
            "followup_answers": [item.model_dump(mode="json") for item in answers],
            "escalations": [],
            "queue_estimate": None,
            "rule_config": {},
        }
    sources = db.get_sources(case_id, cipher)
    facts = db.get_facts(case_id, cipher)
    warnings = db.get_warnings(case_id)
    discrepancies = db.get_discrepancies(case_id)
    summary = db.get_summary(case_id)
    note = db.get_note(case_id)
    reviews = db.get_reviews(case_id)
    runs = db.get_runs(case_id)
    documents = db.get_documents(case_id)
    communications = _communications(request).list_for_case(case_id)
    followups = _followups(request).list_for_case(case_id)
    answers = _followups(request).answers_for_case(case_id)
    escalations = (
        _escalations(request).list(case_id=case_id)
        if principal.is_clinician
        else []
    )
    if principal.role == "patient":
        communications = [
            item for item in communications if item.status == CommunicationStatus.sent
        ]
        followups = [
            item
            for item in followups
            if item.status in {FollowUpStatus.sent, FollowUpStatus.answered}
        ]
        answers = [item for item in answers if item.role == "patient"]
    queue_estimate = None
    if principal.is_clinician:
        try:
            queue_estimate = _queue_ops(request).estimate_for_case(case_id).model_dump(
                mode="json"
            )
        except KeyError:
            queue_estimate = None
    if _settings(request).source_access_audit:
        db.audit(
            principal.user_id,
            principal.role,
            "source_access",
            case_id,
            {"source_ids": [s.source_id for s in sources], "count": len(sources)},
        )
    return {
        "case": case.model_dump(mode="json"),
        "sources": [_source_public(s, include_text=True) for s in sources],
        "facts": [f.model_dump(mode="json") for f in facts],
        "warnings": [w.model_dump(mode="json") for w in warnings],
        "discrepancies": [d.model_dump(mode="json") for d in discrepancies],
        "summary": summary.model_dump(mode="json") if summary else None,
        "note": note.model_dump(mode="json") if note else None,
        "reviews": reviews,
        "processing_runs": runs,
        "documents": documents,
        "communications": [item.model_dump(mode="json") for item in communications],
        "followups": [item.model_dump(mode="json") for item in followups],
        "followup_answers": [item.model_dump(mode="json") for item in answers],
        "escalations": [item.model_dump(mode="json") for item in escalations],
        "queue_estimate": queue_estimate,
        "rule_config": {
            "config_id": _engine(request).config.get("config_id"),
            "version": _engine(request).config.get("version"),
            "status": _engine(request).config.get("status"),
            "description": _engine(request).config.get("description"),
            "review_deadlines_minutes": _engine(request).config.get("review_deadlines_minutes"),
        },
    }


@router.get("/cases/{case_id}/patient-card")
def get_patient_card(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    try:
        card = _patient_voice(request).context_card(
            case_id, language="en-IN", include_audio=False
        )
    except PatientVoiceError as exc:
        _raise_service_error(exc)
    card["language_hint"] = case.language_hint
    return card


@router.post("/cases/{case_id}/patient-voice/context-card")
def patient_voice_context_card(
    case_id: str,
    payload: PatientVoiceRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        return _patient_voice(request).context_card(
            case_id,
            language=payload.language,
            include_audio=payload.include_audio,
        )
    except PatientVoiceError as exc:
        _raise_service_error(exc)


@router.post("/cases/{case_id}/patient-voice/detect-language")
async def patient_voice_detect_language(
    case_id: str,
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    _load_case(request, case_id, principal)
    content, filename, _mime = await _read_audio_upload(request, file)
    try:
        return await run_in_threadpool(
            _patient_voice(request).detect_language, content, filename
        )
    except PatientVoiceError as exc:
        _raise_service_error(exc)


@router.post("/cases/{case_id}/patient-voice/followups/{followup_id}")
def patient_voice_followup(
    case_id: str,
    followup_id: str,
    payload: PatientVoiceRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        return _patient_voice(request).localize_followup(
            case_id,
            followup_id,
            language=payload.language,
            include_audio=payload.include_audio,
            patient_view=principal.role == "patient",
        )
    except (PatientVoiceError, FollowUpError) as exc:
        _raise_service_error(exc)


@router.post("/cases/{case_id}/followups/{followup_id}/answers/audio", status_code=201)
async def answer_followup_audio(
    case_id: str,
    followup_id: str,
    request: Request,
    file: UploadFile = File(...),
    reprocess: bool = Query(default=True),
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    _load_case(request, case_id, principal)
    content, filename, mime = await _read_audio_upload(request, file)
    try:
        result = await run_in_threadpool(
            _patient_voice(request).answer_followup_audio,
            case_id,
            followup_id,
            content,
            filename,
            mime,
            principal.user_id,
            principal.role,
            reprocess=reprocess,
        )
    except (PatientVoiceError, FollowUpError) as exc:
        _raise_service_error(exc)
    return {
        "answer": result["answer"].model_dump(mode="json"),
        "followup": result["followup"].model_dump(mode="json"),
        "processing": result["processing"],
        "document_id": result.get("document_id"),
        "transcription": result.get("transcription"),
        "translated_text": result.get("translated_text"),
        "detected_language": result.get("detected_language"),
    }


@router.get("/cases/{case_id}/status")
def case_status(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    case = _load_case(request, case_id, principal)
    return {
        "case_id": case.case_id,
        "status": case.status.value,
        "workflow_status": case.workflow_status.value,
        "meta": case.meta,
        "runs": _db(request).get_runs(case_id),
    }


@router.get("/cases/{case_id}/sources/{source_id}")
def source_detail(
    case_id: str,
    source_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    _load_case(request, case_id, principal)
    source = _db(request).get_source(source_id, _cipher(request))
    if source is None or source.case_id != case_id:
        raise HTTPException(status_code=404, detail="source not found")
    _db(request).audit(
        principal.user_id,
        principal.role,
        "source_access",
        case_id,
        {"source_id": source_id},
    )
    return _source_public(source, include_text=True)


@router.post("/cases/{case_id}/consent")
def grant_consent(
    case_id: str,
    payload: ConsentRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    if case.deleted_at is not None:
        raise HTTPException(status_code=410, detail="case has been deleted")
    now = datetime.now(timezone.utc)
    case.consent_granted_at = now
    case.consent_scope = payload.scope
    case.consent_withdrawn_at = None
    if case.status == CaseStatus.received:
        case.status = CaseStatus.consent_pending
    _db(request).save_case(case)
    _db(request).audit(
        principal.user_id,
        principal.role,
        "consent_granted",
        case_id,
        {"scope": payload.scope},
    )
    return {"case_id": case_id, "consent_granted_at": now.isoformat(), "scope": payload.scope}


@router.post("/cases/{case_id}/consent/withdraw")
def withdraw_consent(
    case_id: str,
    payload: WithdrawRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "admin"})
    case = _load_case(request, case_id, principal)
    db = _db(request)
    now = datetime.now(timezone.utc)
    case.consent_withdrawn_at = now
    case.status = CaseStatus.withdrawn
    case.workflow_status = Outcome.unassessed
    db.save_case(case)
    db.audit(
        principal.user_id,
        principal.role,
        "consent_withdrawn",
        case_id,
        {"reason": payload.reason, "delete_data": payload.delete_data},
    )
    if not payload.delete_data:
        return {"case_id": case_id, "consent_withdrawn_at": now.isoformat(), "deletion": "not_requested"}
    existing = [
        job
        for job in db.list_deletion_jobs()
        if job["case_id"] == case_id and job["status"] in {"completed", "partial", "running"}
    ]
    job = existing[-1] if existing else _deletion(request).start(case_id, principal.user_id, payload.reason)
    return {
        "case_id": case_id,
        "consent_withdrawn_at": now.isoformat(),
        "deletion": job,
    }


@router.delete("/cases/{case_id}")
def delete_case(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
    reason: str = Query(default="case deletion requested"),
):
    ensure_roles(principal, {"patient", "admin"})
    case = _load_case(request, case_id, principal, include_deleted=principal.role == "admin")
    job = _deletion(request).start(case_id, principal.user_id, reason)
    return JSONResponse(status_code=202, content={"case_id": case_id, "deletion": job})


@router.get("/deletion-jobs", dependencies=[])
def list_deletion_jobs(
    request: Request,
    principal: Principal = Depends(current_principal),
    status: Optional[str] = Query(default=None),
):
    ensure_roles(principal, {"admin"})
    return {"items": _db(request).list_deletion_jobs(status=status)}


@router.post("/deletion-jobs/{job_id}/retry")
def retry_deletion(
    job_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"admin"})
    try:
        job = _deletion(request).run(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="deletion job not found")
    return job


@router.post("/cases/{case_id}/sources/stt")
def ingest_stt(
    case_id: str,
    payload: Dict[str, Any],
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    try:
        sources, notes = adapt_stt(payload, case_id)
    except InputError as exc:
        _db(request).audit(principal.user_id, principal.role, "stt_ingest_rejected", case_id, {"code": exc.code})
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message})
    return _ingest(request, case, sources, notes, principal, "stt")


@router.post("/cases/{case_id}/sources/ocr")
def ingest_ocr(
    case_id: str,
    payload: Dict[str, Any],
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    try:
        sources, notes = sources_from_ocr_pipeline(payload, case_id)
    except InputError as exc:
        _db(request).audit(principal.user_id, principal.role, "ocr_ingest_rejected", case_id, {"code": exc.code})
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message})
    return _ingest(request, case, sources, notes, principal, "ocr")


@router.post("/cases/{case_id}/sources/message")
def ingest_message(
    case_id: str,
    payload: MessageRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    try:
        sources, notes = adapt_typed(payload.model_dump(exclude_none=True), case_id)
    except InputError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message})
    return _ingest(request, case, sources, notes, principal, "message")


@router.post("/cases/{case_id}/uploads", status_code=201)
async def upload_document(
    case_id: str,
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    _require_consent(case)
    settings = _settings(request)
    mime = (file.content_type or "application/octet-stream").split(";")[0].strip().lower()
    if mime not in settings.allowed_upload_mime:
        raise HTTPException(status_code=415, detail="unsupported upload content type")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="upload exceeds configured size limit")
    if not content:
        raise HTTPException(status_code=422, detail="empty upload")
    import hashlib

    digest = hashlib.sha256(content).hexdigest()
    existing = _db(request).find_document_by_hash(case_id, digest)
    if existing:
        return {"document_id": existing["document_id"], "duplicate": True}
    stored = _documents(request).save(case_id, file.filename or "upload.bin", content)
    document_id = random_id("doc")
    _db(request).add_document(
        document_id,
        case_id,
        stored["stored_name"],
        stored["original_name"],
        mime,
        stored["size"],
        stored["sha256"],
    )
    _db(request).audit(
        principal.user_id,
        principal.role,
        "document_uploaded",
        case_id,
        {"document_id": document_id, "mime": mime, "size": stored["size"]},
    )
    return {
        "document_id": document_id,
        "duplicate": False,
        "original_name": stored["original_name"],
        "mime": mime,
        "size": stored["size"],
        "sha256": stored["sha256"],
    }


@router.get("/documents/{document_id}")
def download_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
    inline: bool = Query(default=False),
):
    db = _db(request)
    document = db.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    case = _load_case(request, document["case_id"], principal)
    content = _documents(request).read(document["stored_name"])
    db.audit(
        principal.user_id,
        principal.role,
        "source_access",
        case.case_id,
        {"document_id": document_id},
    )
    disposition = (
        "inline"
        if inline
        or document["mime"].startswith("audio/")
        or document["mime"].startswith("image/")
        or document["mime"] == "application/pdf"
        else "attachment"
    )
    return Response(
        content=content,
        media_type=document["mime"],
        headers={"Content-Disposition": f'{disposition}; filename="{document["original_name"]}"'},
    )


# ----------------------------------------------------------- direct intake API
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}
ALLOWED_AUDIO_MIMES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/ogg", "audio/vorbis",
    "audio/flac", "audio/x-flac", "audio/webm", "audio/aac", "application/octet-stream"
}

ALLOWED_DOC_MIMES = {
    "application/pdf", "image/png", "image/jpeg", "image/webp", "image/bmp",
    "image/tiff"
}


async def _read_audio_upload(request: Request, file: UploadFile) -> tuple[bytes, str, str]:
    """Validate an audio upload and return ``(bytes, filename, mime)``."""
    settings = _settings(request)
    filename = file.filename or "audio.wav"
    ext = Path(filename).suffix.lower()
    raw_mime = (file.content_type or "audio/wav").split(";")[0].strip().lower()
    if raw_mime not in ALLOWED_AUDIO_MIMES and not raw_mime.startswith("audio/"):
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=415, detail="unsupported audio content type")
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS and not raw_mime.startswith("audio/"):
        raise HTTPException(status_code=415, detail="unsupported audio file extension")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="upload exceeds configured size limit")
    if not content:
        raise HTTPException(status_code=422, detail="empty upload")
    return content, filename, raw_mime if raw_mime.startswith("audio/") else "audio/wav"


@router.post("/cases/{case_id}/intake/audio", status_code=202)
async def intake_audio(
    case_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_process: bool = Query(default=True),
    target_language: str = Query(default="en-IN"),
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    _require_consent(case)

    content, filename, raw_mime = await _read_audio_upload(request, file)

    # Save encrypted original through DocumentStore
    stored = _documents(request).save(case_id, filename, content)
    document_id = random_id("doc")
    _db(request).add_document(
        document_id,
        case_id,
        stored["stored_name"],
        stored["original_name"],
        raw_mime if raw_mime.startswith("audio/") else "audio/wav",
        stored["size"],
        stored["sha256"],
    )
    _db(request).audit(
        principal.user_id,
        principal.role,
        "audio_uploaded",
        case_id,
        {"document_id": document_id, "mime": raw_mime, "size": stored["size"]},
    )

    # Create intake job
    job = _intake(request).create_job(
        case_id=case_id,
        document_id=document_id,
        kind="audio",
        filename=stored["original_name"],
        auto_process=auto_process,
        target_language=target_language,
    )

    # Launch background processing task
    background_tasks.add_task(
        _intake(request).process_audio_job,
        job_id=job.job_id,
        audio_bytes=content,
        user_id=principal.user_id,
        role=principal.role,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "case_id": case_id,
            "document_id": document_id,
            "kind": job.kind,
            "stage": job.stage,
            "status": job.status,
            "poll": f"/api/cases/{case_id}/intake/jobs/{job.job_id}",
        },
    )


@router.post("/cases/{case_id}/intake/document", status_code=202)
async def intake_document(
    case_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_process: bool = Query(default=True),
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    _require_consent(case)

    settings = _settings(request)
    filename = file.filename or "document.pdf"
    ext = Path(filename).suffix.lower()
    raw_mime = (file.content_type or "application/octet-stream").split(";")[0].strip().lower()

    if raw_mime not in ALLOWED_DOC_MIMES and ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported document content type")
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported document file extension")

    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="upload exceeds configured size limit")
    if not content:
        raise HTTPException(status_code=422, detail="empty upload")

    # Determine stored mime
    inferred_mime = mimetypes.guess_type(filename)[0] or raw_mime
    if inferred_mime not in ALLOWED_DOC_MIMES:
        inferred_mime = "application/pdf" if ext == ".pdf" else "image/png"

    # Save encrypted original through DocumentStore
    stored = _documents(request).save(case_id, filename, content)
    document_id = random_id("doc")
    _db(request).add_document(
        document_id,
        case_id,
        stored["stored_name"],
        stored["original_name"],
        inferred_mime,
        stored["size"],
        stored["sha256"],
    )
    _db(request).audit(
        principal.user_id,
        principal.role,
        "document_uploaded",
        case_id,
        {"document_id": document_id, "mime": inferred_mime, "size": stored["size"]},
    )

    # Create intake job
    job = _intake(request).create_job(
        case_id=case_id,
        document_id=document_id,
        kind="document",
        filename=stored["original_name"],
        auto_process=auto_process,
    )

    # Launch background processing task
    background_tasks.add_task(
        _intake(request).process_document_job,
        job_id=job.job_id,
        document_bytes=content,
        user_id=principal.user_id,
        role=principal.role,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "case_id": case_id,
            "document_id": document_id,
            "kind": job.kind,
            "stage": job.stage,
            "status": job.status,
            "poll": f"/api/cases/{case_id}/intake/jobs/{job.job_id}",
        },
    )


@router.get("/cases/{case_id}/intake/jobs/{job_id}")
def get_intake_job(
    case_id: str,
    job_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    _load_case(request, case_id, principal)
    job = _intake(request).get_job(job_id)
    if job is None or job.case_id != case_id:
        raise HTTPException(status_code=404, detail="intake job not found")
    return job.to_dict()



@router.post("/cases/{case_id}/process")
def process_case(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
    sync: bool = Query(default=False),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    case = _load_case(request, case_id, principal)
    _require_consent(case)
    db = _db(request)
    db.audit(principal.user_id, principal.role, "processing_requested", case_id, {"sync": sync})
    if _settings(request).processing_mode == "sync" or sync:
        try:
            result = _pipeline(request).process_case(case_id, actor=principal.user_id)
        except ProcessingConflict:
            raise HTTPException(status_code=409, detail="case is already being processed")
        except ConsentRequired as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except CaseUnavailable as exc:
            raise HTTPException(status_code=410, detail=str(exc))
        return JSONResponse(status_code=200, content=result.to_dict())
    current = db.get_case(case_id, include_deleted=True)
    if current is not None and current.status != CaseStatus.processing:
        current.status = CaseStatus.queued
        db.save_case(current)
    _pipeline(request).submit(case_id, principal.user_id)
    return JSONResponse(
        status_code=202,
        content={"case_id": case_id, "status": "queued", "poll": f"/api/cases/{case_id}/status"},
    )


@router.patch("/cases/{case_id}/facts/{fact_id}")
def review_fact(
    case_id: str,
    fact_id: str,
    payload: FactReviewRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    case = _load_case(request, case_id, principal)
    db, cipher = _db(request), _cipher(request)
    fact = db.get_fact(case_id, fact_id, cipher)
    if fact is None:
        raise HTTPException(status_code=404, detail="fact not found")
    mapping = {
        "verify": VerificationStatus.verified,
        "correct": VerificationStatus.corrected,
        "reject": VerificationStatus.rejected,
        "dispute": VerificationStatus.disputed,
    }
    fact.verification_status = mapping[payload.action]
    if payload.action == "correct":
        if payload.value is not None:
            fact.value = payload.value
        if payload.label:
            fact.label = payload.label
        if payload.unit is not None:
            fact.unit = payload.unit
        if payload.note:
            fact.review_note = payload.note
        if not any([payload.value is not None, payload.label, payload.unit is not None, payload.note]):
            raise HTTPException(status_code=422, detail="a correction requires a value, label, unit or note")
    elif payload.note:
        fact.review_note = payload.note
    fact.reviewed_by = principal.user_id
    fact.reviewed_at = datetime.now(timezone.utc)
    fact.updated_at = datetime.now(timezone.utc)
    db.update_fact_review(fact, cipher)
    action_map = {
        "verify": ReviewerAction.verify,
        "correct": ReviewerAction.correct,
        "reject": ReviewerAction.reject,
        "dispute": ReviewerAction.verify,
    }
    db.add_review(
        ReviewEvent(
            event_id=uuid.uuid4().hex,
            case_id=case_id,
            action=action_map[payload.action],
            actor=principal.user_id,
            role=principal.role,
            reason=payload.note,
            detail={"fact_id": fact_id, "action": payload.action},
        )
    )
    db.audit(
        principal.user_id,
        principal.role,
        f"fact_{payload.action}",
        case_id,
        {"fact_id": fact_id},
    )
    meta = dict(case.meta)
    meta["summary_stale"] = True
    case.meta = meta
    db.save_case(case)
    return fact.model_dump(mode="json")


@router.post("/cases/{case_id}/warnings/{warning_id}/acknowledge")
def acknowledge_warning(
    case_id: str,
    warning_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    db = _db(request)
    if not any(w.warning_id == warning_id for w in db.get_warnings(case_id)):
        raise HTTPException(status_code=404, detail="warning not found")
    db.acknowledge_warning(warning_id)
    db.add_review(
        ReviewEvent(
            event_id=uuid.uuid4().hex,
            case_id=case_id,
            action=ReviewerAction.acknowledge_warning,
            actor=principal.user_id,
            role=principal.role,
            detail={"warning_id": warning_id},
        )
    )
    db.audit(principal.user_id, principal.role, "warning_acknowledged", case_id, {"warning_id": warning_id})
    return {"warning_id": warning_id, "status": "acknowledged"}


@router.patch("/cases/{case_id}/note")
def edit_note(
    case_id: str,
    payload: NoteEditRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    db = _db(request)
    note = db.get_note(case_id)
    if note is None:
        summary = db.get_summary(case_id)
        note = summary.draft_note if summary else None
    if note is None:
        raise HTTPException(status_code=404, detail="no draft note exists; run processing first")
    previous = note.model_dump(mode="json")
    note.text = payload.text
    note.status = note.status.__class__.clinician_edited
    note.edited_by = principal.user_id
    note.updated_at = datetime.now(timezone.utc)
    note.version += 1
    note.approved_by = None
    note.approved_at = None
    note.history.append(
        {
            "version": note.version - 1,
            "status": previous.get("status"),
            "approved_by": previous.get("approved_by"),
            "edited_by": previous.get("edited_by"),
            "text": previous.get("text"),
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    db.save_note(case_id, note)
    db.add_review(
        ReviewEvent(
            event_id=uuid.uuid4().hex,
            case_id=case_id,
            action=ReviewerAction.edit_note,
            actor=principal.user_id,
            role=principal.role,
            detail={"version": note.version},
        )
    )
    db.audit(principal.user_id, principal.role, "note_edited", case_id, {"version": note.version})
    return note.model_dump(mode="json")


@router.post("/cases/{case_id}/note/approve")
def approve_note(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    case = _load_case(request, case_id, principal)
    db = _db(request)
    note = db.get_note(case_id)
    if note is None:
        summary = db.get_summary(case_id)
        note = summary.draft_note if summary else None
    if note is None:
        raise HTTPException(status_code=404, detail="no draft note to approve")
    if note.status.value == "approved":
        raise HTTPException(status_code=409, detail="note is already approved")
    if not note.text.strip():
        raise HTTPException(status_code=422, detail="note text is empty")
    now = datetime.now(timezone.utc)
    original_status = case.workflow_status
    note.status = note.status.__class__.approved
    note.approved_by = principal.user_id
    note.approved_at = now
    note.updated_at = now
    db.save_note(case_id, note)
    case.status = CaseStatus.reviewed
    db.save_case(case)
    db.add_review(
        ReviewEvent(
            event_id=uuid.uuid4().hex,
            case_id=case_id,
            action=ReviewerAction.approve_note,
            actor=principal.user_id,
            role=principal.role,
            original_status=original_status,
            final_status=case.workflow_status,
            detail={"note_version": note.version},
        )
    )
    db.audit(
        principal.user_id,
        principal.role,
        "note_approved",
        case_id,
        {"note_version": note.version},
    )
    return note.model_dump(mode="json")


@router.post("/cases/{case_id}/override")
def override_status(
    case_id: str,
    payload: OverrideRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    if not payload.confirmed:
        raise HTTPException(
            status_code=422,
            detail="explicit confirmation is required before overriding workflow status",
        )
    case = _load_case(request, case_id, principal)
    db = _db(request)
    original = case.workflow_status
    now = datetime.now(timezone.utc)
    override = {
        "original_status": original.value,
        "final_status": payload.final_status.value,
        "reviewer": principal.user_id,
        "role": principal.role,
        "timestamp": now.isoformat(),
        "reason": payload.reason,
    }
    meta = dict(case.meta)
    overrides = list(meta.get("status_overrides") or [])
    overrides.append(override)
    meta["status_overrides"] = overrides
    case.meta = meta
    case.workflow_status = payload.final_status
    db.save_case(case)
    db.add_review(
        ReviewEvent(
            event_id=uuid.uuid4().hex,
            case_id=case_id,
            action=ReviewerAction.override_status,
            actor=principal.user_id,
            role=principal.role,
            original_status=original,
            final_status=payload.final_status,
            reason=payload.reason,
            detail=override,
        )
    )
    db.audit(
        principal.user_id,
        principal.role,
        "status_overridden",
        case_id,
        {
            "original_status": original.value,
            "final_status": payload.final_status.value,
            "reason_recorded": True,
        },
    )
    return {"case_id": case_id, "override": override, "case": case.model_dump(mode="json")}


@router.post("/cases/{case_id}/assign")
def assign_reviewer(
    case_id: str,
    payload: AssignRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    if not payload.confirmed:
        raise HTTPException(
            status_code=422,
            detail="explicit confirmation is required before assigning or reassigning a case",
        )
    case = _load_case(request, case_id, principal)
    previous = case.assigned_reviewer
    case.assigned_reviewer = payload.reviewer
    _db(request).save_case(case)
    _db(request).add_review(
        ReviewEvent(
            event_id=uuid.uuid4().hex,
            case_id=case_id,
            action=ReviewerAction.assign,
            actor=principal.user_id,
            role=principal.role,
            detail={"reviewer": payload.reviewer, "previous_reviewer": previous},
        )
    )
    _db(request).audit(
        principal.user_id,
        principal.role,
        "case_assigned",
        case_id,
        {"reviewer": payload.reviewer, "previous_reviewer": previous},
    )
    return {"case_id": case_id, "assigned_reviewer": payload.reviewer}


@router.patch("/cases/{case_id}/urgency")
def edit_urgency(
    case_id: str,
    payload: UrgencyEditRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    """Clinician override of the advisory urgency score.

    The deterministic workflow status is never changed by this action. The
    previous score, editor, role, reason and timestamp are retained.
    """
    ensure_roles(principal, {"clinician", "admin"})
    if not payload.confirmed:
        raise HTTPException(
            status_code=422,
            detail="explicit confirmation is required before editing urgency",
        )
    case = _load_case(request, case_id, principal)
    db = _db(request)
    now = datetime.now(timezone.utc)
    previous = case.urgency
    history = list(previous.history) if previous else []
    history.append(
        UrgencyEdit(
            score=payload.score,
            previous_score=previous.score if previous else None,
            edited_by=principal.user_id,
            role=principal.role,
            reason=payload.reason,
            edited_at=now,
        )
    )
    original_slm_score = previous.slm_score if previous else None
    if original_slm_score is None and previous is not None and previous.source == UrgencySource.slm:
        original_slm_score = previous.score
    assessment = UrgencyAssessment(
        score=payload.score,
        rationale=payload.rationale
        if payload.rationale is not None
        else (previous.rationale if previous else ""),
        source=UrgencySource.clinician,
        model=previous.model if previous else None,
        assessed_at=previous.assessed_at if previous else now,
        assessed_by=previous.assessed_by if previous else None,
        slm_score=original_slm_score,
        edited_by=principal.user_id,
        edited_at=now,
        edit_reason=payload.reason,
        model_error=previous.model_error if previous else None,
        history=history,
    )
    case.urgency = assessment
    db.save_case(case)
    db.add_review(
        ReviewEvent(
            event_id=uuid.uuid4().hex,
            case_id=case_id,
            action=ReviewerAction.edit_urgency,
            actor=principal.user_id,
            role=principal.role,
            original_status=case.workflow_status,
            final_status=case.workflow_status,
            reason=payload.reason,
            detail={
                "previous_score": previous.score if previous else None,
                "new_score": payload.score,
                "source": UrgencySource.clinician.value,
                "workflow_status_unchanged": case.workflow_status.value,
            },
        )
    )
    db.audit(
        principal.user_id,
        principal.role,
        "urgency_edited",
        case_id,
        {
            "previous_score": previous.score if previous else None,
            "new_score": payload.score,
            "reason_recorded": True,
        },
    )
    return {"case_id": case_id, "urgency": assessment.model_dump(mode="json")}


# ------------------------------------------------------- patient communications
@router.get("/communications/templates")
def communication_templates(
    request: Request,
    principal: Principal = Depends(current_principal),
    language: Optional[str] = Query(default=None),
):
    ensure_roles(principal, {"clinician", "admin", "intake"})
    try:
        return _communications(request).templates(language)
    except CommunicationError as exc:
        _raise_service_error(exc)


@router.post("/cases/{case_id}/communications/draft", status_code=201)
def draft_communication(
    case_id: str,
    payload: CommunicationDraftRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    case = _load_case(request, case_id, principal)
    estimate = None
    if payload.intent == CommunicationIntent.wait_update:
        try:
            estimate = _queue_ops(request).estimate_for_case(case_id)
        except KeyError:
            estimate = None
    try:
        message = _communications(request).draft(
            case,
            payload.intent,
            payload.language,
            principal.user_id,
            wait_estimate=estimate,
        )
    except CommunicationError as exc:
        _raise_service_error(exc)
    return message.model_dump(mode="json")


@router.get("/cases/{case_id}/communications")
def list_communications(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    _load_case(request, case_id, principal)
    messages = _communications(request).list_for_case(case_id)
    if principal.role == "patient":
        messages = [item for item in messages if item.status == CommunicationStatus.sent]
    return {"items": [item.model_dump(mode="json") for item in messages]}


@router.post("/cases/{case_id}/communications/{message_id}/approve")
def approve_communication(
    case_id: str,
    message_id: str,
    payload: CommunicationApproveRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        message = _communications(request).approve(
            case_id, message_id, principal.user_id, principal.role, payload.text
        )
    except CommunicationError as exc:
        _raise_service_error(exc)
    return message.model_dump(mode="json")


@router.post("/cases/{case_id}/communications/{message_id}/send")
def send_communication(
    case_id: str,
    message_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        message = _communications(request).send(
            case_id, message_id, principal.user_id, principal.role
        )
    except CommunicationError as exc:
        _raise_service_error(exc)
    return message.model_dump(mode="json")


@router.post("/cases/{case_id}/communications/{message_id}/dismiss")
def dismiss_communication(
    case_id: str,
    message_id: str,
    payload: CommunicationDismissRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        message = _communications(request).dismiss(
            case_id, message_id, principal.user_id, principal.role, payload.reason
        )
    except CommunicationError as exc:
        _raise_service_error(exc)
    return message.model_dump(mode="json")


# ------------------------------------------------------------- follow-up care
@router.get("/cases/{case_id}/followups")
def list_followups(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    _load_case(request, case_id, principal)
    records = _followups(request).list_for_case(case_id)
    if principal.role == "patient":
        records = [
            item
            for item in records
            if item.status in {FollowUpStatus.sent, FollowUpStatus.answered}
        ]
    return {
        "items": [item.model_dump(mode="json") for item in records],
        "answers": [
            item.model_dump(mode="json")
            for item in _followups(request).answers_for_case(case_id)
            if principal.role != "patient" or item.role == "patient"
        ],
    }


@router.post("/cases/{case_id}/followups/sync")
def sync_followups(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    case = _load_case(request, case_id, principal)
    summary = _db(request).get_summary(case_id)
    records = _followups(request).sync_from_summary(
        case, summary, actor=principal.user_id
    )
    return {"items": [item.model_dump(mode="json") for item in records]}


@router.post("/cases/{case_id}/followups/{followup_id}/approve")
def approve_followup(
    case_id: str,
    followup_id: str,
    payload: FollowUpApproveRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        record = _followups(request).approve(
            case_id,
            followup_id,
            principal.user_id,
            principal.role,
            payload.text,
            payload.language,
        )
    except FollowUpError as exc:
        _raise_service_error(exc)
    return record.model_dump(mode="json")


@router.post("/cases/{case_id}/followups/{followup_id}/send")
def send_followup(
    case_id: str,
    followup_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        record = _followups(request).send(
            case_id, followup_id, principal.user_id, principal.role
        )
    except FollowUpError as exc:
        _raise_service_error(exc)
    return record.model_dump(mode="json")


@router.post("/cases/{case_id}/followups/{followup_id}/dismiss")
def dismiss_followup(
    case_id: str,
    followup_id: str,
    payload: FollowUpDismissRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        record = _followups(request).dismiss(
            case_id, followup_id, principal.user_id, principal.role, payload.reason
        )
    except FollowUpError as exc:
        _raise_service_error(exc)
    return record.model_dump(mode="json")


@router.post("/cases/{case_id}/followups/{followup_id}/answers", status_code=201)
def answer_followup(
    case_id: str,
    followup_id: str,
    payload: FollowUpAnswerRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"patient", "intake", "clinician", "admin"})
    _load_case(request, case_id, principal)
    try:
        result = _followups(request).answer(
            case_id,
            followup_id,
            payload.text,
            payload.language,
            principal.user_id,
            principal.role,
            reprocess=payload.reprocess,
        )
    except FollowUpError as exc:
        _raise_service_error(exc)
    return {
        "answer": result["answer"].model_dump(mode="json"),
        "followup": result["followup"].model_dump(mode="json"),
        "processing": result["processing"],
    }


# ----------------------------------------------------------- queue operations
@router.get("/operations/staffing")
def get_staffing(
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "intake", "admin"})
    return _queue_ops(request).staffing().model_dump(mode="json")


@router.put("/operations/staffing")
def update_staffing(
    payload: StaffingUpdateRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    state = _queue_ops(request).update_staffing(
        principal.user_id,
        principal.role,
        payload.clinicians_available,
        payload.intake_available,
        payload.average_review_minutes,
        payload.sample_size,
        payload.note,
    )
    return state.model_dump(mode="json")


@router.get("/queue/intelligence")
def queue_intelligence(
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "intake", "admin"})
    return _queue_ops(request).intelligence().model_dump(mode="json")


# --------------------------------------------------------------- escalations
@router.get("/escalations")
def list_escalations(
    request: Request,
    principal: Principal = Depends(current_principal),
    case_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
):
    ensure_roles(principal, {"clinician", "admin"})
    if case_id:
        _load_case(request, case_id, principal)
    return {
        "items": [
            item.model_dump(mode="json")
            for item in _escalations(request).list(case_id=case_id, status=status)
        ]
    }


@router.post("/escalations/{escalation_id}/acknowledge")
def acknowledge_escalation(
    escalation_id: str,
    payload: EscalationAcknowledgeRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    try:
        record = _escalations(request).acknowledge(
            escalation_id, principal.user_id, principal.role, payload.note
        )
    except EscalationError as exc:
        _raise_service_error(exc)
    _load_case(request, record.case_id, principal)
    return record.model_dump(mode="json")


@router.post("/escalations/{escalation_id}/retry")
def retry_escalation(
    escalation_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    ensure_roles(principal, {"clinician", "admin"})
    try:
        record = _escalations(request).retry(
            escalation_id, principal.user_id, principal.role
        )
    except EscalationError as exc:
        _raise_service_error(exc)
    _load_case(request, record.case_id, principal)
    return record.model_dump(mode="json")


@router.get("/cases/{case_id}/audit")
def case_audit(
    case_id: str,
    request: Request,
    principal: Principal = Depends(current_principal),
    limit: int = Query(default=200, ge=1, le=2000),
):
    ensure_roles(principal, {"clinician", "admin"})
    _load_case(request, case_id, principal, include_deleted=principal.role == "admin")
    return {"items": _db(request).get_audit(case_id, limit=limit)}


# ----------------------------------------------------------------------- app
def create_app(
    settings: Optional[Settings] = None,
    provider: Optional[BaseProvider] = None,
    engine: Optional[RuleEngine] = None,
    stt_service: Optional[SttService] = None,
    document_processor: Optional[DocumentProcessor] = None,
) -> FastAPI:
    settings = settings or load_settings()
    cipher = Cipher(settings)
    db = Database(settings.db_path, cipher)
    db.init_schema()
    documents = DocumentStore(settings, cipher)
    engine = engine or RuleEngine(settings.rules_path)
    provider = build_provider(settings, provider)
    queue_ops = QueueOperationsService(settings, db, engine)
    escalations = EscalationService(settings, db)
    followups = FollowUpService(settings, db, cipher)
    patient_voice = PatientVoiceService(settings, db, cipher, documents, followups)
    communications = CommunicationService(settings, db, cipher, queue_ops=queue_ops)
    pipeline = PipelineService(
        settings,
        db,
        cipher,
        provider,
        engine,
        escalation=escalations,
        followups=followups,
    )
    followups.pipeline = pipeline
    deletion = DeletionService(db, documents)
    intake = IntakeService(
        settings=settings,
        db=db,
        cipher=cipher,
        documents=documents,
        pipeline=pipeline,
        stt_service=stt_service,
        document_processor=document_processor,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        pipeline._executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="CareFlow AI",
        version="0.1.0",
        description=(
            "Human-in-the-loop clinical intake and triage-support workflow. "
            "Not a diagnostic device. Clinician review is mandatory."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.db = db
    app.state.cipher = cipher
    app.state.documents = documents
    app.state.engine = engine
    app.state.provider = provider
    app.state.pipeline = pipeline
    app.state.deletion = deletion
    app.state.queue_ops = queue_ops
    app.state.escalations = escalations
    app.state.followups = followups
    app.state.patient_voice = patient_voice
    app.state.communications = communications
    app.state.intake = intake

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.exception_handler(InputError)
    async def input_error_handler(_request: Request, exc: InputError):
        return JSONResponse(status_code=422, content={"detail": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception):
        logger.exception("unhandled application error")
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    return app


app = create_app()
