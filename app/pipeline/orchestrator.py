"""Pipeline orchestration with persisted stage state and partial-failure safety."""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..config import Settings
from ..db import Database
from ..model.provider import BaseProvider
from ..schemas import (
    CaseStatus,
    ClinicalSummary,
    Discrepancy,
    DraftNote,
    Fact,
    FollowUpQuestion,
    Gap,
    Outcome,
    ProcessingRun,
    SourceRecord,
    SourceRef,
    UrgencySource,
    Warning,
    WarningKind,
)
from ..storage import Cipher, safe_error, sha256_text
from ..triage.engine import RuleEngine, TriageResult
from .extract import ExtractionOutcome, SourceFailure, extract_facts
from .gaps import detect_gaps
from .interpret import interpret_sources
from .normalize import InputError
from .reconcile import reconcile
from .summary import generate_summary
from .timeline import build_timeline
from .urgency import assess_urgency


class ConsentRequired(RuntimeError):
    pass


class ProcessingConflict(RuntimeError):
    pass


class CaseUnavailable(RuntimeError):
    pass


@dataclass
class PipelineOutcome:
    case_id: str
    case_status: CaseStatus
    workflow_status: Outcome
    summary: Optional[ClinicalSummary]
    failures: List[Dict[str, Any]] = field(default_factory=list)
    stage_failures: List[Dict[str, Any]] = field(default_factory=list)
    triggered_rule_ids: List[str] = field(default_factory=list)
    unresolved_rule_ids: List[str] = field(default_factory=list)
    fact_count: int = 0
    warning_count: int = 0
    discrepancy_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_status": self.case_status.value,
            "workflow_status": self.workflow_status.value,
            "warning_count": self.warning_count,
            "fact_count": self.fact_count,
            "discrepancy_count": self.discrepancy_count,
            "triggered_rule_ids": self.triggered_rule_ids,
            "unresolved_rule_ids": self.unresolved_rule_ids,
            "processing_failures": self.failures,
            "stage_failures": self.stage_failures,
            "summary": self.summary.model_dump(mode="json") if self.summary else None,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PipelineService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        cipher: Cipher,
        provider: BaseProvider,
        engine: RuleEngine,
        escalation=None,
        followups=None,
    ):
        self.settings = settings
        self.db = db
        self.cipher = cipher
        self.provider = provider
        self.engine = engine
        self.escalation = escalation
        self.followups = followups
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, case_id: str) -> threading.Lock:
        with self._locks_guard:
            if case_id not in self._locks:
                self._locks[case_id] = threading.Lock()
            return self._locks[case_id]

    # ------------------------------------------------------------- scheduling
    def submit(self, case_id: str, actor: str = "system") -> None:
        self._executor.submit(self._background, case_id, actor)

    def _background(self, case_id: str, actor: str) -> None:
        try:
            self.process_case(case_id, actor=actor)
        except Exception as exc:  # pragma: no cover - defensive boundary
            try:
                case = self.db.get_case(case_id, include_deleted=True)
                if case and case.status != CaseStatus.deleted:
                    self.db.update_case(
                        case_id,
                        status=CaseStatus.processing_failed,
                        workflow_status=Outcome.unassessed,
                    )
                self.db.audit(
                    actor,
                    "system",
                    "pipeline_background_failed",
                    case_id,
                    {"error": safe_error(exc)},
                )
            except Exception:
                pass

    # --------------------------------------------------------------- pipeline
    def process_case(self, case_id: str, actor: str = "system") -> PipelineOutcome:
        case = self.db.get_case(case_id, include_deleted=True)
        if case is None:
            raise CaseUnavailable("case not found")
        if case.deleted_at is not None or case.status == CaseStatus.deleted:
            raise CaseUnavailable("case has been deleted")
        if case.consent_withdrawn_at is not None:
            raise ConsentRequired("consent has been withdrawn")
        if case.consent_granted_at is None:
            raise ConsentRequired("explicit patient consent is required before processing")

        lock = self._lock_for(case_id)
        if not lock.acquire(blocking=False):
            raise ProcessingConflict("case is already being processed")
        try:
            return self._process_locked(case, actor)
        finally:
            lock.release()

    def _process_locked(self, case, actor: str) -> PipelineOutcome:
        case_id = case.case_id
        self.db.update_case(case_id, status=CaseStatus.processing)
        self.db.audit(actor, "system", "pipeline_started", case_id, {})

        stage_failures: List[Dict[str, Any]] = []
        starter = _now().isoformat()

        # Stage A: load and validate normalized sources.
        sources: List[SourceRecord] = []
        try:
            def load_sources():
                return self.db.get_sources(case_id, self.cipher)

            sources = self._run_stage(case_id, "A_input_normalization", load_sources)
        except Exception as exc:
            stage_failures.append(self._stage_failure("A_input_normalization", exc))

        usable_sources = [s for s in sources if s.usable]

        # Stage B: multilingual interpretation readiness.
        interpretation = interpret_sources(sources)
        try:
            self._run_stage(
                case_id,
                "B_multilingual_interpretation",
                lambda: {"notes": interpretation.notes, "languages": interpretation.languages},
            )
        except Exception as exc:
            stage_failures.append(self._stage_failure("B_multilingual_interpretation", exc))

        # Stage C: structured extraction, per source with bounded retries.
        extraction = ExtractionOutcome(model=self.provider.model)
        try:
            def run_extraction():
                if not usable_sources:
                    return ExtractionOutcome(
                        model=self.provider.model,
                        failures=[
                            SourceFailure(
                                source_id="case",
                                code="no_usable_sources",
                                message="no usable source was available for extraction",
                                retryable=False,
                            )
                        ],
                    )
                return extract_facts(case_id, usable_sources, self.provider, self.settings)

            extraction = self._run_stage(case_id, "C_structured_extraction", run_extraction)
            self._ensure_active(case_id)
            for fact in extraction.facts:
                self.db.upsert_fact(fact, self.cipher)
        except CaseUnavailable:
            raise
        except Exception as exc:
            stage_failures.append(self._stage_failure("C_structured_extraction", exc))

        try:
            facts = self.db.get_facts(case_id, self.cipher)
        except Exception:
            facts = list(extraction.facts)

        # Stage D: timeline.
        timeline = []
        try:
            timeline = self._run_stage(
                case_id, "D_timeline", lambda: build_timeline(facts)
            )
        except Exception as exc:
            stage_failures.append(self._stage_failure("D_timeline", exc))

        # Stage E: cross-modal reconciliation.
        discrepancies: List[Discrepancy] = []
        try:
            discrepancies = self._run_stage(
                case_id,
                "E_cross_modal_reconciliation",
                lambda: reconcile(case_id, facts, sources),
            )
            self.db.save_discrepancies(case_id, discrepancies)
        except Exception as exc:
            stage_failures.append(self._stage_failure("E_cross_modal_reconciliation", exc))

        # Component warnings are preserved independently of rule warnings.
        component_warnings = self._component_warnings(
            case_id, sources, extraction, interpretation
        )
        degraded_source = any(
            any(
                warning in {"unreadable_critical_field", "empty_text", "low_source_confidence"}
                for warning in source.warnings
            )
            for source in sources
        )
        critical_failure = (
            bool(extraction.failures) or bool(stage_failures) or degraded_source
        )

        # Stage F rule engine (deterministic, independent of the SLM).
        triage = TriageResult(outcome=Outcome.unassessed)
        try:
            triage = self._run_stage(
                case_id,
                "F_rule_engine",
                lambda: self.engine.evaluate(
                    case_id,
                    facts,
                    sources,
                    component_warnings=component_warnings,
                    critical_failure=critical_failure,
                ),
            )
            self.db.save_warnings(case_id, triage.warnings)
        except Exception as exc:
            stage_failures.append(self._stage_failure("F_rule_engine", exc))
            triage = TriageResult(
                outcome=Outcome.unassessed,
                warnings=component_warnings,
                config_id=self.engine.config.get("config_id"),
                config_version=str(self.engine.config.get("version")),
                config_status=self.engine.config.get("status"),
                review_deadlines_minutes={
                    k: int(v)
                    for k, v in (self.engine.config.get("review_deadlines_minutes") or {}).items()
                },
            )
            fallback = Warning(
                warning_id="warn_" + sha256_text(case_id + "rule_engine_failed")[:20],
                case_id=case_id,
                kind=WarningKind.component,
                code="RULE_ENGINE_FAILURE",
                label="Rule engine could not complete",
                explanation="The case is unassessed; no routine or reassuring status can be assigned.",
                outcome=Outcome.unassessed,
                component="rule_engine",
            )
            triage.warnings.append(fallback)
            self.db.save_warnings(case_id, triage.warnings)

        # Immediate emergency escalation. This deliberately happens before
        # optional follow-up, summary or patient-message generation so those
        # steps can never delay an urgent human alert.
        if self.escalation is not None:
            try:
                escalation_case = self.db.get_case(case_id, include_deleted=True)
                if escalation_case is not None:
                    escalation_case.workflow_status = triage.outcome
                    self.escalation.raise_for_case(
                        escalation_case, triage.warnings, actor=actor
                    )
            except Exception as exc:
                stage_failures.append(
                    self._stage_failure("F1_urgent_escalation", exc)
                )
                self.db.audit(
                    actor,
                    "system",
                    "urgent_escalation_failed",
                    case_id,
                    {"error": safe_error(exc)},
                )

        # Stage G: gaps, follow-up questions and summary.
        gaps: List[Gap] = []
        questions: List[FollowUpQuestion] = []
        try:
            gaps, questions = self._run_stage(
                case_id,
                "G_missing_information",
                lambda: detect_gaps(
                    facts,
                    discrepancies,
                    sources,
                    extraction.failures,
                    triage.outcome,
                    extraction.questions,
                ),
            )
        except Exception as exc:
            stage_failures.append(self._stage_failure("G_missing_information", exc))

        rule_meta = {
            "config_id": triage.config_id,
            "config_version": triage.config_version,
            "config_status": triage.config_status,
        }

        summary: Optional[ClinicalSummary] = None
        try:
            summary = self._run_stage(
                case_id,
                "G_summary_generation",
                lambda: generate_summary(
                    case_id=case_id,
                    facts=facts,
                    sources=sources,
                    timeline=timeline,
                    discrepancies=discrepancies,
                    warnings=triage.warnings,
                    gaps=gaps,
                    questions=questions,
                    outcome=triage.outcome,
                    rule_meta=rule_meta,
                    provider=self.provider,
                    settings=self.settings,
                ),
            )
            self._ensure_active(case_id)
            if summary.draft_note:
                existing_note = self.db.get_note(case_id)
                if (
                    existing_note is not None
                    and existing_note.status.value != "draft_ai"
                ):
                    # Clinician edits and approvals are never overwritten by a
                    # later automated run.
                    summary.draft_note = existing_note
                else:
                    self.db.save_note(case_id, summary.draft_note)
            self.db.save_summary(summary)
        except CaseUnavailable:
            raise
        except Exception as exc:
            stage_failures.append(self._stage_failure("G_summary_generation", exc))
            summary = self._fallback_summary(case_id, triage, facts)
            self.db.save_summary(summary)

        # Final state - never silently downgrade an urgent workflow status.
        if not facts and (extraction.failures or stage_failures):
            case_status = CaseStatus.processing_failed
        elif extraction.failures or stage_failures:
            case_status = CaseStatus.partially_processed
        else:
            case_status = CaseStatus.processed

        finished = _now()
        case = self.db.get_case(case_id, include_deleted=True)
        if case is None:
            raise CaseUnavailable("case disappeared during processing")
        if (
            case.deleted_at is not None
            or case.status == CaseStatus.deleted
            or case.consent_withdrawn_at is not None
        ):
            raise CaseUnavailable("case was withdrawn or deleted during processing")

        # Advisory urgency is computed after all currently available evidence,
        # warnings, gaps and contradictions are assembled. It never changes the
        # deterministic workflow category. Clinician edits are preserved.
        urgency = self._run_stage(
            case_id,
            "G_urgency_assessment",
            lambda: assess_urgency(
                case_id=case_id,
                facts=facts,
                warnings=triage.warnings,
                gaps=gaps,
                discrepancies=discrepancies,
                timeline=timeline,
                outcome=triage.outcome,
                provider=self.provider,
                settings=self.settings,
                existing=case.urgency,
            ),
        )
        if urgency.source == UrgencySource.fallback:
            self.db.audit(
                actor,
                "system",
                "urgency_fallback",
                case_id,
                {
                    "error_code": urgency.model_error or "unknown",
                    "score": urgency.score,
                },
            )
        case.urgency = urgency

        meta = dict(case.meta)
        meta.update(
            {
                "last_processed_at": finished.isoformat(),
                "processing_started_at": starter,
                "awaiting_review_since": meta.get("awaiting_review_since") or finished.isoformat(),
                "rule_config_id": triage.config_id,
                "rule_config_version": triage.config_version,
                "rule_config_status": triage.config_status,
                "review_deadlines_minutes": triage.review_deadlines_minutes,
                "triggered_rule_ids": triage.triggered_rule_ids,
                "unresolved_rule_ids": triage.unresolved_rule_ids,
                "source_count": len(sources),
                "usable_source_count": len(usable_sources),
                "fact_count": len(facts),
                "processing_failures": [
                    {
                        "source_id": f.source_id,
                        "code": f.code,
                        "retryable": f.retryable,
                    }
                    for f in extraction.failures
                ],
                "stage_failures": stage_failures,
                "languages": interpretation.languages,
                "provider": self.provider.name,
                "model": self.provider.model,
            }
        )
        case.meta = meta
        case.status = case_status
        case.workflow_status = triage.outcome
        self.db.save_case(case)

        if self.followups is not None and summary is not None:
            try:
                self.followups.sync_from_summary(case, summary, actor=actor)
            except Exception as exc:
                self.db.audit(
                    actor,
                    "system",
                    "followup_sync_failed",
                    case_id,
                    {"error": safe_error(exc)},
                )

        self.db.audit(
            actor,
            "system",
            "pipeline_completed",
            case_id,
            {
                "case_status": case_status.value,
                "workflow_status": triage.outcome.value,
                "fact_count": len(facts),
                "warning_count": len(triage.warnings),
                "discrepancy_count": len(discrepancies),
                "failed_sources": len(extraction.failures),
                "stage_failures": len(stage_failures),
            },
        )
        for failure in stage_failures:
            self.db.audit(actor, "system", "pipeline_stage_failed", case_id, failure)

        return PipelineOutcome(
            case_id=case_id,
            case_status=case_status,
            workflow_status=triage.outcome,
            summary=summary,
            failures=[
                {"source_id": f.source_id, "code": f.code, "retryable": f.retryable}
                for f in extraction.failures
            ],
            stage_failures=stage_failures,
            triggered_rule_ids=triage.triggered_rule_ids,
            unresolved_rule_ids=triage.unresolved_rule_ids,
            fact_count=len(facts),
            warning_count=len(triage.warnings),
            discrepancy_count=len(discrepancies),
        )

    # --------------------------------------------------------------- helpers
    def _ensure_active(self, case_id: str) -> None:
        """Stop a run that races with consent withdrawal or case deletion."""
        current = self.db.get_case(case_id, include_deleted=True)
        if (
            current is None
            or current.deleted_at is not None
            or current.status == CaseStatus.deleted
            or current.consent_withdrawn_at is not None
        ):
            raise CaseUnavailable("case was withdrawn or deleted during processing")

    def _run_stage(self, case_id: str, stage: str, func):
        runs = self.db.get_runs(case_id)
        attempt = 1 + len([r for r in runs if r["stage"] == stage])
        run = ProcessingRun(
            run_id=f"{case_id}:{stage}:{attempt}",
            case_id=case_id,
            stage=stage,
            status="running",
            attempt=attempt,
            started_at=_now(),
        )
        self.db.save_run(run)
        try:
            value = func()
        except Exception as exc:
            run.status = "failed"
            run.error_code = type(exc).__name__
            run.error_detail = safe_error(exc)
            run.finished_at = _now()
            self.db.save_run(run)
            raise
        run.status = "succeeded"
        run.finished_at = _now()
        self.db.save_run(run)
        return value

    @staticmethod
    def _stage_failure(stage: str, exc: Exception) -> Dict[str, Any]:
        return {
            "stage": stage,
            "error_code": type(exc).__name__,
            "error_detail": safe_error(exc),
        }

    def _component_warnings(
        self,
        case_id: str,
        sources: Sequence[SourceRecord],
        extraction: ExtractionOutcome,
        interpretation,
    ) -> List[Warning]:
        warnings: List[Warning] = []

        if extraction.injection_codes:
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "injection")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="PROMPT_INJECTION_SUSPECTED",
                    label="Prompt-injection-like text detected in source material",
                    explanation="Source text contained instruction-like phrases such as: "
                    + ", ".join(extraction.injection_codes)
                    + ". It was treated as untrusted data and did not change system behavior.",
                    component="safety",
                )
            )

        for failure in extraction.failures:
            source = next((s for s in sources if s.source_id == failure.source_id), None)
            refs = []
            if source:
                refs = [
                    SourceRef(
                        source_id=source.source_id,
                        segment_id=source.segment_id,
                        quote=None,
                        page=source.page,
                        bbox=source.bbox,
                        audio_start=source.audio_start,
                        audio_end=source.audio_end,
                    )
                ]
            warnings.append(
                Warning(
                    warning_id="warn_"
                    + sha256_text(case_id + failure.source_id + failure.code)[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code=f"SOURCE_EXTRACTION_FAILED:{failure.code}",
                    label="A source could not be fully processed",
                    explanation=(
                        "Extraction failed or the source was unusable. Usable facts from other sources were retained; "
                        "this case cannot receive a reassuring routine status from incomplete processing."
                    ),
                    source_refs=refs,
                    component="extraction",
                )
            )

        if interpretation.ambiguous_source_ids:
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "ambiguous")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="AMBIGUOUS_LANGUAGE",
                    label="Ambiguous wording requires verification",
                    explanation="At least one source contains uncertain wording; extracted claims from it are marked uncertain.",
                    source_refs=[
                        SourceRef(source_id=s.source_id, segment_id=s.segment_id)
                        for s in sources
                        if s.source_id in interpretation.ambiguous_source_ids
                    ],
                    component="interpretation",
                )
            )

        low_quality = [
            s
            for s in sources
            if any(
                w in {"low_source_confidence", "unreadable_critical_field", "language_undetermined"}
                for w in s.warnings
            )
        ]
        if low_quality:
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "source_quality")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="SOURCE_QUALITY_WARNING",
                    label="Low-quality or incomplete source metadata",
                    explanation="One or more sources had low confidence, an unreadable field, or unknown language. Verify against the original.",
                    component="normalization",
                )
            )

        if not any(s.usable for s in sources):
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "no_usable")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="NO_USABLE_SOURCES",
                    label="No usable clinical source",
                    explanation="No non-empty, supported source was available. The case cannot be assessed automatically.",
                    outcome=Outcome.unassessed,
                    component="pipeline",
                )
            )
        return warnings

    def _fallback_summary(
        self, case_id: str, triage: TriageResult, facts: Sequence[Fact]
    ) -> ClinicalSummary:
        text = (
            "AI DRAFT - FOR CLINICIAN REVIEW\n"
            "Automated summary generation failed. No clinical conclusion is available.\n"
            f"Workflow status remains {triage.outcome.value}. A clinician must review the original sources.\n"
            "This note is not approved and must not be treated as clinical clearance."
        )
        note = DraftNote(
            note_id="note_" + sha256_text(case_id + "fallback")[:16],
            text=text,
            status="draft_ai",
            ai_generated=True,
        )
        return ClinicalSummary(
            case_id=case_id,
            pipeline_version=self.settings.pipeline_version,
            model=self.provider.model,
            warnings=[w for w in triage.warnings if w.kind.value == "rule"],
            component_warnings=[w for w in triage.warnings if w.kind.value != "rule"],
            workflow_status=triage.outcome,
            rule_config_id=triage.config_id,
            rule_config_version=triage.config_version,
            rule_config_status=triage.config_status,
            draft_note=note,
            disclaimers=[
                "Automated summary unavailable. Clinician review of original sources is required."
            ],
        )
