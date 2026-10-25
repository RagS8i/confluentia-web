"""Stage C - structured fact extraction with schema validation and retries.

Each source is extracted independently so a failure on one document cannot
destroy usable facts from another source.  Facts are only attributed to source
text when an exact (or whitespace-normalized) quote can be located; otherwise
they are retained as explicitly unverified AI inference with a warning.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from ..config import Settings
from ..model.provider import BaseProvider, ModelError
from ..schemas import (
    Assertion,
    ExtractionMeta,
    ExtractedFact,
    Fact,
    FactStatus,
    FactType,
    LanguageStatus,
    Modality,
    SourceRecord,
    SourceRef,
)
from ..safety import scan_sources
from ..storage import sha256_text
from .timeline import find_temporal, parse_temporal

_NEGATION = re.compile(
    r"(?i)\b(no|not|without|denies|denied|never|n't|nahi|nahin|à¤¨à¤¹à¥€à¤‚|à¤¬à¤¿à¤¨à¤¾)\b"
)


@dataclass
class SourceFailure:
    source_id: str
    code: str
    message: str
    retryable: bool = True


@dataclass
class ExtractionOutcome:
    facts: List[Fact] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    failures: List[SourceFailure] = field(default_factory=list)
    injection_codes: List[str] = field(default_factory=list)
    used_source_ids: List[str] = field(default_factory=list)
    failed_source_ids: List[str] = field(default_factory=list)
    attempts: int = 0
    model: Optional[str] = None
    narrative: Optional[str] = None
    draft_note: Optional[str] = None


def _locate(source_text: str, quote: str) -> Optional[Tuple[int, int]]:
    quote = (quote or "").strip()
    if not quote:
        return None
    index = source_text.find(quote)
    if index >= 0:
        return index, index + len(quote)
    flexible = re.escape(quote).replace(r"\ ", r"\s+")
    match = re.search(flexible, source_text, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def _source_ref(
    source: SourceRecord, start: Optional[int], end: Optional[int], quote: str
) -> SourceRef:
    return SourceRef(
        source_id=source.source_id,
        segment_id=source.segment_id,
        quote=quote,
        start_char=start,
        end_char=end,
        audio_start=source.audio_start,
        audio_end=source.audio_end,
        page=source.page,
        bbox=source.bbox,
    )


def _modality_assertion(source: SourceRecord) -> Assertion:
    if source.modality == Modality.ocr:
        return Assertion.document_derived
    if source.modality in (Modality.stt, Modality.typed_message):
        return Assertion.patient_reported
    return Assertion.unknown


def _fact_from_extracted(
    case_id: str,
    source: SourceRecord,
    extracted: ExtractedFact,
    settings: Settings,
    retries: int,
    model: str,
) -> Fact:
    warnings: List[str] = []
    quote = (extracted.original_text or "").strip()
    located = _locate(source.original_text, quote)
    if located:
        start, end = located
        refs = [_source_ref(source, start, end, quote)]
    else:
        start, end = None, None
        refs = []
        warnings.append("no_source_reference")

    status = extracted.status
    if status == FactStatus.absent and not _NEGATION.search(quote or ""):
        status = FactStatus.uncertain
        warnings.append("negation_not_explicitly_supported")

    assertion = extracted.assertion
    if assertion in (Assertion.ai_generated, Assertion.inferred_for_review):
        pass
    elif not located:
        assertion = Assertion.inferred_for_review
    else:
        assertion = _modality_assertion(source)

    ai_suggestion = assertion in (Assertion.ai_generated, Assertion.inferred_for_review)

    language = source.language
    language_status = source.language_status
    if language == "unknown":
        warnings.append("language_unknown")
    if language_status == LanguageStatus.unsupported:
        warnings.append("unsupported_language_interpretation_uncertain")

    expression = extracted.symptom_onset or extracted.duration
    if not expression:
        expression = find_temporal(quote)
    temporal = None
    if expression:
        reference = source.captured_at or source.received_at or datetime.now(timezone.utc)
        temporal = parse_temporal(expression, reference)
        if temporal and temporal.certainty == "uncertain" and temporal.offset_hours is None:
            warnings.append("unparsed_time_expression")

    confidence = extracted.confidence
    if confidence is None:
        confidence = source.confidence if source.confidence is not None else 0.6
    if "low_source_confidence" in source.warnings:
        confidence = min(confidence, 0.45)
        warnings.append("low_extraction_confidence")
    if status == FactStatus.uncertain:
        confidence = min(confidence, 0.5)
    if not located:
        confidence = min(confidence, 0.4)

    fact_type = extracted.fact_type
    label = extracted.label.strip() if extracted.label else fact_type.value
    fact_id = "fact_" + sha256_text(
        "|".join(
            [
                case_id,
                source.source_id,
                fact_type.value,
                label.lower(),
                quote,
                status.value,
            ]
        )
    )[:20]

    fact = Fact(
        fact_id=fact_id,
        case_id=case_id,
        fact_type=fact_type,
        label=label,
        value=extracted.value,
        value_normalized=extracted.normalized_text,
        unit=extracted.unit,
        status=status,
        assertion=assertion,
        language=language,
        language_status=language_status,
        original_text=quote,
        normalized_text=extracted.normalized_text,
        source_refs=refs,
        temporal=temporal,
        extraction=ExtractionMeta(
            method="slm",
            model=model,
            model_version=None,
            prompt_version=settings.prompt_version,
            retries=retries,
        ),
        verification_status="unverified",
        confidence=confidence,
        warnings=warnings,
        ai_suggestion=ai_suggestion,
    )
    if extracted.notes:
        fact.review_note = extracted.notes
    return fact


def extract_facts(
    case_id: str,
    sources: Sequence[SourceRecord],
    provider: BaseProvider,
    settings: Settings,
) -> ExtractionOutcome:
    outcome = ExtractionOutcome(model=provider.model)
    outcome.injection_codes = scan_sources(sources).codes

    for source in sources:
        outcome.used_source_ids.append(source.source_id)
        if not source.usable:
            outcome.failed_source_ids.append(source.source_id)
            outcome.failures.append(
                SourceFailure(
                    source_id=source.source_id,
                    code="unusable_source",
                    message="source was empty, unsupported or unusable and was not sent to the model",
                    retryable=False,
                )
            )
            continue

        last_error: Optional[ModelError] = None
        result = None
        attempts = 0
        for attempt in range(settings.max_model_retries + 1):
            attempts = attempt
            try:
                result = provider.extract_facts([source], settings)
                last_error = None
                break
            except ModelError as exc:
                last_error = exc
                if attempt < settings.max_model_retries and exc.retryable:
                    time.sleep(settings.model_backoff_s * (2 ** attempt))
                    continue
                break
            except Exception as exc:  # pragma: no cover - defensive boundary
                last_error = ModelError("internal", f"unexpected model error ({type(exc).__name__})", False)
                break

        outcome.attempts += attempts + 1
        if last_error is not None or result is None:
            outcome.failed_source_ids.append(source.source_id)
            outcome.failures.append(
                SourceFailure(
                    source_id=source.source_id,
                    code=last_error.code if last_error else "unknown",
                    message=last_error.code if last_error else "model extraction failed",
                    retryable=last_error.retryable if last_error else True,
                )
            )
            continue

        outcome.questions.extend(q for q in result.follow_up_questions if q)
        outcome.notes.extend(n for n in result.interpretation_notes if n)
        if result.summary_narrative:
            outcome.narrative = result.summary_narrative
        if result.draft_referral_note:
            outcome.draft_note = result.draft_referral_note

        for extracted in result.facts:
            fact = _fact_from_extracted(case_id, source, extracted, settings, attempts, provider.model)
            outcome.facts.append(fact)

            if (
                fact.fact_type == FactType.symptom
                and extracted.progression
                and extracted.progression.strip()
            ):
                progression_quote = fact.original_text
                progression_label = f"progression of {fact.label}"
                outcome.facts.append(
                    Fact(
                        fact_id="fact_"
                        + sha256_text(
                            case_id
                            + source.source_id
                            + "progression"
                            + progression_label
                            + progression_quote
                        )[:20],
                        case_id=case_id,
                        fact_type=FactType.progression,
                        label=progression_label,
                        value=extracted.progression.strip(),
                        status=FactStatus.present
                        if fact.status != FactStatus.uncertain
                        else FactStatus.uncertain,
                        assertion=fact.assertion,
                        language=fact.language,
                        language_status=fact.language_status,
                        original_text=progression_quote,
                        normalized_text=extracted.progression.strip(),
                        source_refs=fact.source_refs,
                        extraction=fact.extraction,
                        verification_status="unverified",
                        confidence=fact.confidence,
                        warnings=list(fact.warnings),
                        ai_suggestion=fact.ai_suggestion,
                    )
                )

    # Deduplicate identical facts (enables retries and repeated ingest).
    deduped: List[Fact] = []
    seen: set[str] = set()
    for fact in outcome.facts:
        if fact.fact_id in seen:
            continue
        seen.add(fact.fact_id)
        deduped.append(fact)
    outcome.facts = deduped
    return outcome

