"""Stage F - missing-information detection and follow-up question suggestions."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from ..schemas import (
    Discrepancy,
    Fact,
    FactStatus,
    FactType,
    FollowUpQuestion,
    Gap,
    Outcome,
    SourceRecord,
)
from ..storage import sha256_text
from .extract import SourceFailure

_TEMPLATES = {
    "en": {
        "ONSET_UNKNOWN": "When did your symptoms first start?",
        "DURATION_UNKNOWN": "How long have you had these symptoms?",
        "SEVERITY_UNKNOWN": "How severe are your symptoms right now?",
        "MEDICATION_UNKNOWN": "Are you currently taking any medicines? If yes, which ones?",
        "ALLERGY_UNKNOWN": "Do you have any known allergies?",
        "HISTORY_UNKNOWN": "Do you have any ongoing medical conditions?",
        "MEASUREMENTS_UNKNOWN": "Do you have any recent measurements such as temperature, blood pressure, or oxygen saturation?",
    },
    "hi": {
        "ONSET_UNKNOWN": "आपके लक्षण पहली बार कब शुरू हुए?",
        "DURATION_UNKNOWN": "ये लक्षण कितने समय से हैं?",
        "SEVERITY_UNKNOWN": "अभी आपके लक्षण कितने गंभीर हैं?",
        "MEDICATION_UNKNOWN": "क्या आप अभी कोई दवा ले रहे हैं? अगर हाँ, कौन सी?",
        "ALLERGY_UNKNOWN": "क्या आपको किसी चीज़ से एलर्जी है?",
        "HISTORY_UNKNOWN": "क्या आपको कोई पुरानी बीमारी है?",
        "MEASUREMENTS_UNKNOWN": "क्या आपके पास हाल की माप है, जैसे तापमान, रक्तचाप या ऑक्सीजन स्तर?",
    },
    "hinglish": {
        "ONSET_UNKNOWN": "Aapke symptoms pehli baar kab shuru hue?",
        "DURATION_UNKNOWN": "Ye symptoms kitne samay se hain?",
        "SEVERITY_UNKNOWN": "Abhi aapke symptoms kitne severe hain?",
        "MEDICATION_UNKNOWN": "Kya aap abhi koi dawa le rahe hain? Agar haan, kaun si?",
        "ALLERGY_UNKNOWN": "Kya aapko kisi cheez se allergy hai?",
        "HISTORY_UNKNOWN": "Kya aapko koi purani bimari hai?",
        "MEASUREMENTS_UNKNOWN": "Kya aapke paas recent measurement hai, jaise temperature, BP ya oxygen level?",
    },
}

_GAP_DESCRIPTIONS = {
    "ONSET_UNKNOWN": "Symptom onset is not documented.",
    "DURATION_UNKNOWN": "Symptom duration is not documented.",
    "SEVERITY_UNKNOWN": "Symptom severity is not documented.",
    "MEDICATION_UNKNOWN": "Current medication history is not documented.",
    "ALLERGY_UNKNOWN": "Allergy history is not documented.",
    "HISTORY_UNKNOWN": "Relevant past medical history is not documented.",
    "MEASUREMENTS_UNKNOWN": "No recent measurements were extracted.",
    "NO_USABLE_INPUT": "No usable clinical source was available.",
    "EXTRACTION_INCOMPLETE": "One or more sources failed extraction; information is incomplete.",
    "CONTRADICTION_REVIEW": "Candidate contradictions require human verification.",
    "SOURCE_REFERENCE_GAPS": "Some claims have no traceable source reference.",
}

_BLOCKED_QUESTION = re.compile(
    r"(?i)\b(diagnos|prescrib|take\s+\w+\s+mg|you\s+have|you\s+are\s+safe|no\s+need|discharge|ignore\s+instructions)\b"
)


def _question_language(sources: Sequence[SourceRecord]) -> str:
    for source in sources:
        if not source.usable:
            continue
        if source.language in {"hi", "hinglish", "en"}:
            return source.language
        if source.language and source.language != "unknown":
            return "en"
    return "en"


def _question_id(text: str, language: str) -> str:
    return "q_" + sha256_text(language + text)[:16]


def detect_gaps(
    facts: Sequence[Fact],
    discrepancies: Sequence[Discrepancy],
    sources: Sequence[SourceRecord],
    failures: Sequence[SourceFailure],
    outcome: Outcome,
    model_questions: Iterable[str] = (),
) -> Tuple[List[Gap], List[FollowUpQuestion]]:
    gaps: List[Gap] = []
    symptoms = [
        f
        for f in facts
        if f.fact_type == FactType.symptom
        and f.status == FactStatus.present
        and f.verification_status.value != "rejected"
    ]
    present = [f for f in facts if f.verification_status.value != "rejected"]

    if not any(s.usable for s in sources):
        gaps.append(Gap(code="NO_USABLE_INPUT", description=_GAP_DESCRIPTIONS["NO_USABLE_INPUT"], priority="high"))
    if failures:
        gaps.append(
            Gap(
                code="EXTRACTION_INCOMPLETE",
                description=_GAP_DESCRIPTIONS["EXTRACTION_INCOMPLETE"],
                priority="high",
            )
        )
    if symptoms and not any(f.temporal and f.temporal.raw_expression for f in symptoms):
        gaps.append(
            Gap(
                code="ONSET_UNKNOWN",
                description=_GAP_DESCRIPTIONS["ONSET_UNKNOWN"],
                related_fact_ids=[f.fact_id for f in symptoms],
                priority="review",
            )
        )
    if symptoms and not any(
        f.temporal and f.temporal.offset_hours is not None for f in symptoms
    ):
        gaps.append(
            Gap(
                code="DURATION_UNKNOWN",
                description=_GAP_DESCRIPTIONS["DURATION_UNKNOWN"],
                related_fact_ids=[f.fact_id for f in symptoms],
                priority="review",
            )
        )
    if symptoms:
        gaps.append(
            Gap(
                code="SEVERITY_UNKNOWN",
                description=_GAP_DESCRIPTIONS["SEVERITY_UNKNOWN"],
                related_fact_ids=[f.fact_id for f in symptoms],
                priority="review",
            )
        )
    if not any(f.fact_type in (FactType.medication, FactType.medication_history) for f in present):
        gaps.append(Gap(code="MEDICATION_UNKNOWN", description=_GAP_DESCRIPTIONS["MEDICATION_UNKNOWN"], priority="review"))
    if not any("allergy" in (f.label or "").lower() for f in present):
        gaps.append(Gap(code="ALLERGY_UNKNOWN", description=_GAP_DESCRIPTIONS["ALLERGY_UNKNOWN"], priority="info"))
    if not any(f.fact_type == FactType.history for f in present):
        gaps.append(Gap(code="HISTORY_UNKNOWN", description=_GAP_DESCRIPTIONS["HISTORY_UNKNOWN"], priority="info"))
    if not any(
        f.fact_type in (FactType.measurement, FactType.vital_sign) for f in present
    ):
        gaps.append(Gap(code="MEASUREMENTS_UNKNOWN", description=_GAP_DESCRIPTIONS["MEASUREMENTS_UNKNOWN"], priority="info"))
    if discrepancies:
        gaps.append(
            Gap(
                code="CONTRADICTION_REVIEW",
                description=_GAP_DESCRIPTIONS["CONTRADICTION_REVIEW"],
                related_fact_ids=sorted({fid for d in discrepancies for fid in d.fact_ids}),
                priority="high",
            )
        )
    if any(not f.source_refs for f in present):
        gaps.append(
            Gap(
                code="SOURCE_REFERENCE_GAPS",
                description=_GAP_DESCRIPTIONS["SOURCE_REFERENCE_GAPS"],
                related_fact_ids=[f.fact_id for f in present if not f.source_refs],
                priority="high",
            )
        )

    language = _question_language(sources)
    questions: List[FollowUpQuestion] = []

    if outcome != Outcome.emergency:
        templates = _TEMPLATES.get(language, _TEMPLATES["en"])
        for gap in gaps:
            if gap.code not in templates:
                continue
            text = templates[gap.code]
            questions.append(
                FollowUpQuestion(
                    question_id=_question_id(text, language),
                    text=text,
                    language=language,
                    addresses=[gap.code],
                    rationale="Missing information identified from the structured record.",
                )
            )
        for text in model_questions:
            text = (text or "").strip()
            if not text or _BLOCKED_QUESTION.search(text):
                continue
            questions.append(
                FollowUpQuestion(
                    question_id=_question_id(text, language),
                    text=text,
                    language=language,
                    addresses=[],
                    rationale="Suggested by the language model; verify before asking.",
                )
            )
    else:
        gaps.append(
            Gap(
                code="QUESTIONS_SUPPRESSED_EMERGENCY",
                description="Routine follow-up questions are suppressed because an emergency workflow status is active. Escalation must not wait for answers.",
                priority="high",
            )
        )

    deduped: List[FollowUpQuestion] = []
    seen = set()
    for question in questions:
        if question.question_id in seen:
            continue
        seen.add(question.question_id)
        deduped.append(question)
    return gaps, deduped[:5]
