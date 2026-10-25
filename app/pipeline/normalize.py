"""Stage A - input normalization, STT adapter, and structured OCR mapping.

The STT adapter accepts the vendor payload as delivered. OCR no longer accepts
arbitrary vendor layouts: the in-process organizer emits a strict source-level
contract, and ``sources_from_ocr_pipeline`` maps that to persistence records.
Both paths preserve provenance (timestamps, pages, optional bounding boxes
and confidences, document identifiers and language metadata), and original text
is never overwritten by the normalized representation.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..language import resolve_language
from ..safety import detect_prompt_injection
from ..schemas import LanguageStatus, Modality, SourceRecord
from ..storage import sha256_text


class InputError(ValueError):
    """Safe, user-facing input problem with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        ch
        for ch in text
        if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
    )
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    if number < 0.0:
        return None
    return min(number, 1.0)


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            if {"x0", "y0", "x1", "y1"} <= set(value):
                return [float(value["x0"]), float(value["y0"]), float(value["x1"]), float(value["y1"])]
            if {"x", "y", "width", "height"} <= set(value):
                x, y = float(value["x"]), float(value["y"])
                return [x, y, x + float(value["width"]), y + float(value["height"])]
        except (TypeError, ValueError):
            return None
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    return None


def _segment_source_id(prefix: str, case_id: str, base: str, index: int) -> str:
    return f"{prefix}:{case_id[:10]}:{base}:{index}"


def _base_id(payload: Dict[str, Any], fallback_text: str, prefix: str) -> str:
    for key in ("transcript_id", "document_id", "message_id", "request_id", "id"):
        value = payload.get(key)
        if value:
            return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))[:64]
    return f"{prefix}-{sha256_text(fallback_text)[:12]}"


def _build_source(
    *,
    case_id: str,
    modality: Modality,
    source_id: str,
    segment_id: str,
    original: str,
    normalized: str,
    declared_language: Optional[str],
    source_language_status: Optional[str],
    confidence: Optional[float],
    document_ref: Optional[str],
    page: Optional[int],
    bbox: Optional[List[float]],
    audio_start: Optional[float],
    audio_end: Optional[float],
    captured_at: Optional[datetime],
    parent_source_id: Optional[str],
    extra_warnings: Optional[Iterable[str]] = None,
    raw_metadata: Optional[Dict[str, Any]] = None,
) -> SourceRecord:
    warnings = list(extra_warnings or [])
    if not normalized:
        warnings.append("empty_text")
        language, language_status = "unknown", LanguageStatus.unknown
    else:
        language, language_status, lang_warnings = resolve_language(
            original, declared_language
        )
        warnings.extend(lang_warnings)
        if source_language_status == "declared" and language_status == LanguageStatus.heuristic:
            language_status = LanguageStatus.declared
    injections = detect_prompt_injection(original) if original else []
    for code in injections:
        warnings.append(f"prompt_injection_suspected:{code}")

    if confidence is not None and confidence < 0.5:
        warnings.append("low_source_confidence")
    if source_language_status == "unsupported":
        language_status = LanguageStatus.unsupported

    usable = bool(normalized) and len(normalized) >= 2 and language_status != LanguageStatus.unsupported
    if not normalized:
        usable = False

    return SourceRecord(
        source_id=source_id,
        parent_source_id=parent_source_id,
        case_id=case_id,
        modality=modality,
        segment_id=segment_id,
        original_text=original,
        normalized_text=normalized,
        language=language,
        language_status=language_status,
        confidence=confidence,
        content_hash=sha256_text(normalized),
        document_ref=document_ref,
        page=page,
        bbox=bbox,
        audio_start=audio_start,
        audio_end=audio_end,
        captured_at=captured_at,
        received_at=datetime.now(timezone.utc),
        usable=usable,
        warnings=warnings,
        raw_metadata=raw_metadata or {},
    )


def adapt_stt(payload: Dict[str, Any], case_id: str) -> Tuple[List[SourceRecord], List[str]]:
    if not isinstance(payload, dict):
        raise InputError("malformed_payload", "STT payload must be a JSON object")
    segments = (
        payload.get("segments")
        or payload.get("results")
        or payload.get("chunks")
        or []
    )
    top_text = (
        payload.get("transcription")
        or payload.get("text")
        or payload.get("transcript")
        or payload.get("translated_text")
    )
    if not segments and not top_text:
        raise InputError("empty_transcript", "STT payload contains no transcript text")
    if segments and not isinstance(segments, list):
        raise InputError("malformed_payload", "STT segments must be a list")

    declared = (
        payload.get("language")
        or payload.get("language_code")
        or payload.get("lang")
        or payload.get("target_language")
    )
    status = payload.get("language_status")
    captured = _parse_dt(payload.get("captured_at") or payload.get("recorded_at"))
    doc_ref = (
        payload.get("document_id")
        or payload.get("document_ref")
        or payload.get("id")
        or payload.get("file_name")
        or payload.get("filename")
    )
    base = _base_id(payload, str(top_text or ""), "stt")
    parent = f"stt:{case_id[:10]}:{base}"
    notes: List[str] = []

    records: List[SourceRecord] = []
    if segments:
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                notes.append(f"segment_{index}_invalid")
                continue
            text = (
                segment.get("transcription")
                or segment.get("text")
                or segment.get("transcript")
                or segment.get("translated_text")
                or ""
            )
            original = str(text)
            normalized = normalize_text(original)
            if not normalized:
                notes.append(f"segment_{index}_empty")
                continue
            records.append(
                _build_source(
                    case_id=case_id,
                    modality=Modality.stt,
                    source_id=_segment_source_id("stt", case_id, base, index),
                    segment_id=f"s{index}",
                    original=original,
                    normalized=normalized,
                    declared_language=(
                        segment.get("language")
                        or segment.get("language_code")
                        or declared
                    ),
                    source_language_status=segment.get("language_status") or status,
                    confidence=_norm_confidence(
                        segment.get(
                            "confidence",
                            segment.get(
                                "language_probability",
                                payload.get("confidence"),
                            ),
                        )
                    ),
                    document_ref=str(doc_ref) if doc_ref else None,
                    page=None,
                    bbox=None,
                    audio_start=segment.get(
                        "start",
                        segment.get("start_time", segment.get("start_seconds")),
                    ),
                    audio_end=segment.get(
                        "end",
                        segment.get("end_time", segment.get("end_seconds")),
                    ),
                    captured_at=captured,
                    parent_source_id=parent,
                    raw_metadata={
                        "transcript_id": payload.get("transcript_id") or payload.get("id"),
                        "segment_index": index,
                        "model": payload.get("model") or payload.get("stt_model"),
                        "document_id": str(doc_ref) if doc_ref else None,
                        "translated_text": segment.get("translated_text"),
                    },
                )
            )
    else:
        original = str(top_text)
        normalized = normalize_text(original)
        if not normalized:
            raise InputError("empty_transcript", "STT transcript is empty")
        records.append(
            _build_source(
                case_id=case_id,
                modality=Modality.stt,
                source_id=_segment_source_id("stt", case_id, base, 0),
                segment_id="s0",
                original=original,
                normalized=normalized,
                declared_language=declared,
                source_language_status=status,
                confidence=_norm_confidence(payload.get("confidence")),
                document_ref=str(doc_ref) if doc_ref else None,
                page=None,
                bbox=None,
                audio_start=payload.get("start"),
                audio_end=payload.get("end"),
                captured_at=captured,
                parent_source_id=parent,
                raw_metadata={
                    "transcript_id": payload.get("transcript_id") or payload.get("id"),
                    "model": payload.get("model") or payload.get("stt_model"),
                    "document_id": str(doc_ref) if doc_ref else None,
                    "translated_text": payload.get("translated_text"),
                },
            )
        )
    if not records:
        raise InputError(
            "empty_transcript", "STT payload contains no usable transcript text"
        )
    return records, notes


def sources_from_ocr_pipeline(
    payload: Dict[str, Any], case_id: str
) -> Tuple[List[SourceRecord], List[str]]:
    """Convert the organizer's structured OCR output into source records.

    The OCR pipeline has already filtered non-medical/unrequired blocks and
    supplied exact text, page and bbox provenance. This function only performs
    the deterministic persistence mapping; it does not parse vendor layouts or
    accept an arbitrary OCR payload.
    """
    if not isinstance(payload, dict):
        raise InputError("malformed_payload", "OCR payload must be a JSON object")
    if payload.get("is_medical") is not True:
        raise InputError(
            "non_medical_document",
            "The uploaded document does not contain clinically relevant content",
        )

    blocks = payload.get("sources")
    if not isinstance(blocks, list) or not blocks:
        raise InputError(
            "empty_medical_document",
            "The OCR pipeline found no medically relevant source text",
        )

    declared = payload.get("language") or payload.get("language_code")
    status = payload.get("language_status")
    captured = _parse_dt(payload.get("captured_at") or payload.get("processed_at"))
    doc_ref = (
        payload.get("document_id")
        or payload.get("file_name")
        or payload.get("filename")
        or payload.get("id")
    )
    base = _base_id(payload, "", "ocr")
    parent = f"ocr:{case_id[:10]}:{base}"
    notes: List[str] = []
    records: List[SourceRecord] = []

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            notes.append(f"source_{index}_invalid")
            continue

        original = block.get("text") or ""
        if not isinstance(original, str):
            original = str(original)
        normalized = normalize_text(original)
        if not normalized:
            notes.append(f"source_{index}_empty")
            continue

        page_raw = block.get("page", block.get("page_number"))
        try:
            page_no = int(page_raw) if page_raw is not None else None
        except (TypeError, ValueError):
            page_no = None

        block_id = block.get("block_id")
        segment_id = str(block_id) if block_id else f"p{page_no or 0}-b{index}"

        records.append(
            _build_source(
                case_id=case_id,
                modality=Modality.ocr,
                source_id=_segment_source_id("ocr", case_id, base, index),
                segment_id=segment_id,
                original=original,
                normalized=normalized,
                declared_language=declared,
                source_language_status=status,
                confidence=_norm_confidence(block.get("confidence")),
                document_ref=str(doc_ref) if doc_ref else None,
                page=page_no,
                bbox=_bbox(block.get("bbox") or block.get("box")),
                audio_start=None,
                audio_end=None,
                captured_at=captured,
                parent_source_id=parent,
                extra_warnings=(
                    ["unreadable_critical_field"] if block.get("unreadable") else None
                ),
                raw_metadata={
                    "document_id": payload.get("document_id") or payload.get("id"),
                    "file_name": payload.get("file_name") or payload.get("filename"),
                    "ocr_model": payload.get("model") or payload.get("ocr_model"),
                    "document_type": payload.get("document_type"),
                    "block": block,
                },
            )
        )

    if not records:
        raise InputError(
            "empty_medical_document",
            "The OCR pipeline found no usable medically relevant text",
        )
    return records, notes


def adapt_typed(payload: Dict[str, Any], case_id: str) -> Tuple[List[SourceRecord], List[str]]:
    if not isinstance(payload, dict):
        raise InputError("malformed_payload", "Message payload must be a JSON object")
    text = payload.get("text") or payload.get("message")
    if not text or not normalize_text(str(text)):
        raise InputError("empty_message", "Patient message is empty")
    base = _base_id(payload, str(text), "msg")
    record = _build_source(
        case_id=case_id,
        modality=Modality.typed_message,
        source_id=f"msg:{case_id[:10]}:{base}",
        segment_id="m0",
        original=str(text),
        normalized=normalize_text(str(text)),
        declared_language=payload.get("language"),
        source_language_status=payload.get("language_status"),
        confidence=_norm_confidence(payload.get("confidence")),
        document_ref=None,
        page=None,
        bbox=None,
        audio_start=None,
        audio_end=None,
        captured_at=_parse_dt(payload.get("captured_at")),
        parent_source_id=f"msg:{case_id[:10]}:{base}",
        raw_metadata={"message_id": payload.get("message_id") or payload.get("id")},
    )
    return [record], []
