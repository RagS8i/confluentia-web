"""Stage D - symptom timeline construction.

Raw time expressions are always preserved.  Normalized estimates are marked
approximate and are derived only from a known source receipt time; no exact
calendar date is invented.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..schemas import Fact, FactType, TimelineEntry, Temporal
from ..storage import sha256_text

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "ek": 1, "do": 2,
    "teen": 3, "char": 4, "paanch": 5, "एक": 1, "दो": 2, "तीन": 3,
    "चार": 4, "पांच": 5,
}

_UNIT_HOURS = {
    "hour": 1.0, "hours": 1.0, "hr": 1.0, "hrs": 1.0,
    "day": 24.0, "days": 24.0, "din": 24.0, "दिन": 24.0,
    "week": 168.0, "weeks": 168.0, "hafte": 168.0, "हफ्ते": 168.0,
    "month": 720.0, "months": 720.0, "mahine": 720.0, "महीने": 720.0,
    "year": 8760.0, "years": 8760.0, "saal": 8760.0, "sal": 8760.0, "साल": 8760.0,
}

_TIME_HINT_RE = re.compile(
    r"(?i)(since\s+(?:yesterday|last\s+week|last\s+month|childhood|birth)|"
    r"(?:this\s+morning|last\s+night|yesterday|today|aaj\s+subah|raat\s+se|kal|आज|कल)|"
    r"for\s+(?:approximately\s+|about\s+)?[\w\d.]+\s*(?:hours?|hrs?|days?|din|weeks?|hafte|months?|mahine|years?|saal|दिन|हफ्ते|महीने|साल)|"
    r"[\w\d.]+\s*(?:hours?|hrs?|days?|din|weeks?|months?|years?|saal)\s+(?:ago|pehle|पहले)|"
    r"\d+\s*(?:din|दिन)\s+se)"
)

_QUANTITY_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|ek|do|teen|char|paanch|एक|दो|तीन|चार|पांच)"
    r"\s*(hours?|hrs?|days?|din|दिन|weeks?|hafte|हफ्ते|months?|mahine|महीने|years?|saal|sal|साल)"
)


def find_temporal(text: str) -> Optional[str]:
    if not text:
        return None
    match = _TIME_HINT_RE.search(text)
    return match.group(0).strip() if match else None


def _quantity(match: re.Match) -> Optional[float]:
    raw = match.group(1).lower()
    if raw in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[raw])
    try:
        return float(raw)
    except ValueError:
        return None


def _explicit_date(expression: str) -> Optional[datetime]:
    text = expression.strip()
    for pattern in (r"(\d{4})-(\d{2})-(\d{2})", r"(\d{1,2})/(\d{1,2})/(\d{4})"):
        match = re.search(pattern, text)
        if match:
            try:
                if pattern.startswith(r"(\d{4})"):
                    return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc)
                return datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)), tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def parse_temporal(expression: Optional[str], reference: Optional[datetime]) -> Optional[Temporal]:
    if not expression or not str(expression).strip():
        return None
    raw = str(expression).strip()
    low = raw.lower()
    reference = reference or None

    explicit = _explicit_date(raw)
    if explicit:
        return Temporal(
            raw_expression=raw,
            normalized_expression=explicit.date().isoformat(),
            estimated_start_at=explicit,
            precision="day",
            reference_basis="explicit_date",
            certainty="certain",
        )

    offset_hours: Optional[float] = None
    precision = "unknown"
    certainty = "approximate"

    if re.search(r"(?i)\b(since\s+childhood|birth|बचपन)\b", raw):
        return Temporal(
            raw_expression=raw,
            normalized_expression="long-term (since childhood/birth)",
            precision="unknown",
            reference_basis="unknown",
            certainty="uncertain",
            notes="Long-term expression; no numeric offset inferred.",
        )

    if re.search(r"(?i)\b(since\s+)?yesterday\b|\bkal\b|\bकल\b", raw):
        offset_hours, precision = 24.0, "day"
    elif re.search(r"(?i)\b(this\s+morning|aaj\s+subah|आज\s+सुबह|morning)\b", raw):
        offset_hours, precision, certainty = 8.0, "hour", "approximate"
    elif re.search(r"(?i)\b(last\s+night|raat\s+se|रात\s+से)\b", raw):
        offset_hours, precision = 12.0, "hour"
    elif re.search(r"(?i)\b(today|aaj|आज)\b", raw):
        offset_hours, precision = 12.0, "day"
    elif re.search(r"(?i)\b(since\s+)?last\s+week\b", raw):
        offset_hours, precision = 168.0, "week"
    elif re.search(r"(?i)\b(since\s+)?last\s+month\b", raw):
        offset_hours, precision = 720.0, "month"
    else:
        quantity = _QUANTITY_RE.search(raw)
        if quantity:
            amount = _quantity(quantity)
            unit = quantity.group(2).lower()
            factor = _UNIT_HOURS.get(unit)
            if amount is not None and factor:
                offset_hours = amount * factor
                precision = "hour" if factor == 1 else "day" if factor == 24 else "week" if factor == 168 else "month"

    if offset_hours is None:
        return Temporal(
            raw_expression=raw,
            normalized_expression=None,
            precision="unknown",
            reference_basis="unknown",
            certainty="uncertain",
            notes="Time expression preserved but not normalized; clinician verification required.",
        )

    estimated = None
    basis = "unknown"
    if reference is not None:
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        estimated = reference - timedelta(hours=offset_hours)
        basis = "source_received_at"

    return Temporal(
        raw_expression=raw,
        normalized_expression=f"approximately {offset_hours:g} hours before source receipt ({precision} precision)",
        offset_hours=offset_hours,
        estimated_start_at=estimated,
        precision=precision,
        reference_basis=basis,
        certainty=certainty,
    )


def build_timeline(facts: List[Fact]) -> List[TimelineEntry]:
    entries: List[TimelineEntry] = []
    seen: set[str] = set()
    for fact in facts:
        if fact.verification_status.value == "rejected":
            continue
        if not fact.temporal and fact.fact_type not in {FactType.symptom, FactType.progression}:
            continue
        if fact.temporal is None:
            continue
        temporal = fact.temporal
        entry_id = "tl_" + sha256_text(fact.fact_id + (temporal.raw_expression or ""))[:16]
        if entry_id in seen:
            continue
        seen.add(entry_id)
        entries.append(
            TimelineEntry(
                entry_id=entry_id,
                fact_id=fact.fact_id,
                label=fact.label,
                raw_time=temporal.raw_expression,
                normalized_time=temporal.normalized_expression,
                offset_hours=temporal.offset_hours,
                precision=temporal.precision,
                reference_basis=temporal.reference_basis,
                estimated_start_at=temporal.estimated_start_at,
                certainty=temporal.certainty,
                progression=(fact.value if fact.fact_type == FactType.progression else None),
                source_refs=fact.source_refs,
            )
        )

    def sort_key(entry: TimelineEntry):
        if entry.estimated_start_at is not None:
            return (0, entry.estimated_start_at.timestamp(), entry.label)
        if entry.offset_hours is not None:
            return (1, -entry.offset_hours, entry.label)
        return (2, 0.0, entry.label)

    return sorted(entries, key=sort_key)
