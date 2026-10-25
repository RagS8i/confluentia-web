"""Stage E - cross-modal reconciliation.

Differences are never silently resolved.  Each candidate discrepancy is
classified and linked to the underlying claims and source material so a human
can adjudicate it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..schemas import (
    Discrepancy,
    DiscrepancyCategory,
    Fact,
    FactStatus,
    FactType,
    Severity,
    SourceRecord,
    SourceRef,
)
from ..storage import sha256_text

_CANONICAL = {
    "breathlessness": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "bp": "blood pressure",
    "blood pressure": "blood pressure",
    "temp": "temperature",
    "temperature": "temperature",
    "sugar": "blood glucose",
    "blood sugar": "blood glucose",
    "glucose": "blood glucose",
    "oxygen saturation": "spo2",
    "o2 sat": "spo2",
    "saturation": "spo2",
    "heart rate": "pulse",
    "hr": "pulse",
    "pulse": "pulse",
}

_MEASUREMENT_TYPES = {FactType.measurement, FactType.vital_sign}


def canonical_label(fact: Fact) -> str:
    label = (fact.label or "").strip().lower()
    return _CANONICAL.get(label, label)


def _numbers(fact: Fact) -> List[float]:
    value = fact.value if fact.value is not None else fact.value_normalized
    if value is None:
        return []
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value))]


def _dedupe_refs(refs: Sequence[SourceRef]) -> List[SourceRef]:
    seen = set()
    result = []
    for ref in refs:
        key = (ref.source_id, ref.segment_id, ref.start_char, ref.end_char)
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result


def _make(
    case_id: str,
    category: DiscrepancyCategory,
    summary: str,
    explanation: str,
    facts: Sequence[Fact],
    extra_refs: Sequence[SourceRef] = (),
    severity: Severity = Severity.review,
) -> Discrepancy:
    fact_ids = sorted(f.fact_id for f in facts)
    refs = _dedupe_refs([ref for f in facts for ref in f.source_refs] + list(extra_refs))
    discrepancy_id = "disc_" + sha256_text(
        case_id + category.value + "|".join(fact_ids) + summary
    )[:20]
    return Discrepancy(
        discrepancy_id=discrepancy_id,
        case_id=case_id,
        category=category,
        severity=severity,
        summary=summary,
        explanation=explanation,
        fact_ids=fact_ids,
        source_refs=refs,
        requires_human_verification=True,
    )


def _presence_conflict(case_id: str, a: Fact, b: Fact) -> Optional[Discrepancy]:
    statuses = {a.status, b.status}
    if FactStatus.uncertain in statuses:
        return _make(
            case_id,
            DiscrepancyCategory.ambiguous_interpretation,
            f"Uncertain interpretation for '{canonical_label(a)}'",
            "At least one claim is marked uncertain; human interpretation is required before comparison.",
            [a, b],
        )
    if not a.source_refs or not b.source_refs:
        return _make(
            case_id,
            DiscrepancyCategory.unable_to_compare,
            f"Cannot compare '{canonical_label(a)}': missing source reference",
            "One of the claims has no locatable source reference, so it cannot be verified against the other.",
            [a, b],
        )

    present = [f for f in (a, b) if f.status == FactStatus.present]
    absent = [f for f in (a, b) if f.status == FactStatus.absent]
    historical = [f for f in (a, b) if f.status == FactStatus.historical]

    if present and absent:
        return _make(
            case_id,
            DiscrepancyCategory.possible_direct_contradiction,
            f"Conflicting reports for '{canonical_label(a)}'",
            f"One source reports '{canonical_label(a)}' as present while another explicitly denies it. "
            "Both original claims are preserved; do not assume either is correct.",
            [a, b],
            severity=Severity.high,
        )
    if present and historical:
        return _make(
            case_id,
            DiscrepancyCategory.different_reference_time,
            f"Historical versus current report for '{canonical_label(a)}'",
            "One source documents a past/historical occurrence and another a current occurrence; "
            "this may be legitimate and requires timing verification.",
            [a, b],
        )
    if absent and historical:
        return _make(
            case_id,
            DiscrepancyCategory.different_reference_time,
            f"Denial versus historical report for '{canonical_label(a)}'",
            "A current explicit denial coexists with a documented historical occurrence; timing should be confirmed.",
            [a, b],
        )
    return None


def _measurement_conflict(case_id: str, a: Fact, b: Fact) -> Optional[Discrepancy]:
    label = canonical_label(a)
    if FactStatus.uncertain in {a.status, b.status}:
        return _make(
            case_id,
            DiscrepancyCategory.ambiguous_interpretation,
            f"Uncertain measurement interpretation for '{label}'",
            "At least one measurement is uncertain; values are preserved without selecting one.",
            [a, b],
        )
    unit_a = (a.unit or "").strip().lower() or None
    unit_b = (b.unit or "").strip().lower() or None
    if unit_a and unit_b and unit_a != unit_b:
        return _make(
            case_id,
            DiscrepancyCategory.unit_mismatch,
            f"Unit mismatch for '{label}'",
            f"Values were reported with different units ({a.unit} vs {b.unit}); no conversion was applied.",
            [a, b],
            severity=Severity.high,
        )

    values_a, values_b = _numbers(a), _numbers(b)
    if not values_a or not values_b:
        return _make(
            case_id,
            DiscrepancyCategory.unable_to_compare,
            f"Cannot compare numerical value for '{label}'",
            "A numeric value or unit was missing, so the two claims could not be compared.",
            [a, b],
        )

    differing = len(values_a) != len(values_b) or any(
        abs(x - y) > max(1.0, 0.05 * max(abs(x), abs(y)))
        for x, y in zip(values_a, values_b)
    )
    if not differing:
        return None

    if a.status == FactStatus.historical or b.status == FactStatus.historical:
        return _make(
            case_id,
            DiscrepancyCategory.different_reference_time,
            f"Historical versus current value for '{label}'",
            f"A historical value ({a.value} {a.unit or ''}) differs from a current value "
            f"({b.value} {b.unit or ''}); this may be legitimate over time.",
            [a, b],
        )

    offsets = [
        f.temporal.offset_hours
        for f in (a, b)
        if f.temporal and f.temporal.offset_hours is not None
    ]
    if len(offsets) == 2 and abs(offsets[0] - offsets[1]) > 6:
        return _make(
            case_id,
            DiscrepancyCategory.different_reference_time,
            f"Values for '{label}' refer to different times",
            f"Recorded values differ and are associated with different time references "
            f"({offsets[0]:g}h vs {offsets[1]:g}h before receipt).",
            [a, b],
        )

    if a.assertion != b.assertion:
        return _make(
            case_id,
            DiscrepancyCategory.possible_direct_contradiction,
            f"Patient-reported and document values differ for '{label}'",
            f"Patient-reported value ({a.value} {a.unit or ''}) differs from the value documented in a report "
            f"({b.value} {b.unit or ''}). Both are preserved for verification.",
            [a, b],
        )

    return _make(
        case_id,
        DiscrepancyCategory.possible_direct_contradiction,
        f"Conflicting values for '{label}'",
        f"Two claims report different values ({a.value} {a.unit or ''} vs {b.value} {b.unit or ''}).",
        [a, b],
        severity=Severity.high,
    )


def reconcile(
    case_id: str, facts: Sequence[Fact], sources: Sequence[SourceRecord]
) -> List[Discrepancy]:
    discrepancies: List[Discrepancy] = []
    seen: set[str] = set()

    def add(item: Optional[Discrepancy]):
        if item and item.discrepancy_id not in seen:
            seen.add(item.discrepancy_id)
            discrepancies.append(item)

    groups: Dict[Tuple[str, str], List[Fact]] = {}
    for fact in facts:
        if fact.verification_status.value == "rejected":
            continue
        if fact.status == FactStatus.unknown:
            continue
        key = ("measured" if fact.fact_type in _MEASUREMENT_TYPES else fact.fact_type.value, canonical_label(fact))
        groups.setdefault(key, []).append(fact)

    for (category, _label), group in groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if category != "measured" and a.status == b.status:
                    # Two aligned reports are not a contradiction.  Only
                    # presence/absence or historical/current differences matter.
                    continue
                if category == "measured" and a.status == b.status == FactStatus.absent:
                    continue
                if category == "measured":
                    add(_measurement_conflict(case_id, a, b))
                else:
                    add(_presence_conflict(case_id, a, b))

    # Facts without any source reference cannot be compared with anything.
    for fact in facts:
        if fact.verification_status.value == "rejected":
            continue
        if not fact.source_refs:
            add(
                _make(
                    case_id,
                    DiscrepancyCategory.unable_to_compare,
                    f"No traceable source for '{fact.label}'",
                    "This claim has no locatable source reference and is presented only as unverified AI inference.",
                    [fact],
                )
            )

    # Duplicate content across documents is surfaced, never silently merged.
    by_hash: Dict[str, List[SourceRecord]] = {}
    for source in sources:
        if source.content_hash:
            by_hash.setdefault(source.content_hash, []).append(source)
    for content_hash, group in by_hash.items():
        parents = {s.parent_source_id or s.source_id for s in group}
        if len(group) > 1 and len(parents) > 1:
            refs = [
                SourceRef(
                    source_id=s.source_id,
                    segment_id=s.segment_id,
                    quote=s.original_text[:200],
                    page=s.page,
                    bbox=s.bbox,
                )
                for s in group
            ]
            add(
                _make(
                    case_id,
                    DiscrepancyCategory.potential_duplicate,
                    "Potentially duplicated content across sources",
                    "Multiple sources contain identical normalized text; this may be a duplicate or stale document.",
                    [],
                    extra_refs=refs,
                    severity=Severity.info,
                )
            )

    order = {"high": 0, "review": 1, "info": 2}
    return sorted(discrepancies, key=lambda d: (order.get(d.severity.value, 3), d.category.value))
