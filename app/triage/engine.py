"""Deterministic, configurable workflow-rule engine.

This component is deliberately independent of the SLM.  It evaluates explicit,
versioned rules over structured facts and returns reproducible workflow
statuses.  It is NOT a validated clinical triage scale and it never concludes
that a patient does not need care.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schemas import (
    Fact,
    FactStatus,
    Outcome,
    SourceRecord,
    SourceRef,
    Warning,
    WarningKind,
)
from ..storage import sha256_text
from ..pipeline.reconcile import canonical_label


class RuleConfigError(RuntimeError):
    pass


@dataclass
class TriageResult:
    outcome: Outcome
    warnings: List[Warning] = field(default_factory=list)
    triggered_rule_ids: List[str] = field(default_factory=list)
    unresolved_rule_ids: List[str] = field(default_factory=list)
    config_id: Optional[str] = None
    config_version: Optional[str] = None
    config_status: Optional[str] = None
    review_deadlines_minutes: Dict[str, int] = field(default_factory=dict)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _numbers(fact: Fact) -> List[float]:
    value = fact.value if fact.value is not None else fact.value_normalized
    if value is None:
        return []
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value))]


def _canonical_set(values: Iterable[str]) -> set[str]:
    return {canonical_label_value(v) for v in values}


def canonical_label_value(value: str) -> str:
    from ..pipeline.reconcile import _CANONICAL

    text = str(value).strip().lower()
    return _CANONICAL.get(text, text)


class RuleEngine:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError) as exc:
            raise RuleConfigError(f"unable to load triage configuration: {exc}") from exc
        for key in ("config_id", "version", "status", "outcomes", "rules"):
            if key not in config:
                raise RuleConfigError(f"triage configuration missing '{key}'")
        if not isinstance(config["rules"], list):
            raise RuleConfigError("triage 'rules' must be a list")
        return config

    @property
    def rank(self) -> Dict[str, int]:
        return {
            "EMERGENCY": 0,
            "PRIORITY_REVIEW": 1,
            "UNASSESSED": 2,
            "ROUTINE_REVIEW": 3,
        }

    def _leaf_match(self, fact: Fact, leaf: Dict[str, Any], strict: bool) -> bool:
        if fact.verification_status.value == "rejected":
            return False
        fact_types = _as_list(leaf.get("fact_type"))
        if fact_types and fact.fact_type.value not in fact_types:
            return False
        statuses = _as_list(leaf.get("status_any"))
        if statuses:
            status_ok = fact.status.value in statuses
            if (
                not status_ok
                and not strict
                and fact.status == FactStatus.uncertain
                and "present" in statuses
            ):
                # An uncertain claim is treated as a *potential* match only in
                # lenient mode, so it can route to human verification.
                status_ok = True
            if not status_ok:
                return False
        labels = leaf.get("canonical_any") or leaf.get("label_any")
        if labels and canonical_label(fact) not in _canonical_set(_as_list(labels)):
            return False
        units = _as_list(leaf.get("unit_any"))
        if units and (fact.unit or "").strip().lower() not in {u.lower() for u in units}:
            return False
        assertions = _as_list(leaf.get("assertion_any"))
        if assertions and fact.assertion.value not in assertions:
            return False

        numbers = _numbers(fact)
        if "value_gte" in leaf and not any(n >= float(leaf["value_gte"]) for n in numbers):
            return False
        if "value_gt" in leaf and not any(n > float(leaf["value_gt"]) for n in numbers):
            return False
        if "value_lte" in leaf and not any(n <= float(leaf["value_lte"]) for n in numbers):
            return False
        if "value_lt" in leaf and not any(n < float(leaf["value_lt"]) for n in numbers):
            return False
        if "systolic_gte" in leaf:
            if len(numbers) < 1 or numbers[0] < float(leaf["systolic_gte"]):
                return False
        if "diastolic_gte" in leaf:
            if len(numbers) < 2 or numbers[1] < float(leaf["diastolic_gte"]):
                return False

        if strict:
            if fact.verification_status.value == "rejected":
                return False
            if not fact.source_refs:
                return False
            if fact.status == FactStatus.uncertain:
                return False
            minimum = float(leaf.get("min_confidence", 0.0))
            if fact.confidence is not None and fact.confidence < minimum:
                return False
            if leaf.get("verified_only") and fact.verification_status.value != "verified":
                return False
        return True

    def _eval_node(
        self, node: Dict[str, Any], facts: Sequence[Fact], strict: bool
    ) -> Tuple[bool, List[Fact]]:
        if not isinstance(node, dict) or not node:
            return False, []
        if "all" in node:
            matched: List[Fact] = []
            for child in node["all"]:
                ok, child_facts = self._eval_node(child, facts, strict)
                if not ok:
                    return False, []
                matched.extend(child_facts)
            return True, self._unique_facts(matched)
        if "any" in node:
            matched = []
            for child in node["any"]:
                ok, child_facts = self._eval_node(child, facts, strict)
                if ok:
                    matched.extend(child_facts)
            return bool(matched), self._unique_facts(matched)
        if "none" in node:
            for child in node["none"]:
                ok, _ = self._eval_node(child, facts, strict)
                if ok:
                    return False, []
            return True, []
        matched = [fact for fact in facts if self._leaf_match(fact, node, strict)]
        return bool(matched), self._unique_facts(matched)

    @staticmethod
    def _unique_facts(facts: Sequence[Fact]) -> List[Fact]:
        seen = set()
        result = []
        for fact in facts:
            if fact.fact_id in seen:
                continue
            seen.add(fact.fact_id)
            result.append(fact)
        return result

    def _refs(self, facts: Sequence[Fact]) -> List[SourceRef]:
        seen = set()
        refs = []
        for fact in facts:
            for ref in fact.source_refs:
                key = (ref.source_id, ref.segment_id, ref.start_char, ref.end_char)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(ref)
        return refs

    def _make_warning(
        self,
        case_id: str,
        rule: Dict[str, Any],
        code: str,
        outcome: Outcome,
        label: str,
        explanation: str,
        facts: Sequence[Fact],
    ) -> Warning:
        evidence = sorted(f.fact_id for f in facts)
        warning_id = "warn_" + sha256_text(
            case_id + rule.get("id", "") + code + "|".join(evidence)
        )[:20]
        return Warning(
            warning_id=warning_id,
            case_id=case_id,
            kind=WarningKind.rule,
            code=code,
            label=label,
            explanation=explanation,
            outcome=outcome,
            rule_id=rule.get("id"),
            rule_version=str(rule.get("version") or self.config.get("version")),
            evidence_fact_ids=evidence,
            source_refs=self._refs(facts),
            component="rule_engine",
        )

    def evaluate(
        self,
        case_id: str,
        facts: Sequence[Fact],
        sources: Sequence[SourceRecord],
        component_warnings: Optional[Sequence[Warning]] = None,
        critical_failure: bool = False,
        required_info_missing: bool = False,
    ) -> TriageResult:
        warnings: List[Warning] = list(component_warnings or [])
        triggered: List[str] = []
        unresolved: List[str] = []
        outcomes: List[Outcome] = []

        evaluable = [
            f
            for f in facts
            if f.verification_status.value != "rejected"
            and f.source_refs
            and f.status != FactStatus.unknown
        ]
        present_symptoms = [
            f
            for f in evaluable
            if f.fact_type.value == "symptom"
            and f.status == FactStatus.present
            and f.verification_status.value != "rejected"
        ]

        for rule in self.config["rules"]:
            when = rule.get("when")
            if not when:
                continue
            strict_ok, strict_facts = self._eval_node(when, evaluable, True)
            if strict_ok:
                outcome = Outcome(str(rule.get("outcome")))
                warning = self._make_warning(
                    case_id,
                    rule,
                    f"RULE_{rule.get('id')}",
                    outcome,
                    str(rule.get("label") or rule.get("id")),
                    str(rule.get("explanation") or "Configured demonstration rule matched."),
                    strict_facts,
                )
                warnings.append(warning)
                outcomes.append(outcome)
                triggered.append(str(rule.get("id")))
                continue

            lenient_ok, lenient_facts = self._eval_node(when, facts, False)
            if lenient_ok and lenient_facts:
                warning = self._make_warning(
                    case_id,
                    rule,
                    f"RULE_{rule.get('id')}_UNRESOLVED",
                    Outcome.unassessed,
                    f"Unresolved potential warning: {rule.get('label') or rule.get('id')}",
                    "A configured warning pattern may be present, but at least one matched fact is uncertain, "
                    "low-confidence, or lacks a traceable source. Human assessment is required; this is not a "
                    "confirmed rule trigger.",
                    lenient_facts,
                )
                warnings.append(warning)
                unresolved.append(str(rule.get("id")))

        # Unverified claims cannot silently suppress warnings, but they must be
        # visible to the reviewer.
        untraceable = [f for f in facts if not f.source_refs and f.verification_status.value != "rejected"]
        if untraceable:
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "untraceable")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="UNVERIFIED_AI_CLAIMS",
                    label=f"{len(untraceable)} claim(s) lack a traceable source",
                    explanation="These claims are retained as unverified AI inference and must not be treated as observed facts.",
                    evidence_fact_ids=sorted(f.fact_id for f in untraceable),
                    component="extraction",
                )
            )

        if not evaluable:
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "no_evaluable")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="NO_EVALUABLE_FACTS",
                    label="No evaluable clinical facts",
                    explanation="The rule engine could not evaluate any traceable facts. The case cannot be assigned a routine status.",
                    outcome=Outcome.unassessed,
                    component="rule_engine",
                )
            )
            return self._result(
                Outcome.unassessed, warnings, triggered, unresolved
            )

        if not present_symptoms and not any(o != Outcome.routine_review for o in outcomes):
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "no_complaint")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="NO_PRESENTING_COMPLAINT_EXTRACTED",
                    label="No present symptom extracted",
                    explanation="No present patient-reported symptom could be traced. Missing information is not a negative finding.",
                    outcome=Outcome.unassessed,
                    component="rule_engine",
                )
            )
            return self._result(Outcome.unassessed, warnings, triggered, unresolved)

        if outcomes:
            outcome = min(outcomes, key=lambda o: self.rank[o.value])
            if critical_failure and outcome == Outcome.routine_review:
                outcome = Outcome.unassessed
            return self._result(outcome, warnings, triggered, unresolved)

        if unresolved:
            return self._result(Outcome.unassessed, warnings, triggered, unresolved)

        if critical_failure or required_info_missing:
            warnings.append(
                Warning(
                    warning_id="warn_" + sha256_text(case_id + "incomplete")[:20],
                    case_id=case_id,
                    kind=WarningKind.component,
                    code="PROCESSING_INCOMPLETE",
                    label="Processing incomplete; status cannot be routine",
                    explanation="At least one source or required information stream is unavailable. The case is routed for human assessment instead of a reassuring status.",
                    outcome=Outcome.unassessed,
                    component="pipeline",
                )
            )
            return self._result(Outcome.unassessed, warnings, triggered, unresolved)

        warnings.append(
            Warning(
                warning_id="warn_" + sha256_text(case_id + "routine")[:20],
                case_id=case_id,
                kind=WarningKind.component,
                code="NO_CONFIGURED_WARNING_TRIGGERED",
                label="No configured warning rule triggered",
                explanation="This workflow status is not a clinical clearance. A qualified clinician must review the case.",
                outcome=Outcome.routine_review,
                component="rule_engine",
            )
        )
        return self._result(Outcome.routine_review, warnings, triggered, unresolved)

    def _result(
        self,
        outcome: Outcome,
        warnings: List[Warning],
        triggered: List[str],
        unresolved: List[str],
    ) -> TriageResult:
        return TriageResult(
            outcome=outcome,
            warnings=warnings,
            triggered_rule_ids=triggered,
            unresolved_rule_ids=unresolved,
            config_id=self.config.get("config_id"),
            config_version=str(self.config.get("version")),
            config_status=self.config.get("status"),
            review_deadlines_minutes={
                k: int(v)
                for k, v in (self.config.get("review_deadlines_minutes") or {}).items()
            },
        )
