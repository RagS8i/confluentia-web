from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas import (
    Assertion,
    ExtractionMeta,
    Fact,
    FactStatus,
    FactType,
    Outcome,
    SourceRef,
    VerificationStatus,
)
from app.triage.engine import RuleEngine

RULES = Path(__file__).resolve().parent.parent / "app" / "triage" / "rules_demo.json"


def make_fact(
    label: str,
    fact_type: FactType = FactType.symptom,
    *,
    value=None,
    unit: str | None = None,
    status: FactStatus = FactStatus.present,
    confidence: float = 0.9,
    verification: VerificationStatus = VerificationStatus.unverified,
) -> Fact:
    return Fact(
        fact_id=f"fact_{label}_{value}_{unit}",
        case_id="case_rules",
        fact_type=fact_type,
        label=label,
        value=value,
        unit=unit,
        status=status,
        assertion=Assertion.patient_reported,
        confidence=confidence,
        verification_status=verification,
        source_refs=[SourceRef(source_id="src_rules", segment_id="0")],
        extraction=ExtractionMeta(method="test"),
    )


CASES = [
    ("DEMO-EMERGENCY-01", Outcome.emergency, [make_fact("chest pain"), make_fact("shortness of breath")]),
    ("DEMO-EMERGENCY-02", Outcome.emergency, [make_fact("SpO2", FactType.vital_sign, value="88", unit="%")]),
    ("DEMO-EMERGENCY-03", Outcome.emergency, [make_fact("loss of consciousness")]),
    ("DEMO-EMERGENCY-04", Outcome.emergency, [make_fact("seizure")]),
    ("DEMO-EMERGENCY-05", Outcome.emergency, [make_fact("bleeding"), make_fact("dizziness")]),
    ("DEMO-PRIORITY-01", Outcome.priority_review, [make_fact("chest pain")]),
    ("DEMO-PRIORITY-02", Outcome.priority_review, [make_fact("shortness of breath")]),
    (
        "DEMO-PRIORITY-03",
        Outcome.priority_review,
        [make_fact("temperature", FactType.vital_sign, value="39.5", unit="C")],
    ),
    (
        "DEMO-PRIORITY-04",
        Outcome.priority_review,
        [make_fact("temperature", FactType.vital_sign, value="103", unit="F")],
    ),
    (
        "DEMO-PRIORITY-05",
        Outcome.priority_review,
        [make_fact("blood pressure", FactType.vital_sign, value="190/120", unit="mmHg")],
    ),
    (
        "DEMO-PRIORITY-06",
        Outcome.priority_review,
        [make_fact("blood pressure", FactType.vital_sign, value="160/125", unit="mmHg")],
    ),
    (
        "DEMO-PRIORITY-07",
        Outcome.priority_review,
        [make_fact("blood glucose", FactType.measurement, value="320", unit="mg/dL")],
    ),
    (
        "DEMO-PRIORITY-08",
        Outcome.priority_review,
        [make_fact("vomiting"), make_fact("abdominal pain")],
    ),
    ("DEMO-PRIORITY-09", Outcome.priority_review, [make_fact("fever"), make_fact("rash")]),
]


def test_config_metadata_and_all_rule_shapes_are_present():
    engine = RuleEngine(RULES)
    config = engine.config
    assert config["config_id"] == "careflow-demo-triage"
    assert config["status"] == "ILLUSTRATIVE_DEMO_ONLY_NOT_CLINICALLY_APPROVED"
    assert config["clinical_review_required"] is True
    assert set(config["review_deadlines_minutes"]) == {
        "EMERGENCY",
        "PRIORITY_REVIEW",
        "ROUTINE_REVIEW",
        "UNASSESSED",
    }
    ids = {rule["id"] for rule in config["rules"]}
    assert ids == {case[0] for case in CASES}
    for rule in config["rules"]:
        assert rule["version"]
        assert rule["when"]
        assert rule["outcome"] in config["outcomes"]
        assert rule["explanation"]
        assert "illustrative demo" in rule["explanation"].lower()


@pytest.mark.parametrize("rule_id,expected_outcome,facts", CASES)
def test_each_demo_rule_is_reachable(rule_id, expected_outcome, facts):
    engine = RuleEngine(RULES)
    result = engine.evaluate("case_rules", facts, [])
    assert rule_id in result.triggered_rule_ids
    assert expected_outcome in {
        warning.outcome for warning in result.warnings if warning.rule_id == rule_id
    }
    assert result.outcome == expected_outcome
    assert result.config_status == "ILLUSTRATIVE_DEMO_ONLY_NOT_CLINICALLY_APPROVED"
    assert result.review_deadlines_minutes["EMERGENCY"] == 0


def test_rejected_fact_cannot_trigger_or_block_a_rule():
    engine = RuleEngine(RULES)
    fact = make_fact(
        "chest pain",
        verification=VerificationStatus.rejected,
    )
    result = engine.evaluate("case_rules", [fact], [])
    assert "DEMO-PRIORITY-01" not in result.triggered_rule_ids
    assert "DEMO-PRIORITY-01" not in result.unresolved_rule_ids
    assert result.outcome == Outcome.unassessed


def test_absent_symptom_does_not_trigger_priority_rule():
    engine = RuleEngine(RULES)
    fact = make_fact("chest pain", status=FactStatus.absent)
    result = engine.evaluate("case_rules", [fact], [])
    assert "DEMO-PRIORITY-01" not in result.triggered_rule_ids
    assert result.outcome == Outcome.unassessed
