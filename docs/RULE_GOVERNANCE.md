# Rule governance

The rule configuration is the only component allowed to assign a workflow outcome. It is deterministic, versioned and independent of the language model.

## Current demo status

`app/triage/rules_demo.json` is marked:

```
ILLUSTRATIVE_DEMO_ONLY_NOT_CLINICALLY_APPROVED
```

The thresholds (`SpO2 < 94`, temperature, blood pressure, glucose, etc.) are demonstration values. They are **not** an established triage protocol and have not been reviewed or validated for clinical use. Do not describe them as medically correct.

## Config shape

```json
{
  "config_id": "careflow-demo-triage",
  "version": "0.1.0-demo",
  "status": "ILLUSTRATIVE_DEMO_ONLY_NOT_CLINICALLY_APPROVED",
  "clinical_review_required": true,
  "review_deadlines_minutes": {
    "EMERGENCY": 0,
    "PRIORITY_REVIEW": 30,
    "ROUTINE_REVIEW": 240,
    "UNASSESSED": 60
  },
  "outcomes": { "...": { "rank": 0, "label": "...", "is_clinical_diagnosis": false } },
  "rules": [
    {
      "id": "DEMO-EMERGENCY-01",
      "version": "0.1.0-demo",
      "outcome": "EMERGENCY",
      "label": "Chest pain with breathlessness",
      "minimum_confidence": 0.6,
      "when": {
        "all": [
          {
            "fact_type": "symptom",
            "canonical_any": ["chest pain"],
            "status_any": ["present"],
            "min_confidence": 0.6
          }
        ]
      },
      "explanation": "Illustrative demo rule: ... This is a workflow prompt, not a diagnosis."
    }
  ]
}
```

Conditions support `all`, `any`, `none`, and leaves with:

- `fact_type`, `canonical_any`/`label_any`, `status_any`, `unit_any`, `assertion_any`
- numeric comparisons: `value_gte`, `value_gt`, `value_lte`, `value_lt`, `systolic_gte`, `diastolic_gte`
- `min_confidence`, `verified_only`

## Evaluation semantics

1. Strict match: present status, traceable source, not rejected, not uncertain, confidence at or above threshold.
2. Lenient match: uncertain facts can count as potential matches; used only to emit an **unresolved** warning.
3. Outcome aggregation: lowest rank wins (`EMERGENCY` 0, `PRIORITY_REVIEW` 1, `UNASSESSED` 2, `ROUTINE_REVIEW` 3), except critical failures cannot produce `ROUTINE_REVIEW`.
4. No evaluable facts -> `UNASSESSED`.
5. No present symptom and no non-routine rule -> `UNASSESSED`, not routine.
6. Unresolved potential warnings -> `UNASSESSED` if no stronger outcome applies.
7. `ROUTINE_REVIEW` is emitted only after a configured rule set is evaluated and no warning matches; it is still a review queue, not clearance.

## Change control

Before adopting any new rule set:

1. Clinician authors/approves thresholds and wording, including intended population and exclusions.
2. Add/extend tests with positive, negative, missing-data, negation, historical and uncertainty cases.
3. Run `python scripts/validate_rules.py` and `python -m pytest -q`.
4. Record config ID, version, approval owner/date and change rationale.
5. Deploy with a new version; never edit an approved version in place.
6. Monitor override rates, unresolved warnings, missed-warning reports and review delays.
7. Maintain a rollback path.

`tests/test_rules.py` includes reachability tests for every bundled rule plus rejection/absence cases. Passing tests show engineering behavior, not clinical validity.
