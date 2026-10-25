"""Validate a CareFlow rule configuration before deployment.

Checks structure and governance fields only. Passing this script does NOT mean
the thresholds are clinically correct; a qualified clinician must review and
approve the rule set separately.

Usage:
    python scripts/validate_rules.py
    python scripts/validate_rules.py --rules app/triage/rules_demo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.triage.engine import RuleConfigError, RuleEngine  # noqa: E402


def validate(path: Path) -> dict:
    engine = RuleEngine(path)
    config = engine.config
    problems = []
    ids = []
    for rule in config["rules"]:
        rule_id = rule.get("id")
        ids.append(rule_id)
        if not rule_id:
            problems.append("rule missing id")
        if not rule.get("when"):
            problems.append(f"{rule_id}: missing 'when'")
        if rule.get("outcome") not in config["outcomes"]:
            problems.append(f"{rule_id}: unknown outcome {rule.get('outcome')}")
        if not rule.get("explanation"):
            problems.append(f"{rule_id}: missing explanation")
        leaf = json.dumps(rule.get("when"))
        for banned in ("final_status", "diagnosis_result", "prescribe"):
            if banned in leaf:
                problems.append(f"{rule_id}: condition contains unsupported key '{banned}'")
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate rule ids: {duplicates}")
    for outcome in config["outcomes"]:
        if outcome not in (config.get("review_deadlines_minutes") or {}):
            problems.append(f"missing review deadline for {outcome}")
    if not config.get("clinical_review_required", False):
        problems.append("clinical_review_required should be true for demo rules")
    return {
        "config_id": config["config_id"],
        "version": str(config["version"]),
        "status": config["status"],
        "rule_count": len(config["rules"]),
        "rule_ids": sorted(i for i in ids if i),
        "ok": not problems,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default=str(PROJECT_ROOT / "app" / "triage" / "rules_demo.json"))
    args = parser.parse_args()
    try:
        report = validate(Path(args.rules))
    except RuleConfigError as exc:
        print(json.dumps({"ok": False, "problems": [str(exc)]}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
