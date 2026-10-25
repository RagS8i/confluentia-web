"""Benchmark the configured extraction provider on synthetic, non-clinical text.

This measures engineering properties only: JSON schema validity, latency, whether
returned quotes are grounded in the source, and recall of expected canonical
labels. It does NOT measure clinical accuracy and must not be presented as a
validation of any model or rule set.

Examples:
    python scripts/benchmark_extraction.py --provider mock
    python scripts/benchmark_extraction.py --provider ollama --model qwen3:8b
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings  # noqa: E402
from app.model.provider import ModelError, build_provider  # noqa: E402
from app.schemas import Modality, SourceRecord  # noqa: E402

DATASET = [
    {
        "id": "en_emergency",
        "language": "en",
        "text": "I have had chest pain and shortness of breath since yesterday.",
        "expected_labels": ["chest pain", "shortness of breath"],
    },
    {
        "id": "en_negation",
        "language": "en",
        "text": "I have no chest pain and no shortness of breath. I have a mild headache.",
        "expected_labels": ["chest pain", "shortness of breath", "headache"],
    },
    {
        "id": "en_measurements",
        "language": "en",
        "text": "Temperature 103 F and blood sugar 310 mg/dL. SpO2 is 91%.",
        "expected_labels": ["temperature", "blood glucose", "SpO2"],
    },
    {
        "id": "hi_fever",
        "language": "hi",
        "text": "मुझे कल से बुखार और खांसी है।",
        "expected_labels": ["fever", "cough"],
    },
    {
        "id": "hinglish_uncertain",
        "language": "hinglish",
        "text": "Seene me dard hai shayad, aur saans bhi phool rahi hai.",
        "expected_labels": ["chest pain", "shortness of breath"],
    },
    {
        "id": "en_vomiting",
        "language": "en",
        "text": "I have been vomiting and have abdominal pain for two days.",
        "expected_labels": ["vomiting", "abdominal pain"],
    },
    {
        "id": "en_rash",
        "language": "en",
        "text": "Fever for 3 days with a rash on the chest.",
        "expected_labels": ["fever", "rash"],
    },
    {
        "id": "en_history",
        "language": "en",
        "text": "Blood sugar was 300 mg/dL last week. Now it is 120 mg/dL.",
        "expected_labels": ["blood glucose"],
    },
]


def _source(item: dict) -> SourceRecord:
    return SourceRecord(
        source_id=f"bench:{item['id']}",
        parent_source_id=f"bench:{item['id']}",
        case_id="benchmark_case",
        modality=Modality.stt,
        segment_id="s0",
        original_text=item["text"],
        normalized_text=" ".join(item["text"].split()),
        language=item["language"],
        confidence=0.9,
    )


def run(args) -> dict:
    settings = load_settings().with_overrides(
        llm_provider=args.provider,
        ollama_model=args.model or load_settings().ollama_model,
    )
    provider = build_provider(settings)
    results = []
    latencies = []
    for item in DATASET:
        source = _source(item)
        started = time.perf_counter()
        try:
            extraction = provider.extract_facts([source], settings)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            labels = {fact.label for fact in extraction.facts}
            grounded = all(
                (fact.original_text or "").strip() and fact.original_text.strip() in source.original_text
                for fact in extraction.facts
            )
            expected = set(item["expected_labels"])
            results.append(
                {
                    "id": item["id"],
                    "language": item["language"],
                    "status": "ok",
                    "latency_s": round(elapsed, 3),
                    "labels": sorted(labels),
                    "missing_labels": sorted(expected - labels),
                    "grounded": grounded,
                }
            )
        except ModelError as exc:
            results.append(
                {
                    "id": item["id"],
                    "language": item["language"],
                    "status": "error",
                    "error_code": exc.code,
                }
            )

    ok = [r for r in results if r["status"] == "ok"]
    total = len(results)
    grounded_ok = [r for r in ok if r["grounded"]]
    for r in ok:
        expected = set(r["labels"]) | set(r["missing_labels"])
        r["recall"] = round(len(expected - set(r["missing_labels"])) / max(len(expected), 1), 3)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider.name,
        "model": provider.model,
        "clinical_validation": False,
        "disclaimer": "Engineering-only benchmark on synthetic text; not clinical validation.",
        "cases": total,
        "ok": len(ok),
        "schema_valid_rate": round(len(ok) / total, 3) if total else 0.0,
        "grounded_rate": round(len(grounded_ok) / len(ok), 3) if ok else 0.0,
        "mean_recall": round(statistics.mean(r["recall"] for r in ok), 3) if ok else 0.0,
        "latency_s": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3)
            if latencies
            else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "results": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="mock", choices=["mock", "ollama"])
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the JSON report (default: reports/benchmark-<timestamp>.json)",
    )
    args = parser.parse_args()

    summary = run(args)
    if args.output:
        output = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = PROJECT_ROOT / "reports" / f"benchmark-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    printable = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
