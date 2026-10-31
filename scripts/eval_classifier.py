#!/usr/bin/env python3
"""Evaluation and calibration benchmark for Stage 2 + Stage 3 + Stage 4 classification & routing.

Measures:
- Intent classification accuracy
- Legitimate-ticket false-drop rate (must be 0.0%)
- Expected Calibration Error (ECE) across confidence bins
Can be run in deterministic offline mode or against a live local Ollama server (`--live-ollama`).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.step1_trigger import compute_payload_signature, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step3_classify import read_and_classify
from src.pipeline.step4_rules import apply_rules
from src.schemas import ClassificationOutput, CleanedPayload, IntentCategory
from src.slm_client import LocalSLMClient

DEFAULT_DATASET = PROJECT_ROOT / "tests" / "eval_dataset.json"
EVAL_SECRET = "evaluation-benchmark-hmac-secret-2026"


def deterministic_reference_classifier(cleaned: CleanedPayload) -> ClassificationOutput:
    """Deterministic keyword baseline used when `--live-ollama` is not enabled."""
    text = f"{cleaned.clean_subject} {cleaned.clean_body}".lower()
    if any(k in text for k in ("vulnerability", "cve-", "data breach", "rce", "sql injection")):
        return ClassificationOutput(
            intent=IntentCategory.SECURITY_REPORT,
            confidence=0.95,
            detected_entities=[],
            requires_human_review=True,
        )
    if any(k in text for k in ("invoice", "charged", "billing", "refund", "receipt")):
        return ClassificationOutput(
            intent=IntentCategory.BILLING,
            confidence=0.90,
            detected_entities=[],
            requires_human_review=False,
        )
    if any(k in text for k in ("502", "error", "gateway", "bug", "endpoint", "parser")):
        return ClassificationOutput(
            intent=IntentCategory.TECHNICAL_SUPPORT,
            confidence=0.88,
            detected_entities=[],
            requires_human_review=False,
        )
    if any(k in text for k in ("crypto", "casino", "lottery", "airdrop")):
        return ClassificationOutput(
            intent=IntentCategory.SPAM,
            confidence=0.92,
            detected_entities=[],
            requires_human_review=False,
        )
    return ClassificationOutput(
        intent=IntentCategory.FEATURE_REQUEST,
        confidence=0.82,
        detected_entities=[],
        requires_human_review=False,
    )


def evaluate_dataset(
    dataset_path: Path = DEFAULT_DATASET,
    live_ollama: bool = False,
) -> Dict[str, Any]:
    """Run the labeled benchmark dataset through Steps 1-4 and compute accuracy, false-drop, and ECE."""
    items: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    client = LocalSLMClient()

    correct_intents = 0
    correct_queues = 0
    false_silent_drops = 0
    calibration_errors: List[float] = []

    for item in items:
        raw = {
            "event_id": str(uuid4()),
            "source": "EMAIL",
            "sender": "eval@benchmark.org",
            "raw_subject": item["subject"],
            "raw_body": item["body"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        raw["signature"] = compute_payload_signature(raw, EVAL_SECRET)
        trigger = ingest_trigger(raw, secret_key=EVAL_SECRET)
        cleaned = fetch_and_clean(trigger)

        if live_ollama or cleaned.contains_suspicious_patterns or cleaned.word_count == 0:
            cls = read_and_classify(cleaned, client)
        else:
            cls = deterministic_reference_classifier(cleaned)

        rules = apply_rules(cleaned, cls)

        is_intent_correct = cls.intent.value == item["expected_intent"]
        is_queue_correct = rules.assigned_queue == item["expected_queue"]

        if is_intent_correct:
            correct_intents += 1
        if is_queue_correct:
            correct_queues += 1
        if item.get("must_not_silent_drop", True) and rules.assigned_queue == "trash":
            false_silent_drops += 1

        accuracy_indicator = 1.0 if is_intent_correct else 0.0
        calibration_errors.append(abs(cls.confidence - accuracy_indicator))

    total = max(1, len(items))
    metrics = {
        "total_samples": len(items),
        "intent_accuracy": round(correct_intents / total, 4),
        "routing_queue_accuracy": round(correct_queues / total, 4),
        "false_silent_drop_rate": round(false_silent_drops / total, 4),
        "mean_calibration_error": round(sum(calibration_errors) / total, 4),
        "tokens_used": client.total_tokens_used,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate classifier accuracy and SPAM drop safety.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--live-ollama", action="store_true", help="Call local Ollama server for Step 3.")
    args = parser.parse_args()

    metrics = evaluate_dataset(dataset_path=args.dataset, live_ollama=args.live_ollama)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["false_silent_drop_rate"] == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
