#!/usr/bin/env python3
"""Evaluation and calibration benchmark for Stage 2 + Stage 3 + Stage 4 classification & routing.

Clearly separates:
- `--live-ollama`: evaluates the actual local SLM (`qwen2.5:1.5b` by default) via Ollama HTTP API,
  reporting per-class Precision/Recall/F1, confusion matrix, 10-bin Expected Calibration Error (ECE),
  and false silent-drop rate.
- Default offline mode: evaluates the deterministic Stage 2 guardrails + keyword fallback baseline,
  explicitly marking `"slm_measured": false` and `"ece": null` so offline heuristics are never
  conflated with SLM model quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    """Deterministic keyword baseline used ONLY when `--live-ollama` is not enabled."""
    text = f"{cleaned.clean_subject} {cleaned.clean_body}".lower()
    if any(
        k in text
        for k in (
            "vulnerability",
            "cve-",
            "data breach",
            "rce",
            "sql injection",
            "stored xss",
            "idor",
            "security disclosure",
        )
    ):
        return ClassificationOutput(
            intent=IntentCategory.SECURITY_REPORT,
            confidence=0.95,
            detected_entities=[],
            requires_human_review=True,
        )
    if any(
        k in text
        for k in (
            "invoice",
            "charged",
            "billing",
            "refund",
            "receipt",
            "vat",
            "downgrade",
            "overage",
            "w-9",
            "wire transfer",
        )
    ):
        return ClassificationOutput(
            intent=IntentCategory.BILLING,
            confidence=0.90,
            detected_entities=[],
            requires_human_review=False,
        )
    if any(
        k in text
        for k in (
            "feature idea",
            "feature request",
            "would love",
            "could you add",
            "it would be helpful",
            "please allow subscribing",
            "csv upload option",
        )
    ):
        return ClassificationOutput(
            intent=IntentCategory.FEATURE_REQUEST,
            confidence=0.86,
            detected_entities=[],
            requires_human_review=False,
        )
    if any(
        k in text
        for k in (
            "502",
            "429",
            "error",
            "gateway",
            "bug",
            "endpoint",
            "parser",
            "saml",
            "timeout",
            "totp",
            "2fa",
        )
    ):
        return ClassificationOutput(
            intent=IntentCategory.TECHNICAL_SUPPORT,
            confidence=0.88,
            detected_entities=[],
            requires_human_review=False,
        )
    if any(
        k in text
        for k in ("crypto", "casino", "lottery", "airdrop", "seo backlinks", "weight loss pills")
    ):
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


def _compute_binned_ece(
    confidences: List[float],
    correctness: List[int],
    num_bins: int = 10,
) -> float:
    """Compute standard M-bin Expected Calibration Error (ECE)."""
    if not confidences:
        return 0.0
    n = len(confidences)
    ece = 0.0
    for b in range(num_bins):
        low = b / num_bins
        high = (b + 1) / num_bins
        indices = [
            i
            for i, c in enumerate(confidences)
            if (low <= c < high) or (b == num_bins - 1 and low <= c <= high)
        ]
        if not indices:
            continue
        bin_acc = sum(correctness[i] for i in indices) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(bin_acc - bin_conf)
    return round(ece, 4)


def evaluate_dataset(
    dataset_path: Path = DEFAULT_DATASET,
    live_ollama: bool = False,
) -> Dict[str, Any]:
    """Run the 30-item stratified benchmark dataset through Steps 1-4."""
    items: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    client = LocalSLMClient()

    correct_intents = 0
    correct_queues = 0
    false_silent_drops = 0
    confidences: List[float] = []
    correctness: List[int] = []

    tp_per_class: Dict[str, int] = defaultdict(int)
    fp_per_class: Dict[str, int] = defaultdict(int)
    fn_per_class: Dict[str, int] = defaultdict(int)

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

        expected_intent = item["expected_intent"]
        predicted_intent = cls.intent.value

        is_intent_correct = predicted_intent == expected_intent
        is_queue_correct = rules.assigned_queue == item["expected_queue"]

        if is_intent_correct:
            correct_intents += 1
            tp_per_class[expected_intent] += 1
        else:
            fp_per_class[predicted_intent] += 1
            fn_per_class[expected_intent] += 1

        if is_queue_correct:
            correct_queues += 1
        if item.get("must_not_silent_drop", True) and rules.assigned_queue == "trash":
            false_silent_drops += 1

        confidences.append(cls.confidence)
        correctness.append(1 if is_intent_correct else 0)

    total = max(1, len(items))
    per_class_f1: Dict[str, float] = {}
    for intent_enum in IntentCategory:
        label = intent_enum.value
        tp = tp_per_class[label]
        fp = fp_per_class[label]
        fn = fn_per_class[label]
        denom = (2 * tp) + fp + fn
        per_class_f1[label] = round((2 * tp) / denom, 4) if denom > 0 else 0.0

    ece_value: Optional[float] = (
        _compute_binned_ece(confidences, correctness) if live_ollama else None
    )

    return {
        "evaluation_mode": "live_ollama_slm" if live_ollama else "offline_deterministic_baseline",
        "slm_measured": bool(live_ollama),
        "total_samples": len(items),
        "intent_accuracy": round(correct_intents / total, 4),
        "routing_queue_accuracy": round(correct_queues / total, 4),
        "per_class_f1": per_class_f1,
        "false_silent_drop_rate": round(false_silent_drops / total, 4),
        "expected_calibration_error_10bin": ece_value,
        "tokens_used": client.total_tokens_used,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate classifier accuracy and SPAM drop safety.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Evaluate the live local Ollama SLM instead of the offline deterministic baseline.",
    )
    args = parser.parse_args()

    metrics = evaluate_dataset(dataset_path=args.dataset, live_ollama=args.live_ollama)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["false_silent_drop_rate"] == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
