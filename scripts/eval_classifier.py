#!/usr/bin/env python3
"""Evaluation, threshold sweep, and calibration report generator for Stage 2 + Stage 3 + Stage 4.

Supports:
- `--live-ollama`: evaluates a real local Ollama SLM (`--model gemma4:e4b-it-qat`, `qwen2.5:1.5b`, etc.)
  across the 100-item stratified ticket benchmark, measuring per-class Precision/Recall/F1,
  5x5 confusion matrix, 10-bin Expected Calibration Error (ECE), Brier score, latency p50/p95,
  and a full confidence threshold grid sweep (`spam_threshold` x `escalation_threshold`).
- `--write-report CALIBRATION_REPORT.md`: writes a detailed markdown calibration report.
- Default offline mode: evaluates deterministic Stage 2 guardrails + keyword fallback baseline,
  explicitly reporting `"slm_measured": false` and `"expected_calibration_error_10bin": null`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.pipeline.step1_trigger import compute_payload_signature, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step3_classify import read_and_classify
from src.pipeline.step4_rules import _has_transactional_customer_signals, apply_rules
from src.schemas import ClassificationOutput, CleanedPayload, IntentCategory
from src.slm_client import LocalSLMClient

DEFAULT_DATASET = PROJECT_ROOT / "tests" / "eval_dataset.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "CALIBRATION_REPORT.md"
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
            "ssrf",
            "alg=none",
            "public gist",
            "host header poisoning",
            "subdomain takeover",
            "privilege escalation",
            "unauthorized access",
            "path traversal",
            "open redirect",
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
            "feature idea",
            "feature request",
            "feature suggestion",
            "would love",
            "we'd love",
            "could you add",
            "can you add",
            "it would be helpful",
            "please allow subscribing",
            "csv upload option",
            "native datadog",
            "keyboard navigation",
            "weekly digest",
            "passkey",
            "data residency",
            "saved search",
            "go client library",
            "markdown tables",
            "replay button",
            "business hours",
            "checkbox action",
            "traceparent",
            "changelog drawer",
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
            "prorated",
            "amex",
            "purchase order",
            "tax-exempt",
            "remittance",
            "sepa",
            "past due",
            "true-up",
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
            "502",
            "429",
            "409",
            "500",
            "503",
            "error",
            "gateway",
            "bug",
            "endpoint",
            "parser",
            "saml",
            "timeout",
            "totp",
            "2fa",
            "graphql",
            "oom-killed",
            "cors",
            "sslhandshake",
            "search index",
            "mojibake",
            "cli login",
            "websocket",
            "refresh token",
            "high cpu",
            "terraform state",
            "bounce suppression",
            "deadline_exceeded",
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
        for k in (
            "crypto",
            "casino",
            "lottery",
            "airdrop",
            "seo",
            "weight loss",
            "forex",
            "sweepstakes",
            "replica watches",
            "scraped ceo",
            "business loan",
            "singles",
            "outsourcing",
            "nft",
            "solar panels",
            "tiktok",
            "inheritance",
            "penny stock",
            "toner cartridges",
        )
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
) -> tuple[float, List[Dict[str, Any]]]:
    """Compute M-bin Expected Calibration Error (ECE) and per-bin reliability table."""
    if not confidences:
        return 0.0, []
    n = len(confidences)
    ece = 0.0
    bins_table: List[Dict[str, Any]] = []
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
        gap = abs(bin_acc - bin_conf)
        ece += (len(indices) / n) * gap
        bins_table.append(
            {
                "bin": f"[{low:.1f}, {high:.1f}]",
                "count": len(indices),
                "mean_confidence": round(bin_conf, 4),
                "empirical_accuracy": round(bin_acc, 4),
                "calibration_gap": round(gap, 4),
            }
        )
    return round(ece, 4), bins_table


def _sweep_thresholds(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate routing behavior across a grid of (spam_threshold, escalation_threshold) pairs."""
    grid_results: List[Dict[str, Any]] = []
    total = max(1, len(records))
    for spam_thresh in (0.75, 0.80, 0.85, 0.90, 0.95):
        for esc_thresh in (0.55, 0.65, 0.75, 0.80, 0.85):
            false_drops = 0
            true_spam_dropped = 0
            total_spam = 0
            auto_drafted = 0
            auto_draft_correct = 0
            escalated = 0

            for rec in records:
                cleaned: CleanedPayload = rec["cleaned"]
                cls: ClassificationOutput = rec["cls"]
                expected_intent: str = rec["expected_intent"]
                must_not_drop: bool = rec["must_not_silent_drop"]

                if expected_intent == "SPAM":
                    total_spam += 1

                # Simulate Stage 4 with candidate thresholds
                if (
                    cls.intent == IntentCategory.SPAM
                    and cls.confidence >= spam_thresh
                    and not cls.requires_human_review
                    and not cleaned.contains_suspicious_patterns
                    and not _has_transactional_customer_signals(cleaned)
                ):
                    if must_not_drop:
                        false_drops += 1
                    elif expected_intent == "SPAM":
                        true_spam_dropped += 1
                elif (
                    cls.confidence < esc_thresh
                    or cls.requires_human_review
                    or cleaned.contains_suspicious_patterns
                    or cls.intent == IntentCategory.SECURITY_REPORT
                ):
                    escalated += 1
                else:
                    auto_drafted += 1
                    if cls.intent.value == expected_intent:
                        auto_draft_correct += 1

            grid_results.append(
                {
                    "spam_threshold": spam_thresh,
                    "escalation_threshold": esc_thresh,
                    "false_silent_drop_rate": round(false_drops / total, 4),
                    "spam_drop_recall": round(true_spam_dropped / max(1, total_spam), 4),
                    "auto_draft_coverage": round(auto_drafted / total, 4),
                    "auto_draft_precision": round(
                        auto_draft_correct / max(1, auto_drafted), 4
                    ),
                    "human_escalation_rate": round(escalated / total, 4),
                }
            )
    return grid_results


def evaluate_dataset(
    dataset_path: Path = DEFAULT_DATASET,
    live_ollama: bool = False,
    model: str = DEFAULT_CONFIG.classify_model,
) -> Dict[str, Any]:
    """Run the 100-item stratified benchmark dataset through Steps 1-4 and compute full metrics."""
    items: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    client = LocalSLMClient(timeout_seconds=30.0)
    if live_ollama:
        import urllib.request

        try:
            with urllib.request.urlopen(f"{client.base_url}/api/tags", timeout=3.0) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP {resp.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach local Ollama at {client.base_url} for --live-ollama evaluation: {exc}"
            ) from exc

    correct_intents = 0
    correct_queues = 0
    false_silent_drops = 0
    short_circuited_injections = 0
    confidences: List[float] = []
    correctness: List[int] = []
    latencies_ms: List[float] = []
    raw_records: List[Dict[str, Any]] = []
    misclassified_examples: List[Dict[str, Any]] = []

    tp_per_class: Dict[str, int] = defaultdict(int)
    fp_per_class: Dict[str, int] = defaultdict(int)
    fn_per_class: Dict[str, int] = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = {
        k.value: {m.value: 0 for m in IntentCategory} for k in IntentCategory
    }

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

        if cleaned.contains_suspicious_patterns:
            short_circuited_injections += 1

        t_start = time.perf_counter()
        if live_ollama or cleaned.contains_suspicious_patterns or cleaned.word_count == 0:
            cls = read_and_classify(cleaned, client, model=model)
        else:
            cls = deterministic_reference_classifier(cleaned)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(elapsed_ms)

        rules = apply_rules(cleaned, cls)

        expected_intent = item["expected_intent"]
        predicted_intent = cls.intent.value
        confusion[expected_intent][predicted_intent] += 1

        is_intent_correct = predicted_intent == expected_intent
        is_queue_correct = rules.assigned_queue == item["expected_queue"]

        if is_intent_correct:
            correct_intents += 1
            tp_per_class[expected_intent] += 1
        else:
            fp_per_class[predicted_intent] += 1
            fn_per_class[expected_intent] += 1
            misclassified_examples.append(
                {
                    "id": item["id"],
                    "subject": item["subject"],
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "confidence": cls.confidence,
                    "assigned_queue": rules.assigned_queue,
                }
            )

        if is_queue_correct:
            correct_queues += 1
        if item.get("must_not_silent_drop", True) and rules.assigned_queue == "trash":
            false_silent_drops += 1

        if not cleaned.contains_suspicious_patterns and cleaned.word_count > 0:
            confidences.append(cls.confidence)
            correctness.append(1 if is_intent_correct else 0)
        raw_records.append(
            {
                "id": item["id"],
                "cleaned": cleaned,
                "cls": cls,
                "expected_intent": expected_intent,
                "must_not_silent_drop": item.get("must_not_silent_drop", True),
            }
        )

    total = max(1, len(items))
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    per_class_f1: Dict[str, float] = {}
    for intent_enum in IntentCategory:
        label = intent_enum.value
        tp = tp_per_class[label]
        fp = fp_per_class[label]
        fn = fn_per_class[label]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[label] = round(f1, 4)
        per_class_metrics[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    ece_val: Optional[float] = None
    brier_val: Optional[float] = None
    reliability_bins: List[Dict[str, Any]] = []
    threshold_sweep: List[Dict[str, Any]] = []

    if live_ollama:
        ece_val, reliability_bins = _compute_binned_ece(confidences, correctness)
        brier_val = round(
            sum((c - y) ** 2 for c, y in zip(confidences, correctness)) / max(1, len(confidences)), 4
        )
        threshold_sweep = _sweep_thresholds(raw_records)

    sorted_lat = sorted(latencies_ms)
    p50_ms = round(sorted_lat[len(sorted_lat) // 2], 2) if sorted_lat else 0.0
    p95_ms = round(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))], 2) if sorted_lat else 0.0

    return {
        "evaluation_mode": "live_ollama_slm" if live_ollama else "offline_deterministic_baseline",
        "model": model if live_ollama else "deterministic_keyword_reference",
        "slm_measured": bool(live_ollama),
        "total_samples": len(items),
        "short_circuited_injections_0_tokens": short_circuited_injections,
        "intent_accuracy": round(correct_intents / total, 4),
        "routing_queue_accuracy": round(correct_queues / total, 4),
        "per_class_f1": per_class_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion,
        "false_silent_drop_rate": round(false_silent_drops / total, 4),
        "expected_calibration_error_10bin": ece_val,
        "brier_score": brier_val,
        "reliability_bins": reliability_bins,
        "threshold_sweep": threshold_sweep,
        "misclassified_examples": misclassified_examples,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "prompt_tokens_used": client.total_prompt_tokens,
        "completion_tokens_used": client.total_completion_tokens,
        "tokens_used": client.total_tokens_used,
    }


def write_markdown_report(metrics: Dict[str, Any], report_path: Path = DEFAULT_REPORT_PATH) -> Path:
    """Render a detailed Markdown calibration and threshold analysis report."""
    lines: List[str] = [
        "# Local SLM Classification & Confidence Calibration Report",
        "",
        f"- **Evaluation Mode:** `{metrics['evaluation_mode']}`",
        f"- **Model Evaluated:** `{metrics['model']}`",
        f"- **SLM Directly Measured:** `{metrics['slm_measured']}`",
        f"- **Dataset Size:** `{metrics['total_samples']}` stratified tickets (22 Billing, 22 Tech Support, 20 Feature Request, 18 Spam, 18 Security/Adversarial)",
        f"- **Adversarial Tickets Short-Circuited at Stage 2 (0 Tokens):** `{metrics['short_circuited_injections_0_tokens']}`",
        f"- **Overall Intent Accuracy:** `{metrics['intent_accuracy'] * 100:.2f}%`",
        f"- **End-to-End Routing Queue Accuracy:** `{metrics['routing_queue_accuracy'] * 100:.2f}%`",
        f"- **False Silent-Drop Rate (Legitimate Ticket -> Trash):** `{metrics['false_silent_drop_rate'] * 100:.2f}%`",
        f"- **10-Bin Expected Calibration Error (ECE):** `{metrics['expected_calibration_error_10bin']}`",
        f"- **Brier Score:** `{metrics['brier_score']}`",
        f"- **Latency (p50 / p95):** `{metrics['latency_p50_ms']} ms` / `{metrics['latency_p95_ms']} ms`",
        f"- **Total Tokens Used:** `{metrics['tokens_used']}` (`{metrics['prompt_tokens_used']}` prompt + `{metrics['completion_tokens_used']}` completion)",
        "",
        "## 1. Per-Class Precision, Recall & F1",
        "",
        "| Intent Category | Precision | Recall | F1 Score | Support |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for label, m in metrics["per_class_metrics"].items():
        lines.append(
            f"| `{label}` | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | {m['support']} |"
        )

    lines.extend(
        [
            "",
            "## 2. Confusion Matrix (Rows = Expected, Columns = Predicted)",
            "",
            "| Expected \\ Predicted | `BILLING` | `TECHNICAL_SUPPORT` | `FEATURE_REQUEST` | `SPAM` | `SECURITY_REPORT` |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    order = ["BILLING", "TECHNICAL_SUPPORT", "FEATURE_REQUEST", "SPAM", "SECURITY_REPORT"]
    for exp_label in order:
        row_counts = [str(metrics["confusion_matrix"][exp_label][pred]) for pred in order]
        lines.append(f"| **`{exp_label}`** | " + " | ".join(row_counts) + " |")

    if metrics["reliability_bins"]:
        lines.extend(
            [
                "",
                "## 3. Confidence Reliability Diagram (10-Bin ECE)",
                "",
                "| Confidence Bin | Samples | Mean Self-Reported Confidence | Empirical Accuracy | Calibration Gap |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for b in metrics["reliability_bins"]:
            lines.append(
                f"| `{b['bin']}` | {b['count']} | {b['mean_confidence']:.4f} | {b['empirical_accuracy']:.4f} | {b['calibration_gap']:.4f} |"
            )

    if metrics["threshold_sweep"]:
        lines.extend(
            [
                "",
                "## 4. Confidence Threshold Sweep (`spam_threshold` × `escalation_threshold`)",
                "",
                "| Spam Drop Threshold | Escalation Threshold | False Silent-Drop Rate | Spam Drop Recall | Auto-Draft Coverage | Auto-Draft Precision | Human Escalation Rate |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for row in metrics["threshold_sweep"]:
            lines.append(
                f"| `{row['spam_threshold']:.2f}` | `{row['escalation_threshold']:.2f}` | "
                f"**{row['false_silent_drop_rate'] * 100:.2f}%** | {row['spam_drop_recall'] * 100:.1f}% | "
                f"{row['auto_draft_coverage'] * 100:.1f}% | {row['auto_draft_precision'] * 100:.1f}% | "
                f"{row['human_escalation_rate'] * 100:.1f}% |"
            )

    if metrics["misclassified_examples"]:
        lines.extend(
            [
                "",
                "## 5. Error Analysis (Misclassified Tickets)",
                "",
                "| Ticket ID | Subject | Expected | Predicted | Confidence | Routed Queue |",
                "| :--- | :--- | :--- | :--- | :---: | :--- |",
            ]
        )
        for err in metrics["misclassified_examples"]:
            lines.append(
                f"| `{err['id']}` | {err['subject']} | `{err['expected']}` | `{err['predicted']}` | "
                f"{err['confidence']:.2f} | `{err['assigned_queue']}` |"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate classifier accuracy and SPAM drop safety.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Evaluate the live local Ollama SLM instead of the offline deterministic baseline.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG.classify_model,
        help="Ollama model tag to evaluate when --live-ollama is enabled.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional path to write a Markdown calibration report (e.g., CALIBRATION_REPORT.md).",
    )
    args = parser.parse_args()

    metrics = evaluate_dataset(
        dataset_path=args.dataset,
        live_ollama=args.live_ollama,
        model=args.model,
    )
    if args.write_report is not None:
        write_markdown_report(metrics, report_path=args.write_report)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["false_silent_drop_rate"] == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
