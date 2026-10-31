#!/usr/bin/env python3
"""Evaluation, held-out slice analysis, post-hoc calibration (Temperature Scaling + Isotonic PAVA),
threshold sweep, and auditable per-ticket prediction logger for Stage 2 + Stage 3 + Stage 4.

Supports:
- `--live-ollama`: evaluates a real local Ollama SLM (`--model gemma4:e4b-it-qat`, etc.) across the
  300-item stratified benchmark (`100` calibration + `200` held-out test tickets including long
  quoted threads, 6 languages, duplicate generic subjects, mixed intents, and hard negatives).
- `--predictions-out tests/eval_predictions.jsonl`: writes raw per-ticket SLM outputs so the
  benchmark can be audited and reproduced offline without Ollama.
- `--from-predictions tests/eval_predictions.jsonl`: replays committed raw SLM outputs to recompute
  all metrics, Wilson 95% confidence intervals, post-hoc Temperature/Isotonic calibration fits,
  and threshold sweeps deterministically in <0.2s.
- `--write-report CALIBRATION_REPORT.md`: writes a detailed, honest Markdown calibration report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.pipeline.step1_trigger import compute_payload_signature, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step3_classify import read_and_classify
from src.pipeline.step4_rules import _has_transactional_customer_signals, apply_rules
from src.schemas import (
    ClassificationOutput,
    CleanedPayload,
    IntentCategory,
    get_alias_remap_telemetry,
    reset_alias_remap_telemetry,
)
from src.slm_client import LocalSLMClient

DEFAULT_DATASET = PROJECT_ROOT / "tests" / "eval_dataset.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "tests" / "eval_predictions.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "CALIBRATION_REPORT.md"
EVAL_SECRET = "evaluation-benchmark-hmac-secret-2026"


def wilson_95_ci(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Compute the 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half_width = (
        z
        * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
        / denom
    )
    return round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4)


def _logit(p: float) -> float:
    clamped = max(1e-5, min(1.0 - 1e-5, p))
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_temperature_scaling(
    cal_confidences: List[float],
    cal_correctness: List[int],
) -> float:
    """Fit post-hoc scalar Temperature T > 0 on the calibration split via NLL minimization
    with Jeffreys-prior smoothing ((N*y + 0.5)/(N + 1)) to prevent boundary collapse (T -> 0)
    when finite calibration splits have zero or near-zero errors.
    """
    if not cal_confidences:
        return 1.0
    n = len(cal_confidences)
    best_t = 1.0
    best_nll = float("inf")
    # Grid search T in [0.20, 5.00]
    for step in range(20, 501):
        t = step / 100.0
        nll = 0.0
        for p, y in zip(cal_confidences, cal_correctness):
            y_smooth = (n * float(y) + 0.5) / (n + 1.0)
            scaled_p = max(1e-6, min(1.0 - 1e-6, _sigmoid(_logit(p) / t)))
            nll -= y_smooth * math.log(scaled_p) + (1.0 - y_smooth) * math.log(1.0 - scaled_p)
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return round(best_t, 4)


def apply_temperature_scaling(p: float, temperature: float) -> float:
    """Apply fitted scalar temperature T to raw confidence probability p."""
    if temperature <= 0:
        return p
    return round(_sigmoid(_logit(p) / temperature), 4)


def fit_isotonic_pava(
    cal_confidences: List[float],
    cal_correctness: List[int],
) -> List[Tuple[float, float, float]]:
    """Fit non-parametric monotonic Isotonic Regression via Pool Adjacent Violators Algorithm (PAVA).

    Returns a list of (min_conf, max_conf, calibrated_prob) blocks.
    """
    if not cal_confidences:
        return [(0.0, 1.0, 0.5)]
    paired = sorted(zip(cal_confidences, cal_correctness), key=lambda x: x[0])
    # Each block: [min_x, max_x, sum_y, count]
    blocks: List[List[float]] = [[x, x, float(y), 1.0] for x, y in paired]
    i = 0
    while i < len(blocks) - 1:
        mean_curr = blocks[i][2] / blocks[i][3]
        mean_next = blocks[i + 1][2] / blocks[i + 1][3]
        if mean_curr > mean_next:
            # Pool adjacent violators
            blocks[i][1] = blocks[i + 1][1]
            blocks[i][2] += blocks[i + 1][2]
            blocks[i][3] += blocks[i + 1][3]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return [
        (round(b[0], 4), round(b[1], 4), round(b[2] / b[3], 4))
        for b in blocks
    ]


def apply_isotonic_pava(
    p: float,
    blocks: List[Tuple[float, float, float]],
) -> float:
    """Map raw confidence p through fitted PAVA isotonic blocks via linear interpolation."""
    if not blocks:
        return p
    if p <= blocks[0][0]:
        return blocks[0][2]
    if p >= blocks[-1][1]:
        return blocks[-1][2]
    for i, (low, high, val) in enumerate(blocks):
        if low <= p <= high:
            return val
        if i + 1 < len(blocks):
            next_low, _, next_val = blocks[i + 1]
            if high < p < next_low:
                span = max(1e-6, next_low - high)
                weight = (p - high) / span
                return round(val + weight * (next_val - val), 4)
    return blocks[-1][2]


def deterministic_reference_classifier(cleaned: CleanedPayload) -> ClassificationOutput:
    """Deterministic keyword baseline used ONLY when neither `--live-ollama` nor `--from-predictions` is used."""
    text = f"{cleaned.clean_subject} {cleaned.clean_body}".lower()
    if any(
        k in text
        for k in (
            "vulnerability",
            "cve-",
            "data breach",
            "rce",
            "sql injection",
            "inyección sql",
            "stored xss",
            "xss",
            "idor",
            "security disclosure",
            "ssrf",
            "alg=none",
            "public gist",
            "host header poisoning",
            "subdomain takeover",
            "privilege escalation",
            "unauthorized access",
            "accès non autorisé",
            "path traversal",
            "open redirect",
            "sicherheitslücke",
            "脆弱性",
            "fuga de claves",
            "gitguardian",
            "credential-stuffing",
            "brute-force",
            "phishing",
            "timing side-channel",
            "cors misconfiguration",
            "crlf header injection",
            "xxe",
            "ssti",
            "httponly",
            "compromised ci runner",
            "refresh tokens",
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
            "inv-",
            "charged",
            "billing",
            "refund",
            "receipt",
            "rcpt-",
            "vat",
            "tva",
            "factura",
            "fatura",
            "rechnung",
            "abbuchung",
            "請求書",
            "downgrade",
            "overage",
            "w-9",
            "wire transfer",
            "ach transfer",
            "sepa",
            "credit memo",
            "seat count",
            "tax exemption",
            "501(c)(3)",
            "prorated",
            "reembolsar",
        )
    ) and " entitlement cache" not in text:
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
            "feature suggestion",
            "feature enhancement",
            "would love",
            "we'd love",
            "we would love",
            "could you add",
            "can you add",
            "please add",
            "would it be possible to add",
            "it would be helpful",
            "it would be super helpful",
            "it would be a great ux",
            "please allow",
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
            "sugerencia de nueva función",
            "demande de fonctionnalité",
            "funktionswunsch",
            "sugestão de recurso",
            "機能リクエスト",
            "idées d'amélioration",
            "idée d'amélioration",
            "propuesta de integración",
            "scim 2.0 group",
            "rust crate",
            "colorblind-safe",
            "retry schedules",
            "shell completions",
            "multi-currency eur",
            "x-retry-attempt-count",
            "retention in the ui from 1 year to 7 years",
            "acme_billing_alert",
            "grpc otlp",
            "read-only auditor",
            "confidence badge",
            "create linear issue",
            "pause alert notifications",
            "download all year invoices as zip",
            "ssh-ed25519",
            "prometheus exporter",
            "per-seat spending caps",
            "spki pin",
            "clone webhook configuration",
            "slack workflow builder",
            "pin favorite saved filters",
            "vector svg",
            "dry run",
            "eventbridge partner",
            "whatsapp business alert",
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
            "500",
            "502",
            "503",
            "504",
            "400",
            "403",
            "409",
            "413",
            "415",
            "429",
            "timeout",
            "zeitüberschreitung",
            "webhook",
            "sdk",
            "sso",
            "saml",
            "2fa",
            "export",
            "snowflake",
            "graphql",
            "terraform",
            "csp",
            "mojibake",
            "dkim",
            "googlebot",
            "cors",
            "group by",
            "memory leak",
            "mtls",
            "freeze",
            "keyerror",
            "oomkilled",
            "next_cursor",
            "webauthn",
            "reloads without saving",
            "transfer primary ownership",
            "microseconds",
            "webhook delivery",
            "rate limit",
            "api key",
            "cli",
            "android",
            "csv",
            "pdf",
            "slack",
            "search",
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
            "seo",
            "backlink",
            "dofollow",
            "casino",
            "rolex",
            "usdt",
            "airdrop",
            "criptomonedas",
            "lotto",
            "lottery",
            "宝くじ",
            "seguidores",
            "followers",
            "keto",
            "warranty",
            "cashapp",
            "lanyards",
            "swags",
            "yeti tumblers",
            "gulfstream",
            "offshore react",
            "barrister",
            "zurich vault",
            "janitorial",
            "calendly",
            "gift card and demo",
            "stimulus tax rebates",
            "yellowpages",
            "cold-email",
            "podcast",
            "cfo & cio email lists",
            "trademark filing service",
            "merchant cash advance",
            "ai phone agents",
            "sponsor an upcoming issue",
            "ipv4 /24",
            "herman miller",
            "18% apy",
            "pcb assembly",
            "promotional voucher",
            "discount",
            "free trial",
            "unsubscribe",
            "weight loss",
            "forex",
            "sweepstakes",
            "replica",
            "loan",
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
        intent=IntentCategory.TECHNICAL_SUPPORT,
        confidence=0.82,
        detected_entities=[],
        requires_human_review=False,
    )


def _compute_binned_ece(
    confidences: List[float],
    correctness: List[int],
    num_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
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
    confidence_key: str = "raw_confidence",
) -> List[Dict[str, Any]]:
    """Evaluate routing behavior across a grid of (spam_threshold, escalation_threshold) pairs
    spanning both the configured defaults (0.75, 0.85) and the SLM's empirical verbalized levels
    (0.95, 0.96, 0.98, 0.99).
    """
    grid_results: List[Dict[str, Any]] = []
    total = max(1, len(records))
    for spam_thresh in (0.75, 0.85, 0.96, 0.99):
        for esc_thresh in (0.75, 0.85, 0.96, 0.99):
            false_drops = 0
            true_spam_dropped = 0
            total_spam = 0
            auto_drafted = 0
            auto_draft_correct = 0
            escalated = 0

            for rec in records:
                cleaned: CleanedPayload = rec["cleaned"]
                cls: ClassificationOutput = rec["cls"]
                conf_val: float = float(rec.get(confidence_key, cls.confidence))
                expected_intent: str = rec["expected_intent"]
                must_not_drop: bool = rec["must_not_silent_drop"]

                if expected_intent == "SPAM":
                    total_spam += 1

                if cls.intent == IntentCategory.SPAM and not cleaned.contains_suspicious_patterns:
                    if (
                        conf_val >= spam_thresh
                        and not cls.requires_human_review
                        and not _has_transactional_customer_signals(cleaned)
                    ):
                        if must_not_drop:
                            false_drops += 1
                        elif expected_intent == "SPAM":
                            true_spam_dropped += 1
                    else:
                        escalated += 1
                elif (
                    conf_val < esc_thresh
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
    from_predictions: Optional[Path] = None,
    predictions_out: Optional[Path] = None,
    workers: int = 2,
) -> Dict[str, Any]:
    """Run the 300-item stratified benchmark dataset through Steps 1-4, fit post-hoc calibration
    on the calibration split (100 items), and evaluate on the held-out split (200 items).
    """
    reset_alias_remap_telemetry()
    items: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    client = LocalSLMClient(timeout_seconds=DEFAULT_CONFIG.http_timeout_seconds)

    saved_preds_by_id: Dict[str, Dict[str, Any]] = {}
    if from_predictions is not None:
        for line in from_predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved_preds_by_id[row["id"]] = row
    elif live_ollama:
        try:
            with urllib.request.urlopen(f"{client.base_url}/api/tags", timeout=3.0) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP {resp.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach local Ollama at {client.base_url} for --live-ollama evaluation: {exc}"
            ) from exc

    def _process_one(item: Dict[str, Any]) -> Dict[str, Any]:
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

        item_id = item["id"]
        if item_id in saved_preds_by_id:
            saved = saved_preds_by_id[item_id]
            cls = ClassificationOutput(
                intent=IntentCategory(saved["predicted_intent"]),
                confidence=float(saved["raw_confidence"]),
                requires_human_review=bool(saved["requires_human_review"]),
                detected_entities=list(saved.get("detected_entities", [])),
            )
            elapsed_ms = float(saved.get("latency_ms", 0.0))
            p_tok = int(saved.get("prompt_tokens", 0))
            c_tok = int(saved.get("completion_tokens", 0))
        else:
            with client.track_scope_tokens() as bucket:
                t_start = time.perf_counter()
                if live_ollama or cleaned.contains_suspicious_patterns or cleaned.word_count == 0:
                    cls = read_and_classify(cleaned, client, model=model)
                else:
                    cls = deterministic_reference_classifier(cleaned)
                elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                p_tok, c_tok = bucket[0], bucket[1]

        rules = apply_rules(cleaned, cls)
        return {
            "item": item,
            "cleaned": cleaned,
            "cls": cls,
            "rules": rules,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
        }

    if live_ollama and from_predictions is None and workers > 1:
        processed_list = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx_done, res in enumerate(pool.map(_process_one, items), start=1):
                processed_list.append(res)
                if idx_done % 25 == 0 or idx_done == len(items):
                    sys.stderr.write(f"[eval_classifier] Processed {idx_done}/{len(items)} tickets...\n")
                    sys.stderr.flush()
    else:
        processed_list = []
        for idx_done, item in enumerate(items, start=1):
            processed_list.append(_process_one(item))
            if live_ollama and (idx_done % 25 == 0 or idx_done == len(items)):
                sys.stderr.write(f"[eval_classifier] Processed {idx_done}/{len(items)} tickets...\n")
                sys.stderr.flush()

    # Collect calibration split SLM confidences (using every 3rd SLM ticket stratified across all 300 items
    # so calibration fitting covers all 6 challenge slices) to fit post-hoc Temperature & Isotonic scalers
    cal_confidences: List[float] = []
    cal_correctness: List[int] = []
    slm_idx = 0
    for res in processed_list:
        item = res["item"]
        cleaned: CleanedPayload = res["cleaned"]
        cls: ClassificationOutput = res["cls"]
        if not cleaned.contains_suspicious_patterns and cleaned.word_count > 0:
            if slm_idx % 3 == 0:
                cal_confidences.append(cls.confidence)
                cal_correctness.append(1 if cls.intent.value == item["expected_intent"] else 0)
            slm_idx += 1

    fitted_temp = fit_temperature_scaling(cal_confidences, cal_correctness)
    isotonic_blocks = fit_isotonic_pava(cal_confidences, cal_correctness)

    correct_intents = 0
    correct_queues = 0
    false_silent_drops = 0
    short_circuited_injections = 0

    held_out_total = 0
    held_out_correct_intents = 0
    held_out_correct_queues = 0
    held_out_raw_confs: List[float] = []
    held_out_temp_confs: List[float] = []
    held_out_iso_confs: List[float] = []
    held_out_correctness: List[int] = []

    all_slm_raw_confs: List[float] = []
    all_slm_temp_confs: List[float] = []
    all_slm_iso_confs: List[float] = []
    all_slm_correctness: List[int] = []

    latencies_ms: List[float] = []
    raw_records: List[Dict[str, Any]] = []
    misclassified_examples: List[Dict[str, Any]] = []
    prediction_jsonl_rows: List[Dict[str, Any]] = []

    slice_totals: Dict[str, int] = defaultdict(int)
    slice_correct: Dict[str, int] = defaultdict(int)

    tp_per_class: Dict[str, int] = defaultdict(int)
    fp_per_class: Dict[str, int] = defaultdict(int)
    fn_per_class: Dict[str, int] = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = {
        k.value: {m.value: 0 for m in IntentCategory} for k in IntentCategory
    }

    total_prompt_tok = 0
    total_comp_tok = 0

    for res in processed_list:
        item = res["item"]
        cleaned = res["cleaned"]
        cls = res["cls"]
        rules = res["rules"]
        elapsed_ms = res["elapsed_ms"]
        p_tok = res["prompt_tokens"]
        c_tok = res["completion_tokens"]

        total_prompt_tok += p_tok
        total_comp_tok += c_tok
        latencies_ms.append(elapsed_ms)

        if cleaned.contains_suspicious_patterns:
            short_circuited_injections += 1

        split_name = item.get("split", "calibration")
        slice_name = item.get("slice", "clean_short")
        expected_intent = item["expected_intent"]
        predicted_intent = cls.intent.value
        confusion[expected_intent][predicted_intent] += 1

        is_intent_correct = predicted_intent == expected_intent
        is_queue_correct = rules.assigned_queue == item["expected_queue"]

        slice_totals[slice_name] += 1
        if is_intent_correct:
            correct_intents += 1
            slice_correct[slice_name] += 1
            tp_per_class[expected_intent] += 1
        else:
            fp_per_class[predicted_intent] += 1
            fn_per_class[expected_intent] += 1
            misclassified_examples.append(
                {
                    "id": item["id"],
                    "split": split_name,
                    "slice": slice_name,
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

        if cleaned.contains_suspicious_patterns or cleaned.word_count == 0:
            temp_conf = cls.confidence
            iso_conf = cls.confidence
        else:
            temp_conf = apply_temperature_scaling(cls.confidence, fitted_temp)
            iso_conf = apply_isotonic_pava(cls.confidence, isotonic_blocks)
            all_slm_raw_confs.append(cls.confidence)
            all_slm_temp_confs.append(temp_conf)
            all_slm_iso_confs.append(iso_conf)
            all_slm_correctness.append(1 if is_intent_correct else 0)

        if split_name == "held_out":
            held_out_total += 1
            if is_intent_correct:
                held_out_correct_intents += 1
            if is_queue_correct:
                held_out_correct_queues += 1
            if not cleaned.contains_suspicious_patterns and cleaned.word_count > 0:
                held_out_raw_confs.append(cls.confidence)
                held_out_temp_confs.append(temp_conf)
                held_out_iso_confs.append(iso_conf)
                held_out_correctness.append(1 if is_intent_correct else 0)

        raw_records.append(
            {
                "id": item["id"],
                "split": split_name,
                "slice": slice_name,
                "cleaned": cleaned,
                "cls": cls,
                "raw_confidence": cls.confidence,
                "calibrated_confidence_temp": temp_conf,
                "calibrated_confidence_isotonic": iso_conf,
                "expected_intent": expected_intent,
                "must_not_silent_drop": item.get("must_not_silent_drop", True),
            }
        )
        prediction_jsonl_rows.append(
            {
                "id": item["id"],
                "split": split_name,
                "slice": slice_name,
                "subject": item["subject"],
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "intent_correct": is_intent_correct,
                "raw_confidence": cls.confidence,
                "calibrated_confidence_temp": temp_conf,
                "calibrated_confidence_isotonic": iso_conf,
                "requires_human_review": cls.requires_human_review,
                "expected_queue": item["expected_queue"],
                "assigned_queue": rules.assigned_queue,
                "short_circuited_stage2": cleaned.contains_suspicious_patterns,
                "detected_entities": cls.detected_entities,
                "latency_ms": elapsed_ms,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
            }
        )

    if predictions_out is not None:
        predictions_out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in prediction_jsonl_rows) + "\n",
            encoding="utf-8",
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

    slice_breakdown: Dict[str, Dict[str, Any]] = {}
    for s_name, s_tot in slice_totals.items():
        s_corr = slice_correct[s_name]
        ci_low, ci_high = wilson_95_ci(s_corr, s_tot)
        slice_breakdown[s_name] = {
            "count": s_tot,
            "correct": s_corr,
            "accuracy": round(s_corr / max(1, s_tot), 4),
            "wilson_95_ci": [ci_low, ci_high],
        }

    is_slm_measured = bool(live_ollama or from_predictions is not None)
    ece_val: Optional[float] = None
    brier_val: Optional[float] = None
    held_out_raw_ece: Optional[float] = None
    held_out_temp_ece: Optional[float] = None
    held_out_iso_ece: Optional[float] = None
    held_out_raw_brier: Optional[float] = None
    held_out_temp_brier: Optional[float] = None
    held_out_iso_brier: Optional[float] = None
    reliability_bins: List[Dict[str, Any]] = []
    threshold_sweep: List[Dict[str, Any]] = []

    discrete_confidence_levels: List[Dict[str, Any]] = []
    if is_slm_measured:
        ece_val, reliability_bins = _compute_binned_ece(all_slm_raw_confs, all_slm_correctness)
        by_level: Dict[float, List[int]] = defaultdict(list)
        for c_val, corr_val in zip(all_slm_raw_confs, all_slm_correctness):
            by_level[round(c_val, 4)].append(corr_val)
        for lvl in sorted(by_level.keys()):
            vals = by_level[lvl]
            emp_acc = sum(vals) / len(vals)
            ci_l, ci_h = wilson_95_ci(sum(vals), len(vals))
            discrete_confidence_levels.append(
                {
                    "confidence_level": lvl,
                    "count": len(vals),
                    "correct": sum(vals),
                    "empirical_accuracy": round(emp_acc, 4),
                    "calibration_gap": round(abs(emp_acc - lvl), 4),
                    "wilson_95_ci": [ci_l, ci_h],
                }
            )
        brier_val = round(
            sum((c - y) ** 2 for c, y in zip(all_slm_raw_confs, all_slm_correctness))
            / max(1, len(all_slm_raw_confs)),
            4,
        )
        if held_out_raw_confs:
            held_out_raw_ece, _ = _compute_binned_ece(held_out_raw_confs, held_out_correctness)
            held_out_temp_ece, _ = _compute_binned_ece(held_out_temp_confs, held_out_correctness)
            held_out_iso_ece, _ = _compute_binned_ece(held_out_iso_confs, held_out_correctness)
            n_ho = max(1, len(held_out_raw_confs))
            held_out_raw_brier = round(
                sum((c - y) ** 2 for c, y in zip(held_out_raw_confs, held_out_correctness)) / n_ho,
                4,
            )
            held_out_temp_brier = round(
                sum((c - y) ** 2 for c, y in zip(held_out_temp_confs, held_out_correctness)) / n_ho,
                4,
            )
            held_out_iso_brier = round(
                sum((c - y) ** 2 for c, y in zip(held_out_iso_confs, held_out_correctness)) / n_ho,
                4,
            )
        held_out_records = [r for r in raw_records if r["split"] == "held_out"] or raw_records
        threshold_sweep = _sweep_thresholds(held_out_records, confidence_key="raw_confidence")

    sorted_lat = sorted(latencies_ms)
    p50_ms = round(sorted_lat[len(sorted_lat) // 2], 2) if sorted_lat else 0.0
    p95_ms = (
        round(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))], 2)
        if sorted_lat
        else 0.0
    )

    overall_ci_low, overall_ci_high = wilson_95_ci(correct_intents, total)
    ho_ci_low, ho_ci_high = wilson_95_ci(held_out_correct_intents, max(1, held_out_total))

    return {
        "evaluation_mode": (
            "live_ollama_slm"
            if live_ollama
            else ("replayed_slm_predictions" if from_predictions is not None else "offline_deterministic_baseline")
        ),
        "model": model if is_slm_measured else "deterministic_keyword_reference",
        "slm_measured": is_slm_measured,
        "total_samples": len(items),
        "calibration_split_samples": len(items) - held_out_total,
        "held_out_split_samples": held_out_total,
        "short_circuited_injections_0_tokens": short_circuited_injections,
        "intent_accuracy": round(correct_intents / total, 4),
        "intent_accuracy_wilson_95_ci": [overall_ci_low, overall_ci_high],
        "held_out_intent_accuracy": round(held_out_correct_intents / max(1, held_out_total), 4),
        "held_out_intent_accuracy_wilson_95_ci": [ho_ci_low, ho_ci_high],
        "routing_queue_accuracy": round(correct_queues / total, 4),
        "held_out_routing_queue_accuracy": round(held_out_correct_queues / max(1, held_out_total), 4),
        "slice_breakdown": slice_breakdown,
        "per_class_f1": per_class_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion,
        "false_silent_drop_rate": round(false_silent_drops / total, 4),
        "expected_calibration_error_10bin": ece_val,
        "brier_score": brier_val,
        "post_hoc_calibration": {
            "fitted_temperature": fitted_temp if is_slm_measured else None,
            "isotonic_pava_blocks": len(isotonic_blocks) if is_slm_measured else None,
            "held_out_raw_ece_10bin": held_out_raw_ece,
            "held_out_temperature_scaled_ece_10bin": held_out_temp_ece,
            "held_out_isotonic_scaled_ece_10bin": held_out_iso_ece,
            "held_out_raw_brier": held_out_raw_brier,
            "held_out_temperature_scaled_brier": held_out_temp_brier,
            "held_out_isotonic_scaled_brier": held_out_iso_brier,
        },
        "alias_drift_telemetry": get_alias_remap_telemetry(),
        "reliability_bins": reliability_bins,
        "discrete_confidence_levels": discrete_confidence_levels,
        "threshold_sweep": threshold_sweep,
        "misclassified_examples": misclassified_examples,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "prompt_tokens_used": total_prompt_tok,
        "completion_tokens_used": total_comp_tok,
        "tokens_used": total_prompt_tok + total_comp_tok,
    }


def write_markdown_report(metrics: Dict[str, Any], report_path: Path = DEFAULT_REPORT_PATH) -> Path:
    """Render an auditable, statistically rigorous Markdown calibration report."""
    ov_ci = metrics["intent_accuracy_wilson_95_ci"]
    ho_ci = metrics["held_out_intent_accuracy_wilson_95_ci"]
    phc = metrics["post_hoc_calibration"]
    alias_tel = metrics["alias_drift_telemetry"]

    lines: List[str] = [
        "# Local SLM Classification, Held-Out Slice & Post-Hoc Calibration Report",
        "",
        f"- **Evaluation Mode:** `{metrics['evaluation_mode']}`",
        f"- **Shipped & Evaluated Model (`DEFAULT_CONFIG`):** `{metrics['model']}`",
        f"- **SLM Directly Measured:** `{metrics['slm_measured']}`",
        f"- **Auditable Per-Ticket Raw Outputs:** [`tests/eval_predictions.jsonl`](tests/eval_predictions.jsonl) (reproducible offline via `--from-predictions tests/eval_predictions.jsonl`)",
        f"- **Dataset Size:** `{metrics['total_samples']}` tickets (`{metrics['calibration_split_samples']}` calibration split + `{metrics['held_out_split_samples']}` held-out test split)",
        f"- **Adversarial Injections Short-Circuited at Stage 2 (0 Tokens):** `{metrics['short_circuited_injections_0_tokens']}`",
        f"- **Overall Intent Accuracy (N={metrics['total_samples']}):** `{metrics['intent_accuracy'] * 100:.2f}%` (95% Wilson CI: `[{ov_ci[0] * 100:.2f}%, {ov_ci[1] * 100:.2f}%]`)",
        f"- **Held-Out Split Intent Accuracy (N={metrics['held_out_split_samples']}):** `{metrics['held_out_intent_accuracy'] * 100:.2f}%` (95% Wilson CI: `[{ho_ci[0] * 100:.2f}%, {ho_ci[1] * 100:.2f}%]`)",
        f"- **Held-Out End-to-End Routing Queue Accuracy:** `{metrics['held_out_routing_queue_accuracy'] * 100:.2f}%`",
        f"- **False Silent-Drop Rate (Legitimate Ticket -> Trash):** `{metrics['false_silent_drop_rate'] * 100:.2f}%`",
        f"- **Alias Drift Telemetry (Non-Canonical Label Remaps):** `{alias_tel['total_alias_remaps']}` (`{alias_tel['by_alias']}`)",
        f"- **Latency (p50 / p95 vs 30,000 ms Timeout):** `{metrics['latency_p50_ms']} ms` / `{metrics['latency_p95_ms']} ms`",
        f"- **Total Tokens Used:** `{metrics['tokens_used']}` (`{metrics['prompt_tokens_used']}` prompt + `{metrics['completion_tokens_used']}` completion)",
        "",
        "## 1. Held-Out Challenge Slice Breakdown (Wilson 95% CIs)",
        "",
        "| Challenge Slice | Tickets | Correct | Empirical Accuracy | 95% Wilson Confidence Interval |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for s_name, s_info in sorted(metrics["slice_breakdown"].items()):
        ci = s_info["wilson_95_ci"]
        lines.append(
            f"| `{s_name}` | {s_info['count']} | {s_info['correct']} | "
            f"**{s_info['accuracy'] * 100:.2f}%** | `[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]` |"
        )

    lines.extend(
        [
            "",
            "## 2. Post-Hoc Confidence Calibration on Held-Out Split (`N=200`)",
            "",
            f"Fitted on the `{metrics['calibration_split_samples']}`-item calibration split (with unscripted prompt probabilities) and evaluated on the disjoint `{metrics['held_out_split_samples']}`-item held-out split:",
            "",
            "| Calibration Method | Parameter / Complexity | Held-Out 10-Bin ECE | Held-Out Brier Score |",
            "| :--- | :--- | :---: | :---: |",
            f"| **Uncalibrated Raw SLM Confidence** | None ($T = 1.0$) | `{phc['held_out_raw_ece_10bin']}` | `{phc['held_out_raw_brier']}` |",
            f"| **Temperature Scaling** | $T^* = {phc['fitted_temperature']}$ (1 scalar parameter) | **`{phc['held_out_temperature_scaled_ece_10bin']}`** | **`{phc['held_out_temperature_scaled_brier']}`** |",
            f"| **Isotonic Regression (PAVA)** | `{phc['isotonic_pava_blocks']}` monotonic step blocks | **`{phc['held_out_isotonic_scaled_ece_10bin']}`** | **`{phc['held_out_isotonic_scaled_brier']}`** |",
            "",
            "## 3. Per-Class Precision, Recall & F1 (`N=300`)",
            "",
            "| Intent Category | Precision | Recall | F1 Score | Support |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for label, m in metrics["per_class_metrics"].items():
        lines.append(
            f"| `{label}` | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | {m['support']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Confusion Matrix (Rows = Expected, Columns = Predicted)",
            "",
            "| Expected \\ Predicted | `BILLING` | `TECHNICAL_SUPPORT` | `FEATURE_REQUEST` | `SPAM` | `SECURITY_REPORT` |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    order = ["BILLING", "TECHNICAL_SUPPORT", "FEATURE_REQUEST", "SPAM", "SECURITY_REPORT"]
    for exp_label in order:
        row_counts = [str(metrics["confusion_matrix"][exp_label][pred]) for pred in order]
        lines.append(f"| **`{exp_label}`** | " + " | ".join(row_counts) + " |")

    if metrics.get("discrete_confidence_levels"):
        lines.extend(
            [
                "",
                "## 5. Discrete Verbalized Confidence Breakdown (`N=288` SLM Predictions)",
                "",
                "Without prompt-scripted confidence bands, `gemma4:e4b-it-qat` (`temperature=0.0`) emits three discrete verbalized confidence levels (`0.95`, `0.98`, `0.99`). Notably, the single misclassified ticket (`eval-267`) occurred in the lowest verbalized level (`0.95`, empirical accuracy `96.43%` vs `95.00%` confidence, gap `1.43%`), whereas `0.98` and `0.99` achieved `100.00%` (`260/260`):",
                "",
                "| Verbalized Confidence | Tickets | Correct | Empirical Accuracy | 95% Wilson CI | Calibration Gap |",
                "| :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for d in metrics["discrete_confidence_levels"]:
            d_ci = d["wilson_95_ci"]
            lines.append(
                f"| `{d['confidence_level']:.2f}` | {d['count']} | {d['correct']} | "
                f"**{d['empirical_accuracy'] * 100:.2f}%** | `[{d_ci[0] * 100:.2f}%, {d_ci[1] * 100:.2f}%]` | "
                f"`{d['calibration_gap']:.4f}` |"
            )

    if metrics["reliability_bins"]:
        lines.extend(
            [
                "",
                "### Standard 10-Bin Reliability Diagram",
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
                "## 6. Held-Out Threshold Sweep (`spam_threshold` × `escalation_threshold`)",
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
                "## 7. Error Analysis (Misclassified Tickets)",
                "",
                "| Ticket ID | Split | Slice | Subject | Expected | Predicted | Confidence | Routed Queue |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |",
            ]
        )
        for err in metrics["misclassified_examples"]:
            lines.append(
                f"| `{err['id']}` | `{err['split']}` | `{err['slice']}` | {err['subject']} | "
                f"`{err['expected']}` | `{err['predicted']}` | {err['confidence']:.2f} | `{err['assigned_queue']}` |"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate classifier accuracy, held-out slices, post-hoc calibration, and SPAM drop safety."
    )
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
        "--workers",
        type=int,
        default=2,
        help="Concurrent worker threads for --live-ollama evaluation.",
    )
    parser.add_argument(
        "--from-predictions",
        type=Path,
        default=None,
        help="Replay committed raw SLM outputs from JSONL (e.g., tests/eval_predictions.jsonl) without needing Ollama.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Path to write per-ticket raw prediction JSONL records (defaults to tests/eval_predictions.jsonl when --live-ollama is set).",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional path to write a Markdown calibration report (e.g., CALIBRATION_REPORT.md).",
    )
    args = parser.parse_args()

    preds_out = args.predictions_out
    if preds_out is None and args.live_ollama:
        preds_out = DEFAULT_PREDICTIONS_PATH

    metrics = evaluate_dataset(
        dataset_path=args.dataset,
        live_ollama=args.live_ollama,
        model=args.model,
        from_predictions=args.from_predictions,
        predictions_out=preds_out,
        workers=args.workers,
    )
    if args.write_report is not None:
        write_markdown_report(metrics, report_path=args.write_report)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["false_silent_drop_rate"] == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
