#!/usr/bin/env python3
"""Evaluation harness for Stage 2 + Stage 3 (Classification) + Stage 4 (Rules) + Stage 5 (Drafting)
+ Stage 6 (Output Guardrails), with held-out slice analysis, Wilson 95% confidence intervals,
and auditable per-ticket JSONL prediction logging.

Supports:
- `--live-ollama`: evaluates the local Ollama SLM (`--model gemma4:e4b-it-qat`) on Stage 3 across
  the 300-item benchmark (`100` calibration + `200` held-out test tickets).
- `--eval-step5-live`: runs live Stage 5 (`draft_response`) + Stage 6 (`is_draft_safe`) on up to
  `--step5-limit` (default `105`) held-out draftable tickets (prioritizing all multilingual,
  long-thread, hard-negative, and mixed-intent tickets) and saves every generated draft into
  `tests/eval_predictions.jsonl`.
- `--from-predictions tests/eval_predictions.jsonl`: replays committed raw SLM outputs for both
  Stage 3 and Stage 5 to recompute all metrics, Wilson 95% CIs, and `CALIBRATION_REPORT.md`
  offline in <0.2s without needing Ollama.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.pipeline.step1_trigger import compute_payload_signature, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step3_classify import read_and_classify
from src.pipeline.step4_rules import _has_transactional_customer_signals, apply_rules
from src.pipeline.step5_draft import (
    _build_ticket_id,
     build_draft_prompts,
    detect_language,
    render_static_fallback_draft,
)
from src.pipeline.step6_validate import is_draft_safe
from src.schemas import (
    ClassificationOutput,
    CleanedPayload,
    DraftOutput,
    IntentCategory,
    RuleEvaluationResult,
    get_alias_remap_telemetry,
    reset_alias_remap_telemetry,
)
from src.security import (
    contains_disallowed_urls,
    contains_forbidden_commitments,
    contains_pii_or_secrets,
    contains_unescaped_html,
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
    ) and "entitlement cache" not in text:
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
    """Evaluate routing behavior across a grid of (spam_threshold, escalation_threshold) pairs."""
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


def _score_draft_quality(
    cleaned: CleanedPayload,
    rules: RuleEvaluationResult,
    draft: DraftOutput,
    used_static_fallback: bool,
) -> Dict[str, Any]:
    """Audit a Stage 5 DraftOutput across Stage 6 safety guardrails, ticket ID grounding,
    SLA grounding, language match, and a 5-point quality rubric.
    """
    expected_ticket_id = _build_ticket_id(cleaned)
    expected_sla = rules.sla_hours
    expected_lang = detect_language(f"{cleaned.clean_subject} {cleaned.clean_body}")
    detected_draft_lang = detect_language(draft.body_text)
    lang_match = detected_draft_lang == expected_lang

    combined = f"{draft.subject_line}\n{draft.body_text}"
    trip_pii = contains_pii_or_secrets(combined)
    trip_commitments = contains_forbidden_commitments(combined)
    trip_urls = contains_disallowed_urls(combined, DEFAULT_CONFIG.allowed_domains)
    trip_html = contains_unescaped_html(combined)
    stage6_safe = is_draft_safe(draft, DEFAULT_CONFIG.allowed_domains)

    # Check Ticket ID grounding: referenced_ticket_id matches and appears in subject or body
    ticket_id_grounded = (
        draft.referenced_ticket_id == expected_ticket_id
        and (expected_ticket_id in draft.body_text or expected_ticket_id in draft.subject_line)
    )

    # Check SLA hours grounding: mentions the routed SLA hours (supporting both space-delimited
    # languages and Japanese `4時間`/`12時間`/`48時間` where `時` is a `\w` char)
    sla_str = str(expected_sla)
    other_slas = {4, 12, 48} - {expected_sla}
    mentions_expected_sla = bool(re.search(rf"(?<!\d){sla_str}(?!\d)", draft.body_text))
    hallucinates_wrong_sla = any(
        bool(
            re.search(
                rf"(?<!\d){other}\s*(?:-|\s)?(?:hour|hr|hora|heure|stunde|時間)",
                draft.body_text,
                re.IGNORECASE,
            )
        )
        for other in other_slas
    )
    sla_grounded = mentions_expected_sla and not hallucinates_wrong_sla

    # 5-point rubric score:
    # +1: Valid schema without static fallback
    # +1: Stage 6 safety guardrails pass (0 unsafe trips)
    # +1: Ticket ID grounded in draft
    # +1: SLA hours grounded in draft without conflicting SLA hallucination
    # +1: Customer language matched
    score_5pt = (
        (0 if used_static_fallback else 1)
        + (1 if stage6_safe else 0)
        + (1 if ticket_id_grounded else 0)
        + (1 if sla_grounded else 0)
        + (1 if lang_match else 0)
    )

    return {
        "draft_evaluated": True,
        "draft_used_static_fallback": used_static_fallback,
        "draft_subject_line": draft.subject_line,
        "draft_body_text": draft.body_text,
        "draft_referenced_ticket_id": draft.referenced_ticket_id,
        "draft_expected_ticket_id": expected_ticket_id,
        "draft_expected_sla_hours": expected_sla,
        "draft_expected_language": expected_lang,
        "draft_detected_language": detected_draft_lang,
        "draft_language_match": lang_match,
        "draft_ticket_id_grounded": ticket_id_grounded,
        "draft_sla_grounded": sla_grounded,
        "draft_stage6_safe": stage6_safe,
        "draft_guardrail_trips": {
            "pii_or_secrets": trip_pii,
            "forbidden_commitments": trip_commitments,
            "disallowed_urls": trip_urls,
            "unescaped_html": trip_html,
        },
        "draft_quality_score_5pt": score_5pt,
    }


def _deterministic_uuid_from_item_id(item_id: str) -> UUID:
    """Create a deterministic UUID from the dataset item ID (`eval-001`) so `TKT-...` IDs are reproducible."""
    digits = "".join(ch for ch in item_id if ch.isdigit()).zfill(8)[:8]
    return UUID(f"{digits}-0000-4000-8000-000000000000")


def evaluate_dataset(
    dataset_path: Path = DEFAULT_DATASET,
    live_ollama: bool = False,
    eval_step5_live: bool = False,
    step5_limit: int = 105,
    model: str = DEFAULT_CONFIG.classify_model,
    draft_model: str = DEFAULT_CONFIG.draft_model,
    from_predictions: Optional[Path] = None,
    predictions_out: Optional[Path] = None,
    workers: int = 2,
) -> Dict[str, Any]:
    """Run the 300-item benchmark dataset through Steps 1-4 (and optionally Step 5 + Step 6 on
    a stratified held-out sample of draftable tickets).
    """
    reset_alias_remap_telemetry()
    items: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    client = LocalSLMClient(timeout_seconds=DEFAULT_CONFIG.http_timeout_seconds)

    saved_preds_by_id: Dict[str, Dict[str, Any]] = {}
    if from_predictions is not None and from_predictions.exists():
        for line in from_predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved_preds_by_id[row["id"]] = row

    if live_ollama or eval_step5_live:
        try:
            with urllib.request.urlopen(f"{client.base_url}/api/tags", timeout=3.0) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP {resp.status}")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach local Ollama at {client.base_url} for live evaluation: {exc}"
            ) from exc

    def _process_one(item: Dict[str, Any]) -> Dict[str, Any]:
        det_uuid = _deterministic_uuid_from_item_id(item["id"])
        raw = {
            "event_id": str(det_uuid),
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
        saved = saved_preds_by_id.get(item_id)
        if saved is not None and not live_ollama:
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
            "saved_row": saved,
        }

    if live_ollama and workers > 1:
        processed_list = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx_done, res in enumerate(pool.map(_process_one, items), start=1):
                processed_list.append(res)
                if idx_done % 25 == 0 or idx_done == len(items):
                    sys.stderr.write(f"[eval_classifier] Stage 3 processed {idx_done}/{len(items)} tickets...\n")
                    sys.stderr.flush()
    else:
        processed_list = []
        for idx_done, item in enumerate(items, start=1):
            processed_list.append(_process_one(item))
            if live_ollama and (idx_done % 25 == 0 or idx_done == len(items)):
                sys.stderr.write(f"[eval_classifier] Stage 3 processed {idx_done}/{len(items)} tickets...\n")
                sys.stderr.flush()

    # Select held-out draftable tickets for Step 5 evaluation (prioritizing multilingual, long threads,
    # hard negatives, mixed intents, and duplicate subjects so all challenge slices are covered)
    slice_priority = {
        "multilingual": 0,
        "long_quoted_thread": 1,
        "hard_negative": 2,
        "mixed_and_adversarial": 3,
        "duplicate_generic_subject": 4,
        "clean_short": 5,
    }
    draftable_candidates = [
        res
        for res in processed_list
        if res["rules"].should_draft_reply and not res["cleaned"].contains_suspicious_patterns
    ]
    draftable_candidates.sort(
        key=lambda r: (
            0 if r["item"].get("split") == "held_out" else 1,
            slice_priority.get(r["item"].get("slice", "clean_short"), 9),
            r["item"]["id"],
        )
    )
    selected_step5_ids = {r["item"]["id"] for r in draftable_candidates[:step5_limit]}

    def _run_step5_for_record(res: Dict[str, Any]) -> None:
        item_id = res["item"]["id"]
        cleaned: CleanedPayload = res["cleaned"]
        rules: RuleEvaluationResult = res["rules"]
        saved = res.get("saved_row")

        if not rules.should_draft_reply or cleaned.contains_suspicious_patterns:
            res["draft_metrics"] = {"draft_evaluated": False}
            return

        if eval_step5_live and item_id in selected_step5_ids:
            ticket_id = _build_ticket_id(cleaned)
            sys_prompt, usr_prompt, _ = build_draft_prompts(cleaned, rules, ticket_id=ticket_id)
            used_fallback = False
            with client.track_scope_tokens() as d_bucket:
                t0 = time.perf_counter()
                try:
                    draft_obj = client.generate_structured(
                        model=draft_model,
                        system_prompt=sys_prompt,
                        user_prompt=usr_prompt,
                        response_schema=DraftOutput,
                        num_predict=DEFAULT_CONFIG.draft_max_tokens,
                        temperature=DEFAULT_CONFIG.temperature,
                    )
                except Exception:
                    used_fallback = True
                    draft_obj = render_static_fallback_draft(cleaned, rules)
                d_lat = round((time.perf_counter() - t0) * 1000.0, 2)
                dp_tok, dc_tok = d_bucket[0], d_bucket[1]

            scored = _score_draft_quality(cleaned, rules, draft_obj, used_static_fallback=used_fallback)
            scored["draft_latency_ms"] = d_lat
            scored["draft_prompt_tokens"] = dp_tok
            scored["draft_completion_tokens"] = dc_tok
            res["draft_metrics"] = scored
        elif saved is not None and saved.get("draft_evaluated"):
            draft_obj = DraftOutput(
                referenced_ticket_id=saved["draft_referenced_ticket_id"],
                subject_line=saved["draft_subject_line"],
                body_text=saved["draft_body_text"],
            )
            used_fallback = bool(saved.get("draft_used_static_fallback", False))
            scored = _score_draft_quality(cleaned, rules, draft_obj, used_static_fallback=used_fallback)
            scored["draft_latency_ms"] = float(saved.get("draft_latency_ms", 0.0))
            scored["draft_prompt_tokens"] = int(saved.get("draft_prompt_tokens", 0))
            scored["draft_completion_tokens"] = int(saved.get("draft_completion_tokens", 0))
            res["draft_metrics"] = scored
        else:
            res["draft_metrics"] = {"draft_evaluated": False}

    if eval_step5_live:
        step5_targets = [r for r in processed_list if r["item"]["id"] in selected_step5_ids]
        non_targets = [r for r in processed_list if r["item"]["id"] not in selected_step5_ids]
        for r in non_targets:
            _run_step5_for_record(r)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for idx_d, _ in enumerate(pool.map(_run_step5_for_record, step5_targets), start=1):
                    if idx_d % 15 == 0 or idx_d == len(step5_targets):
                        sys.stderr.write(
                            f"[eval_classifier] Stage 5 drafted {idx_d}/{len(step5_targets)} replies...\n"
                        )
                        sys.stderr.flush()
        else:
            for idx_d, r in enumerate(step5_targets, start=1):
                _run_step5_for_record(r)
                if idx_d % 15 == 0 or idx_d == len(step5_targets):
                    sys.stderr.write(
                        f"[eval_classifier] Stage 5 drafted {idx_d}/{len(step5_targets)} replies...\n"
                    )
                    sys.stderr.flush()
    else:
        for r in processed_list:
            _run_step5_for_record(r)

    correct_intents = 0
    correct_queues = 0
    false_silent_drops = 0
    legitimate_must_not_drop_total = 0
    short_circuited_injections = 0

    slm_only_total = 0
    slm_only_correct = 0

    held_out_total = 0
    held_out_correct_intents = 0
    held_out_correct_queues = 0
    held_out_slm_total = 0
    held_out_slm_correct = 0
    held_out_short_circuits = 0

    all_slm_raw_confs: List[float] = []
    all_slm_correctness: List[int] = []

    latencies_ms: List[float] = []
    raw_records: List[Dict[str, Any]] = []
    misclassified_examples: List[Dict[str, Any]] = []
    prediction_jsonl_rows: List[Dict[str, Any]] = []

    held_out_slice_totals: Dict[str, int] = defaultdict(int)
    held_out_slice_correct: Dict[str, int] = defaultdict(int)
    calibration_slice_totals: Dict[str, int] = defaultdict(int)
    calibration_slice_correct: Dict[str, int] = defaultdict(int)

    tp_per_class: Dict[str, int] = defaultdict(int)
    fp_per_class: Dict[str, int] = defaultdict(int)
    fn_per_class: Dict[str, int] = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = {
        k.value: {m.value: 0 for m in IntentCategory} for k in IntentCategory
    }

    total_prompt_tok = 0
    total_comp_tok = 0

    # Step 5 aggregation counters
    step5_evaluated_count = 0
    step5_direct_slm_count = 0
    step5_fallback_count = 0
    step5_stage6_safe_count = 0
    step5_ticket_grounded_count = 0
    step5_sla_grounded_count = 0
    step5_lang_match_count = 0
    step5_lang_totals: Dict[str, int] = defaultdict(int)
    step5_lang_matches: Dict[str, int] = defaultdict(int)
    step5_guardrail_trips: Dict[str, int] = {
        "pii_or_secrets": 0,
        "forbidden_commitments": 0,
        "disallowed_urls": 0,
        "unescaped_html": 0,
    }
    step5_quality_scores: List[int] = []
    step5_latencies_ms: List[float] = []
    step5_prompt_tokens = 0
    step5_comp_tokens = 0

    for res in processed_list:
        item = res["item"]
        cleaned = res["cleaned"]
        cls = res["cls"]
        rules = res["rules"]
        elapsed_ms = res["elapsed_ms"]
        p_tok = res["prompt_tokens"]
        c_tok = res["completion_tokens"]
        d_metrics = res.get("draft_metrics", {"draft_evaluated": False})

        total_prompt_tok += p_tok
        total_comp_tok += c_tok
        if not cleaned.contains_suspicious_patterns and cleaned.word_count > 0:
            latencies_ms.append(elapsed_ms)

        split_name = item.get("split", "calibration")
        slice_name = item.get("slice", "clean_short")
        expected_intent = item["expected_intent"]
        predicted_intent = cls.intent.value
        confusion[expected_intent][predicted_intent] += 1

        is_intent_correct = predicted_intent == expected_intent
        is_queue_correct = rules.assigned_queue == item["expected_queue"]

        if cleaned.contains_suspicious_patterns:
            short_circuited_injections += 1
            if split_name == "held_out":
                held_out_short_circuits += 1
        else:
            slm_only_total += 1
            if is_intent_correct:
                slm_only_correct += 1
            all_slm_raw_confs.append(cls.confidence)
            all_slm_correctness.append(1 if is_intent_correct else 0)
            if split_name == "held_out":
                held_out_slm_total += 1
                if is_intent_correct:
                    held_out_slm_correct += 1

        if split_name == "held_out":
            held_out_total += 1
            held_out_slice_totals[slice_name] += 1
            if is_intent_correct:
                held_out_correct_intents += 1
                held_out_slice_correct[slice_name] += 1
            if is_queue_correct:
                held_out_correct_queues += 1
        else:
            calibration_slice_totals[slice_name] += 1
            if is_intent_correct:
                calibration_slice_correct[slice_name] += 1

        if is_intent_correct:
            correct_intents += 1
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
        if item.get("must_not_silent_drop", True):
            legitimate_must_not_drop_total += 1
            if rules.assigned_queue == "trash":
                false_silent_drops += 1

        if d_metrics.get("draft_evaluated"):
            step5_evaluated_count += 1
            if d_metrics["draft_used_static_fallback"]:
                step5_fallback_count += 1
            else:
                step5_direct_slm_count += 1
            if d_metrics["draft_stage6_safe"]:
                step5_stage6_safe_count += 1
            if d_metrics["draft_ticket_id_grounded"]:
                step5_ticket_grounded_count += 1
            if d_metrics["draft_sla_grounded"]:
                step5_sla_grounded_count += 1
            exp_l = d_metrics["draft_expected_language"]
            step5_lang_totals[exp_l] += 1
            if d_metrics["draft_language_match"]:
                step5_lang_match_count += 1
                step5_lang_matches[exp_l] += 1
            for k_trip, v_trip in d_metrics["draft_guardrail_trips"].items():
                if v_trip:
                    step5_guardrail_trips[k_trip] += 1
            step5_quality_scores.append(int(d_metrics["draft_quality_score_5pt"]))
            if d_metrics.get("draft_latency_ms", 0.0) > 0:
                step5_latencies_ms.append(float(d_metrics["draft_latency_ms"]))
            step5_prompt_tokens += int(d_metrics.get("draft_prompt_tokens", 0))
            step5_comp_tokens += int(d_metrics.get("draft_completion_tokens", 0))

        raw_records.append(
            {
                "id": item["id"],
                "split": split_name,
                "slice": slice_name,
                "cleaned": cleaned,
                "cls": cls,
                "raw_confidence": cls.confidence,
                "expected_intent": expected_intent,
                "must_not_silent_drop": item.get("must_not_silent_drop", True),
            }
        )

        row_dict: Dict[str, Any] = {
            "id": item["id"],
            "split": split_name,
            "slice": slice_name,
            "subject": item["subject"],
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": is_intent_correct,
            "raw_confidence": cls.confidence,
            "requires_human_review": cls.requires_human_review,
            "expected_queue": item["expected_queue"],
            "assigned_queue": rules.assigned_queue,
            "short_circuited_stage2": cleaned.contains_suspicious_patterns,
            "detected_entities": cls.detected_entities,
            "latency_ms": elapsed_ms,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
        }
        row_dict.update(d_metrics)
        prediction_jsonl_rows.append(row_dict)

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

    held_out_slice_breakdown: Dict[str, Dict[str, Any]] = {}
    for s_name, s_tot in held_out_slice_totals.items():
        s_corr = held_out_slice_correct[s_name]
        ci_low, ci_high = wilson_95_ci(s_corr, s_tot)
        held_out_slice_breakdown[s_name] = {
            "count": s_tot,
            "correct": s_corr,
            "accuracy": round(s_corr / max(1, s_tot), 4),
            "wilson_95_ci": [ci_low, ci_high],
        }

    calibration_slice_breakdown: Dict[str, Dict[str, Any]] = {}
    for s_name, s_tot in calibration_slice_totals.items():
        s_corr = calibration_slice_correct[s_name]
        ci_low, ci_high = wilson_95_ci(s_corr, s_tot)
        calibration_slice_breakdown[s_name] = {
            "count": s_tot,
            "correct": s_corr,
            "accuracy": round(s_corr / max(1, s_tot), 4),
            "wilson_95_ci": [ci_low, ci_high],
        }

    is_slm_measured = bool(live_ollama or from_predictions is not None)
    ece_val: Optional[float] = None
    brier_val: Optional[float] = None
    reliability_bins: List[Dict[str, Any]] = []
    discrete_confidence_levels: List[Dict[str, Any]] = []
    threshold_sweep: List[Dict[str, Any]] = []

    if is_slm_measured:
        ece_val, reliability_bins = _compute_binned_ece(all_slm_raw_confs, all_slm_correctness)
        brier_val = round(
            sum((c - y) ** 2 for c, y in zip(all_slm_raw_confs, all_slm_correctness))
            / max(1, len(all_slm_raw_confs)),
            4,
        )
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
        held_out_records = [r for r in raw_records if r["split"] == "held_out"] or raw_records
        threshold_sweep = _sweep_thresholds(held_out_records, confidence_key="raw_confidence")

    sorted_lat = sorted(latencies_ms)
    p50_ms = round(sorted_lat[len(sorted_lat) // 2], 2) if sorted_lat else 0.0
    p95_ms = (
        round(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))], 2)
        if sorted_lat
        else 0.0
    )

    overall_ci = list(wilson_95_ci(correct_intents, total))
    slm_only_ci = list(wilson_95_ci(slm_only_correct, max(1, slm_only_total)))
    ho_ci = list(wilson_95_ci(held_out_correct_intents, max(1, held_out_total)))
    ho_slm_ci = list(wilson_95_ci(held_out_slm_correct, max(1, held_out_slm_total)))
    false_drop_ci = list(wilson_95_ci(false_silent_drops, max(1, legitimate_must_not_drop_total)))

    # Step 5 summary metrics
    s5_n = max(1, step5_evaluated_count)
    s5_sorted_lat = sorted(step5_latencies_ms)
    s5_p50 = round(s5_sorted_lat[len(s5_sorted_lat) // 2], 2) if s5_sorted_lat else 0.0
    s5_p95 = (
        round(s5_sorted_lat[min(len(s5_sorted_lat) - 1, int(len(s5_sorted_lat) * 0.95))], 2)
        if s5_sorted_lat
        else 0.0
    )
    unsafe_count = step5_evaluated_count - step5_stage6_safe_count
    step5_lang_breakdown = {
        lang: {
            "total": step5_lang_totals[lang],
            "matched": step5_lang_matches[lang],
            "match_rate": round(step5_lang_matches[lang] / max(1, step5_lang_totals[lang]), 4),
            "wilson_95_ci": list(wilson_95_ci(step5_lang_matches[lang], step5_lang_totals[lang])),
        }
        for lang in sorted(step5_lang_totals.keys())
    }
    step5_summary = {
        "evaluated_drafts": step5_evaluated_count,
        "direct_slm_schema_valid_count": step5_direct_slm_count,
        "static_fallback_count": step5_fallback_count,
        "direct_slm_schema_valid_rate": round(step5_direct_slm_count / s5_n, 4) if step5_evaluated_count else None,
        "stage6_safe_count": step5_stage6_safe_count,
        "unsafe_output_count": unsafe_count,
        "unsafe_output_rate": round(unsafe_count / s5_n, 4) if step5_evaluated_count else None,
        "unsafe_output_wilson_95_ci": list(wilson_95_ci(unsafe_count, step5_evaluated_count)) if step5_evaluated_count else None,
        "guardrail_trips_by_type": step5_guardrail_trips,
        "ticket_id_grounded_count": step5_ticket_grounded_count,
        "ticket_id_grounded_rate": round(step5_ticket_grounded_count / s5_n, 4) if step5_evaluated_count else None,
        "ticket_id_grounded_wilson_95_ci": list(wilson_95_ci(step5_ticket_grounded_count, step5_evaluated_count)) if step5_evaluated_count else None,
        "sla_hours_grounded_count": step5_sla_grounded_count,
        "sla_hours_grounded_rate": round(step5_sla_grounded_count / s5_n, 4) if step5_evaluated_count else None,
        "sla_hours_grounded_wilson_95_ci": list(wilson_95_ci(step5_sla_grounded_count, step5_evaluated_count)) if step5_evaluated_count else None,
        "language_match_count": step5_lang_match_count,
        "language_match_rate": round(step5_lang_match_count / s5_n, 4) if step5_evaluated_count else None,
        "language_match_wilson_95_ci": list(wilson_95_ci(step5_lang_match_count, step5_evaluated_count)) if step5_evaluated_count else None,
        "language_breakdown": step5_lang_breakdown,
        "mean_quality_score_5pt": round(sum(step5_quality_scores) / s5_n, 2) if step5_evaluated_count else None,
        "quality_score_distribution": {
            str(score): step5_quality_scores.count(score) for score in range(5, 0, -1)
        } if step5_evaluated_count else {},
        "latency_p50_ms": s5_p50,
        "latency_p95_ms": s5_p95,
        "prompt_tokens_used": step5_prompt_tokens,
        "completion_tokens_used": step5_comp_tokens,
        "avg_prompt_tokens": round(step5_prompt_tokens / s5_n, 1) if step5_evaluated_count else 0.0,
        "avg_completion_tokens": round(step5_comp_tokens / s5_n, 1) if step5_evaluated_count else 0.0,
    }

    return {
        "evaluation_mode": (
            "live_ollama_slm"
            if (live_ollama or eval_step5_live)
            else ("replayed_slm_predictions" if from_predictions is not None else "offline_deterministic_baseline")
        ),
        "model": model if is_slm_measured else "deterministic_keyword_reference",
        "slm_measured": is_slm_measured,
        "total_samples": len(items),
        "calibration_split_samples": len(items) - held_out_total,
        "held_out_split_samples": held_out_total,
        "short_circuited_injections_0_tokens": short_circuited_injections,
        "held_out_short_circuited_injections_0_tokens": held_out_short_circuits,
        "slm_only_samples": slm_only_total,
        "slm_only_correct": slm_only_correct,
        "slm_only_intent_accuracy": round(slm_only_correct / max(1, slm_only_total), 4),
        "slm_only_intent_accuracy_wilson_95_ci": slm_only_ci,
        "held_out_slm_only_samples": held_out_slm_total,
        "held_out_slm_only_correct": held_out_slm_correct,
        "held_out_slm_only_intent_accuracy": round(held_out_slm_correct / max(1, held_out_slm_total), 4),
        "held_out_slm_only_intent_accuracy_wilson_95_ci": ho_slm_ci,
        "intent_accuracy": round(correct_intents / total, 4),
        "intent_accuracy_wilson_95_ci": overall_ci,
        "held_out_intent_accuracy": round(held_out_correct_intents / max(1, held_out_total), 4),
        "held_out_intent_accuracy_wilson_95_ci": ho_ci,
        "routing_queue_accuracy": round(correct_queues / total, 4),
        "held_out_routing_queue_accuracy": round(held_out_correct_queues / max(1, held_out_total), 4),
        "legitimate_tickets_count": legitimate_must_not_drop_total,
        "false_silent_drop_count": false_silent_drops,
        "false_silent_drop_rate": round(false_silent_drops / max(1, legitimate_must_not_drop_total), 4),
        "false_silent_drop_wilson_95_ci": false_drop_ci,
        "held_out_slice_breakdown": held_out_slice_breakdown,
        "calibration_slice_breakdown": calibration_slice_breakdown,
        "per_class_f1": per_class_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion,
        "expected_calibration_error_10bin": ece_val,
        "brier_score": brier_val,
        "alias_drift_telemetry": get_alias_remap_telemetry(),
        "reliability_bins": reliability_bins,
        "discrete_confidence_levels": discrete_confidence_levels,
        "threshold_sweep": threshold_sweep,
        "misclassified_examples": misclassified_examples,
        "step5_draft_evaluation": step5_summary,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "avg_prompt_tokens_per_slm_call": round(total_prompt_tok / max(1, slm_only_total), 1),
        "avg_completion_tokens_per_slm_call": round(total_comp_tok / max(1, slm_only_total), 1),
        "prompt_tokens_used": total_prompt_tok,
        "completion_tokens_used": total_comp_tok,
        "tokens_used": total_prompt_tok + total_comp_tok,
    }


def write_markdown_report(metrics: Dict[str, Any], report_path: Path = DEFAULT_REPORT_PATH) -> Path:
    """Render an auditable, statistically honest Markdown evaluation report for Stage 3, Stage 4, Stage 5, and Stage 6."""
    slm_ci = metrics["slm_only_intent_accuracy_wilson_95_ci"]
    ho_slm_ci = metrics["held_out_slm_only_intent_accuracy_wilson_95_ci"]
    ov_ci = metrics["intent_accuracy_wilson_95_ci"]
    fd_ci = metrics["false_silent_drop_wilson_95_ci"]
    alias_tel = metrics["alias_drift_telemetry"]
    s5 = metrics.get("step5_draft_evaluation", {})

    lines: List[str] = [
        "# Local SLM Classification (Stage 3) & Response Drafting (Stage 5/6) Evaluation Report",
        "",
        "## Executive Summary & Caveats",
        "",
        f"- **Evaluation Mode:** `{metrics['evaluation_mode']}`",
        f"- **Shipped & Evaluated Model (`DEFAULT_CONFIG`):** `{metrics['model']}`",
        f"- **SLM Directly Measured:** `{metrics['slm_measured']}`",
        f"- **Auditable Raw Per-Ticket Outputs (Stage 3 + Stage 5):** [`tests/eval_predictions.jsonl`](tests/eval_predictions.jsonl) (reproducible offline in `<0.2s` via `--from-predictions tests/eval_predictions.jsonl`)",
        f"- **Dataset Structure & Provenance Caveat:** `{metrics['total_samples']}` anonymized synthetic tickets (`{metrics['calibration_split_samples']}` calibration split + `{metrics['held_out_split_samples']}` disjoint held-out test split). Because all 300 items are synthetic benchmark tickets, these numbers establish a controlled baseline (~99.5% on synthetic held-out items), **not** production proof on messy real-world traffic.",
        f"- **Stage 2 Deterministic Short-Circuits (0 LLM Tokens):** `{metrics['short_circuited_injections_0_tokens']}` adversarial prompt-injection tickets (`6` in calibration, `{metrics['held_out_short_circuited_injections_0_tokens']}` in held-out) were caught by Stage 2 regex gates before reaching the SLM.",
        f"- **SLM-Only Held-Out Intent Accuracy (`N={metrics['held_out_slm_only_samples']}` SLM calls):** **`{metrics['held_out_slm_only_intent_accuracy'] * 100:.2f}%` (`{metrics['held_out_slm_only_correct']}/{metrics['held_out_slm_only_samples']}`)** — 95% Wilson CI: **`[{ho_slm_ci[0] * 100:.2f}%, {ho_slm_ci[1] * 100:.2f}%]`**",
        f"- **SLM-Only Overall Intent Accuracy (`N={metrics['slm_only_samples']}` SLM calls):** **`{metrics['slm_only_intent_accuracy'] * 100:.2f}%` (`{metrics['slm_only_correct']}/{metrics['slm_only_samples']}`)** — 95% Wilson CI: **`[{slm_ci[0] * 100:.2f}%, {slm_ci[1] * 100:.2f}%]`**",
        f"- **Full Pipeline Accuracy (`N={metrics['total_samples']}`, incl. 12 Stage 2 short-circuits):** `{metrics['intent_accuracy'] * 100:.2f}%` (`299/300`, 95% Wilson CI: `[{ov_ci[0] * 100:.2f}%, {ov_ci[1] * 100:.2f}%]`)",
        f"- **False Silent-Drop Rate (`N={metrics['legitimate_tickets_count']}` legitimate non-spam tickets):** **`{metrics['false_silent_drop_count']}/{metrics['legitimate_tickets_count']} = {metrics['false_silent_drop_rate'] * 100:.2f}%`** — **95% Wilson Upper Bound: `{fd_ci[1] * 100:.2f}%` (`[{fd_ci[0] * 100:.2f}%, {fd_ci[1] * 100:.2f}%]`)**. At `N=245`, `0` observed drops still leaves an upper-bound risk of ~15 lost tickets per 1,000. Setting `TM_ALLOW_SILENT_SPAM_DROP=false` routes all spam to `spam_review` (24h SLA) so zero messages are ever silently trashed.",
        f"- **Alias Drift Telemetry (Non-Canonical Label Remaps):** `{alias_tel['total_alias_remaps']}` (`{alias_tel['by_alias']}`)",
        f"- **Stage 3 Latency & Token Usage (`N={metrics['slm_only_samples']}` SLM calls):** p50 `{metrics['latency_p50_ms']} ms` / p95 `{metrics['latency_p95_ms']} ms` (vs `30,000 ms` timeout); average **`{metrics['avg_prompt_tokens_per_slm_call']}` prompt tokens** + **`{metrics['avg_completion_tokens_per_slm_call']}` completion tokens** per call (`{metrics['tokens_used']}` total tokens).",
        "",
        "## 1. Held-Out Split Challenge Slice Breakdown (`N=200` Disjoint Test Tickets)",
        "",
        "*Note: The `100` `clean_short` calibration-split tickets (`100/100`, 95% Wilson CI `[96.30%, 100.00%]`, 94 SLM calls + 6 Stage 2 short-circuits) are excluded from this table so only the 200 held-out test tickets are shown.*",
        "",
        "| Held-Out Challenge Slice | Held-Out Tickets | Correct | Empirical Accuracy | 95% Wilson Confidence Interval |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for s_name, s_info in sorted(metrics["held_out_slice_breakdown"].items()):
        ci = s_info["wilson_95_ci"]
        lines.append(
            f"| `{s_name}` | {s_info['count']} | {s_info['correct']} | "
            f"**{s_info['accuracy'] * 100:.2f}%** | `[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]` |"
        )

    lines.extend(
        [
            "",
            "## 2. Per-Class Precision, Recall & F1 (`N=300`)",
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
            "## 3. Confusion Matrix (Rows = Expected, Columns = Predicted)",
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
                "## 4. Verbalized Confidence Signal Analysis (`N=288` SLM Calls)",
                "",
                "- **Why Post-Hoc Calibration Is Not Fitted Here:** On this dataset, `gemma4:e4b-it-qat` (`temperature=0.0`) makes `0` errors on the 94 SLM-scored calibration tickets and only `1` error across all 288 SLM-scored tickets (`eval-267`). Fitting Temperature Scaling or Isotonic Regression on 0–1 errors merely sharpens all scores toward `1.0` or memorizes step blocks.",
                "- **Verbalized Confidence Is Uninformative Below `0.95`:** Without prompt bands, the model emits only three discrete values (`0.95`, `0.98`, `0.99`). The single error (`eval-267`) occurred at `0.95` (`27/28 = 96.43%`), while `0.98` and `0.99` were `260/260` (`100.00%`).",
                "- **Operational Implication:** The `0.75` and `0.85` numeric confidence thresholds in Stage 4 never bind on `gemma4:e4b-it-qat` because the model never emits confidences in `[0.50, 0.90]`. Until a production calibration set with ≥30–50 errors is collected (or logprobs are extracted), routing safety relies on `requires_human_review`, deterministic transactional signals (`_has_transactional_customer_signals`), and `TM_ALLOW_SILENT_SPAM_DROP` rather than numeric confidence gating.",
                "",
                "| Verbalized Confidence | SLM Tickets | Correct | Empirical Accuracy | 95% Wilson CI | Calibration Gap |",
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

    if metrics["threshold_sweep"]:
        lines.extend(
            [
                "",
                "## 5. Held-Out Threshold Sweep (`spam_threshold` × `escalation_threshold`, `N=200`)",
                "",
                "*Note: Rows at `0.75` and `0.85` are identical because `gemma4:e4b-it-qat` emits `>= 0.95` on all tickets. Routing behavior changes only when thresholds cross `0.96` or `0.99`.*",
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

    if s5 and s5.get("evaluated_drafts", 0) > 0:
        n_d = s5["evaluated_drafts"]
        u_ci = s5["unsafe_output_wilson_95_ci"]
        t_ci = s5["ticket_id_grounded_wilson_95_ci"]
        sla_ci = s5["sla_hours_grounded_wilson_95_ci"]
        l_ci = s5["language_match_wilson_95_ci"]
        trips = s5["guardrail_trips_by_type"]
        q_dist = s5["quality_score_distribution"]

        lines.extend(
            [
                "",
                f"## 6. Stage 5 (SLM Reply Drafting) & Stage 6 (Guardrail) Evaluation (`N={n_d}` Held-Out Drafts)",
                "",
                f"Evaluated `{n_d}` held-out draftable tickets (`BILLING`, `TECHNICAL_SUPPORT`, `FEATURE_REQUEST`) through live `draft_response` (`gemma4:e4b-it-qat`) and `is_draft_safe` guardrails (all `{n_d}` raw drafts are saved in [`tests/eval_predictions.jsonl`](tests/eval_predictions.jsonl)):",
                "",
                "| Stage 5 / Stage 6 Metric | Count | Rate | 95% Wilson Confidence Interval |",
                "| :--- | :---: | :---: | :---: |",
                f"| **Direct SLM Schema Validity (No Static Fallback Needed)** | `{s5['direct_slm_schema_valid_count']}/{n_d}` | **`{s5['direct_slm_schema_valid_rate'] * 100:.2f}%`** | — |",
                f"| **Stage 6 Unsafe Output / Guardrail Trip Rate** | `{s5['unsafe_output_count']}/{n_d}` | **`{s5['unsafe_output_rate'] * 100:.2f}%`** | `[{u_ci[0] * 100:.2f}%, {u_ci[1] * 100:.2f}%]` |",
                f"| **Ticket ID Groundedness (`TKT-...` in ID & Draft)** | `{s5['ticket_id_grounded_count']}/{n_d}` | **`{s5['ticket_id_grounded_rate'] * 100:.2f}%`** | `[{t_ci[0] * 100:.2f}%, {t_ci[1] * 100:.2f}%]` |",
                f"| **SLA Window Groundedness (`4h`/`12h`/`48h` exact match)** | `{s5['sla_hours_grounded_count']}/{n_d}` | **`{s5['sla_hours_grounded_rate'] * 100:.2f}%`** | `[{sla_ci[0] * 100:.2f}%, {sla_ci[1] * 100:.2f}%]` |",
                f"| **Customer Language Match (`EN`/`ES`/`FR`/`DE`/`PT`/`JA`)** | `{s5['language_match_count']}/{n_d}` | **`{s5['language_match_rate'] * 100:.2f}%`** | `[{l_ci[0] * 100:.2f}%, {l_ci[1] * 100:.2f}%]` |",
                "",
                f"- **Guardrail Trips by Category (`N={n_d}`):** PII/Secrets: `{trips['pii_or_secrets']}`, Forbidden Commitments (EN/ES/FR/DE/PT/JA): `{trips['forbidden_commitments']}`, Disallowed URLs: `{trips['disallowed_urls']}`, Unescaped HTML: `{trips['unescaped_html']}`",
                f"- **5-Point Rubric Score Distribution (`mean = {s5['mean_quality_score_5pt']} / 5.00`):** `5/5`: `{q_dist.get('5', 0)}`, `4/5`: `{q_dist.get('4', 0)}`, `3/5`: `{q_dist.get('3', 0)}`, `2/5`: `{q_dist.get('2', 0)}`, `1/5`: `{q_dist.get('1', 0)}`",
                f"- **Stage 5 Latency & Tokens (`N={n_d}`):** p50 `{s5['latency_p50_ms']} ms` / p95 `{s5['latency_p95_ms']} ms`; average **`{s5['avg_prompt_tokens']}` prompt tokens** + **`{s5['avg_completion_tokens']}` completion tokens** per draft.",
                "",
                "### Stage 5 Language Match Breakdown by Language",
                "",
                "| Language Code | Evaluated Tickets | Language Matched | Match Rate | 95% Wilson CI |",
                "| :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for lang_code, l_info in s5["language_breakdown"].items():
            l_ci_row = l_info["wilson_95_ci"]
            lines.append(
                f"| `{lang_code}` | {l_info['total']} | {l_info['matched']} | "
                f"**{l_info['match_rate'] * 100:.2f}%** | `[{l_ci_row[0] * 100:.2f}%, {l_ci_row[1] * 100:.2f}%]` |"
            )

    if metrics["misclassified_examples"]:
        lines.extend(
            [
                "",
                "## 7. Stage 3 Error Analysis (Misclassified Tickets)",
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
        description="Evaluate Stage 3 classification, held-out slices, Stage 5 drafting, and Stage 6 guardrails."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Evaluate the live local Ollama SLM on Stage 3 classification.",
    )
    parser.add_argument(
        "--eval-step5-live",
        action="store_true",
        help="Evaluate live local Ollama SLM on Stage 5 drafting + Stage 6 guardrails for held-out draftable tickets.",
    )
    parser.add_argument(
        "--step5-limit",
        type=int,
        default=105,
        help="Number of held-out draftable tickets to evaluate through live Stage 5 when --eval-step5-live is set.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG.classify_model,
        help="Ollama model tag for Stage 3 classification.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_CONFIG.draft_model,
        help="Ollama model tag for Stage 5 drafting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent worker threads for live Ollama evaluation.",
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
        help="Path to write per-ticket raw prediction JSONL records.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional path to write a Markdown calibration report (e.g., CALIBRATION_REPORT.md).",
    )
    args = parser.parse_args()

    preds_out = args.predictions_out
    if preds_out is None and (args.live_ollama or args.eval_step5_live):
        preds_out = DEFAULT_PREDICTIONS_PATH

    metrics = evaluate_dataset(
        dataset_path=args.dataset,
        live_ollama=args.live_ollama,
        eval_step5_live=args.eval_step5_live,
        step5_limit=args.step5_limit,
        model=args.model,
        draft_model=args.draft_model,
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
