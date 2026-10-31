"""Unit tests for deterministic 0-token stages: Steps 1, 2, 4, and 6."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.database import Database
from src.pipeline.step1_trigger import TriggerIngestionError, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step4_rules import apply_rules
from src.pipeline.step6_validate import validate_save_send
from src.schemas import (
    ClassificationOutput,
    CleanedPayload,
    DispatchStatus,
    DraftOutput,
    IntentCategory,
    PriorityLevel,
    SourceType,
)


# ---------------------------------------------------------------------------
# Stage 1: Trigger Ingestion & Signature Verification (0 Tokens)
# ---------------------------------------------------------------------------
def test_step1_valid_hmac_signature_passes(make_signed_trigger, secret_key: str) -> None:
    raw = make_signed_trigger(
        sender="billing.user@client.io",
        raw_subject="Invoice inquiry",
        raw_body="Please send a copy of my latest receipt.",
    )
    result = ingest_trigger(raw, secret_key=secret_key)
    assert result.sender == "billing.user@client.io"
    assert result.source == SourceType.EMAIL
    assert str(result.event_id) == raw["event_id"]


def test_step1_tampered_payload_fails_hmac(make_signed_trigger, secret_key: str) -> None:
    raw = make_signed_trigger(raw_body="Original legitimate message.")
    raw["raw_body"] = "Tampered message after signing!"
    with pytest.raises(TriggerIngestionError, match="HMAC-SHA256 signature verification failed"):
        ingest_trigger(raw, secret_key=secret_key)


def test_step1_rejects_payload_exceeding_50kb(make_signed_trigger, secret_key: str) -> None:
    oversized_body = "A" * (51 * 1024)
    raw = make_signed_trigger(raw_body=oversized_body)
    with pytest.raises(TriggerIngestionError, match="exceeds maximum limit"):
        ingest_trigger(raw, secret_key=secret_key)


def test_step1_rejects_missing_required_fields(secret_key: str) -> None:
    incomplete = {"event_id": str(uuid4()), "source": "EMAIL"}
    with pytest.raises(TriggerIngestionError, match="Missing required trigger fields"):
        ingest_trigger(incomplete, secret_key=secret_key)


# ---------------------------------------------------------------------------
# Stage 2: HTML Stripping, Normalization & URL Extraction (0 Tokens)
# ---------------------------------------------------------------------------
def test_step2_strips_html_and_normalizes_unicode(make_signed_trigger, secret_key: str) -> None:
    raw_dict = make_signed_trigger(
        raw_subject="  <b>Urgent</b>   Support\u00a0Request  ",
        raw_body=(
            "<div>Hello   <b>world</b>!<style>body{display:none}</style>"
            "<script>evil()</script><p>Visit https://status.example.com/uptime "
            "and https://docs.example.com/guide.</p></div>"
        ),
    )
    trigger = ingest_trigger(raw_dict, secret_key=secret_key)
    cleaned = fetch_and_clean(trigger)

    assert cleaned.clean_subject == "Urgent Support Request"
    assert "<div>" not in cleaned.clean_body
    assert "evil()" not in cleaned.clean_body
    assert "display:none" not in cleaned.clean_body
    assert "Hello world !" in cleaned.clean_body or "Hello world!" in cleaned.clean_body
    assert cleaned.extracted_urls == [
        "https://status.example.com/uptime",
        "https://docs.example.com/guide",
    ]
    assert cleaned.word_count > 0
    assert cleaned.contains_suspicious_patterns is False


def test_step2_enforces_2000_char_cap(make_signed_trigger, secret_key: str) -> None:
    long_text = ("word " * 800).strip()
    assert len(long_text) > 3000
    raw_dict = make_signed_trigger(raw_body=long_text)
    trigger = ingest_trigger(raw_dict, secret_key=secret_key)
    cleaned = fetch_and_clean(trigger)
    assert len(cleaned.clean_body) <= 2000


# ---------------------------------------------------------------------------
# Stage 4: Deterministic Rules Matrix (All 5 Branches, 0 Tokens)
# ---------------------------------------------------------------------------
def test_step4_branch1_high_confidence_spam_dropped(sample_cleaned_payload: CleanedPayload) -> None:
    cls = ClassificationOutput(
        intent=IntentCategory.SPAM,
        confidence=0.92,
        detected_entities=[],
        requires_human_review=False,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    assert rules.should_draft_reply is False
    assert rules.escalate_to_human is False
    assert rules.assigned_queue == "trash"


@pytest.mark.parametrize(
    "intent,confidence,human_review",
    [
        (IntentCategory.BILLING, 0.60, False),          # Low confidence (< 0.75)
        (IntentCategory.TECHNICAL_SUPPORT, 0.95, True), # Explicit human review flag
        (IntentCategory.SECURITY_REPORT, 0.99, False),  # Security report always escalates
    ],
)
def test_step4_branch2_escalation_conditions(
    sample_cleaned_payload: CleanedPayload,
    intent: IntentCategory,
    confidence: float,
    human_review: bool,
) -> None:
    cls = ClassificationOutput(
        intent=intent,
        confidence=confidence,
        detected_entities=["cve-2026"],
        requires_human_review=human_review,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    assert rules.should_draft_reply is False
    assert rules.escalate_to_human is True
    assert rules.priority == PriorityLevel.CRITICAL
    assert rules.sla_hours == 1
    assert rules.assigned_queue == "tier2_human"


def test_step4_branch3_billing_routing(sample_cleaned_payload: CleanedPayload) -> None:
    cls = ClassificationOutput(
        intent=IntentCategory.BILLING,
        confidence=0.91,
        detected_entities=["INV-100"],
        requires_human_review=False,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    assert rules.should_draft_reply is True
    assert rules.escalate_to_human is False
    assert rules.priority == PriorityLevel.HIGH
    assert rules.sla_hours == 4
    assert rules.allowed_tone == "empathetic_formal"
    assert rules.response_template_id == "TPL_BILLING_01"


def test_step4_branch4_technical_support_routing(sample_cleaned_payload: CleanedPayload) -> None:
    cls = ClassificationOutput(
        intent=IntentCategory.TECHNICAL_SUPPORT,
        confidence=0.88,
        detected_entities=["API_502"],
        requires_human_review=False,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    assert rules.should_draft_reply is True
    assert rules.escalate_to_human is False
    assert rules.priority == PriorityLevel.MEDIUM
    assert rules.sla_hours == 12
    assert rules.allowed_tone == "concise_technical"
    assert rules.response_template_id == "TPL_TECH_01"


def test_step4_branch5_feature_request_default(sample_cleaned_payload: CleanedPayload) -> None:
    cls = ClassificationOutput(
        intent=IntentCategory.FEATURE_REQUEST,
        confidence=0.84,
        detected_entities=["dark_mode"],
        requires_human_review=False,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    assert rules.should_draft_reply is True
    assert rules.escalate_to_human is False
    assert rules.priority == PriorityLevel.LOW
    assert rules.sla_hours == 48
    assert rules.allowed_tone == "appreciative_brief"
    assert rules.response_template_id == "TPL_FEAT_01"


# ---------------------------------------------------------------------------
# Stage 6: Output Validation & SQLite Audit Persistence (0 Tokens)
# ---------------------------------------------------------------------------
def test_step6_valid_draft_saved_and_marked_sent(
    sample_cleaned_payload: CleanedPayload,
    memory_db: Database,
) -> None:
    cls = ClassificationOutput(
        intent=IntentCategory.BILLING,
        confidence=0.90,
        detected_entities=[],
        requires_human_review=False,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    draft = DraftOutput(
        subject_line="Re: Billing discrepancy [TKT-1001]",
        body_text=(
            "We have logged your billing inquiry under TKT-1001 and will respond within 4 hours. "
            "See https://billing.example.com for details."
        ),
        referenced_ticket_id="TKT-1001",
    )
    receipt = validate_save_send(
        data=sample_cleaned_payload,
        rules=rules,
        draft=draft,
        db=memory_db,
        tokens_used=115,
    )
    assert receipt.status == DispatchStatus.SENT
    assert receipt.final_payload == draft
    assert receipt.tokens_used == 115

    stored = memory_db.get_audit_log(receipt.audit_log_id)
    assert stored is not None
    assert stored["status"] == "SENT"
    assert stored["tokens_used"] == 115
    assert stored["escalate_to_human"] == 0


def test_step6_blocks_forbidden_commitment_and_unallowed_url(
    sample_cleaned_payload: CleanedPayload,
    memory_db: Database,
) -> None:
    cls = ClassificationOutput(
        intent=IntentCategory.BILLING,
        confidence=0.95,
        detected_entities=[],
        requires_human_review=False,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    unsafe_draft = DraftOutput(
        subject_line="Refund Confirmation",
        body_text="We guarantee a refund immediately! Click https://phishing-site.evil.org/claim",
        referenced_ticket_id="TKT-9999",
    )
    receipt = validate_save_send(
        data=sample_cleaned_payload,
        rules=rules,
        draft=unsafe_draft,
        db=memory_db,
        tokens_used=88,
    )
    assert receipt.status == DispatchStatus.BLOCKED_VALIDATION
    assert receipt.final_payload is None

    stored = memory_db.get_audit_log(receipt.audit_log_id)
    assert stored is not None
    assert stored["status"] == "BLOCKED_VALIDATION"
    assert stored["escalate_to_human"] == 1

