"""Unit tests for deterministic 0-token stages: Steps 1, 2, 4, and 6."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from src.database import Database, DuplicateEventError
from src.pipeline.step1_trigger import ReplayAttackError, TriggerIngestionError, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step4_rules import apply_rules
from src.pipeline.step6_validate import InMemoryOutboxTransport, validate_save_send
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
# Stage 1: Trigger Ingestion, HMAC, Secret Enforcement & Replay Window (0 Tokens)
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


def test_step1_hard_fails_when_secret_is_unset_or_weak(make_signed_trigger) -> None:
    raw = make_signed_trigger()
    with pytest.raises(TriggerIngestionError, match="Webhook HMAC secret is not configured"):
        ingest_trigger(raw, secret_key="")
    with pytest.raises(TriggerIngestionError, match="at least 16 characters"):
        ingest_trigger(raw, secret_key="short-secret")


def test_step1_tampered_payload_fails_hmac(make_signed_trigger, secret_key: str) -> None:
    raw = make_signed_trigger(raw_body="Original legitimate message.")
    raw["raw_body"] = "Tampered message after signing!"
    with pytest.raises(TriggerIngestionError, match="HMAC-SHA256 signature verification failed"):
        ingest_trigger(raw, secret_key=secret_key)


def test_step1_rejects_year_old_replayed_payload(make_signed_trigger, secret_key: str) -> None:
    year_old_ts = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    raw = make_signed_trigger(timestamp=year_old_ts)
    with pytest.raises(ReplayAttackError, match="Stale payload rejected"):
        ingest_trigger(raw, secret_key=secret_key)


def test_step1_rejects_future_skewed_payload(make_signed_trigger, secret_key: str) -> None:
    future_ts = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    raw = make_signed_trigger(timestamp=future_ts)
    with pytest.raises(ReplayAttackError, match="Future-skewed payload rejected"):
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
# Stage 2: HTML Stripping (< 5 and x<y preserved), Normalization & URL Extraction
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


def test_step2_preserves_less_than_comparisons_and_code_snippets(
    make_signed_trigger,
    secret_key: str,
) -> None:
    raw_dict = make_signed_trigger(
        raw_subject="Balance < 5 USD & x<y check",
        raw_body="My balance is < 5 dollars and my card was charged twice when x<y in code.",
    )
    trigger = ingest_trigger(raw_dict, secret_key=secret_key)
    cleaned = fetch_and_clean(trigger)

    assert cleaned.clean_body == (
        "My balance is < 5 dollars and my card was charged twice when x<y in code."
    )
    assert "charged twice" in cleaned.clean_body
    assert "x<y" in cleaned.clean_body


def test_step2_enforces_2000_char_cap(make_signed_trigger, secret_key: str) -> None:
    long_text = ("word " * 800).strip()
    assert len(long_text) > 3000
    raw_dict = make_signed_trigger(raw_body=long_text)
    trigger = ingest_trigger(raw_dict, secret_key=secret_key)
    cleaned = fetch_and_clean(trigger)
    assert len(cleaned.clean_body) <= 2000


# ---------------------------------------------------------------------------
# Stage 4: Deterministic Rules Matrix & Corroborated SPAM / Timeout Guards
# ---------------------------------------------------------------------------
def test_step4_branch1_high_confidence_spam_dropped_when_corroborated() -> None:
    spam_payload = CleanedPayload(
        event_id=uuid4(),
        sender="spammer@promo.biz",
        clean_subject="WIN FREE TOKENS",
        clean_body="Claim your free lottery giveaway prize today!",
        extracted_urls=[],
        word_count=7,
        contains_suspicious_patterns=False,
    )
    cls = ClassificationOutput(
        intent=IntentCategory.SPAM,
        confidence=0.92,
        detected_entities=[],
        requires_human_review=False,
    )
    rules = apply_rules(spam_payload, cls)
    assert rules.should_draft_reply is False
    assert rules.escalate_to_human is False
    assert rules.assigned_queue == "trash"


def test_step4_corroboration_prevents_silent_drop_on_invoice_ticket(
    sample_cleaned_payload: CleanedPayload,
) -> None:
    # sample_cleaned_payload mentions "Billing discrepancy" and "credit card was billed twice"
    cls = ClassificationOutput(
        intent=IntentCategory.SPAM,
        confidence=0.95,
        detected_entities=[],
        requires_human_review=False,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    assert rules.assigned_queue == "spam_review"
    assert rules.escalate_to_human is True
    assert rules.should_draft_reply is False


def test_step4_slm_timeout_routes_to_tier1_triage_not_critical_flood(
    sample_cleaned_payload: CleanedPayload,
) -> None:
    cls = ClassificationOutput(
        intent=IntentCategory.TECHNICAL_SUPPORT,
        confidence=0.0,
        detected_entities=["SLM_UNAVAILABLE"],
        requires_human_review=True,
    )
    rules = apply_rules(sample_cleaned_payload, cls)
    assert rules.assigned_queue == "tier1_triage"
    assert rules.priority == PriorityLevel.MEDIUM
    assert rules.sla_hours == 8
    assert rules.escalate_to_human is True
    assert rules.should_draft_reply is False


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
# Stage 6: Output Validation, Outbox Dispatch & SQLite Replay Protection
# ---------------------------------------------------------------------------
def test_step6_valid_draft_dispatched_to_outbox_and_marked_sent(
    sample_cleaned_payload: CleanedPayload,
    memory_db: Database,
) -> None:
    outbox = InMemoryOutboxTransport()
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
        transport=outbox,
    )
    assert receipt.status == DispatchStatus.SENT
    assert receipt.final_payload == draft
    assert receipt.tokens_used == 115
    assert len(outbox.sent_messages) == 1
    assert outbox.sent_messages[0].recipient == sample_cleaned_payload.sender

    # Attempting to save the same event_id a second time raises DuplicateEventError
    with pytest.raises(DuplicateEventError, match="Replay detected"):
        validate_save_send(
            data=sample_cleaned_payload,
            rules=rules,
            draft=draft,
            db=memory_db,
            tokens_used=115,
            transport=outbox,
        )


def test_step6_blocks_forbidden_commitment_and_unallowed_url(
    sample_cleaned_payload: CleanedPayload,
    memory_db: Database,
) -> None:
    outbox = InMemoryOutboxTransport()
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
        transport=outbox,
    )
    assert receipt.status == DispatchStatus.BLOCKED_VALIDATION
    assert receipt.final_payload is None
    assert len(outbox.sent_messages) == 0


def test_step4_disable_silent_spam_drop_routes_to_spam_review(
    sample_cleaned_payload: CleanedPayload,
) -> None:
    from uuid import uuid4

    benign_looking_spam = CleanedPayload(
        event_id=uuid4(),
        sender="promo@agency.biz",
        clean_subject="SEO ranking offer",
        clean_body="Buy 500 backlinks today for $99.",
        extracted_urls=[],
        word_count=7,
        contains_suspicious_patterns=False,
    )
    cls = ClassificationOutput(
        intent=IntentCategory.SPAM,
        confidence=0.98,
        detected_entities=[],
        requires_human_review=False,
    )
    rules_no_drop = apply_rules(benign_looking_spam, cls, allow_silent_spam_drop=False)
    assert rules_no_drop.assigned_queue == "spam_review"
    assert rules_no_drop.escalate_to_human is True
    assert rules_no_drop.should_draft_reply is False


def test_step5_multilingual_static_fallback_matches_customer_language() -> None:
    from uuid import uuid4
    from src.pipeline.step5_draft import detect_language, render_static_fallback_draft

    es_payload = CleanedPayload(
        event_id=uuid4(),
        sender="cliente@empresa.es",
        clean_subject="Factura duplicada",
        clean_body="Hola equipo, nos han cobrado dos veces la factura en nuestra tarjeta.",
        extracted_urls=[],
        word_count=12,
        contains_suspicious_patterns=False,
    )
    cls = ClassificationOutput(
        intent=IntentCategory.BILLING,
        confidence=0.95,
        detected_entities=[],
        requires_human_review=False,
    )
    rules = apply_rules(es_payload, cls)
    fallback_draft = render_static_fallback_draft(es_payload, rules)
    assert detect_language(fallback_draft.body_text) == "es"
    assert str(rules.sla_hours) in fallback_draft.body_text
    assert fallback_draft.referenced_ticket_id in fallback_draft.body_text
