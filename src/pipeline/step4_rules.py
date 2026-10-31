"""Stage 4 [CODE — 0 Tokens]: Deterministic branching, corroborated SPAM gating,
escalation flood protection, SLA calculation, and routing matrix.
"""

from __future__ import annotations

import re

from src.schemas import (
    ClassificationOutput,
    CleanedPayload,
    IntentCategory,
    PriorityLevel,
    RuleEvaluationResult,
)

# Deterministic transactional keywords that prevent an uncalibrated SLM from silently
# dropping a legitimate customer ticket as SPAM.
_TRANSACTIONAL_SIGNAL_RE = re.compile(
    r"\b(?:inv-\d+|tkt-\d+|ord-\d+|invoice|charged\s+twice|double\s+charged|"
    r"billing\s+discrepancy|unauthorized\s+charge|2fa|two-factor|outage|"
    r"production\s+down|500\s+error|502\s+bad\s+gateway|cve-\d{4}-\d+|"
    r"security\s+vulnerability|data\s+breach)\b",
    re.IGNORECASE,
)


def _has_transactional_customer_signals(data: CleanedPayload) -> bool:
    """Return True if the message contains strong deterministic customer support/billing signals."""
    combined = f"{data.clean_subject}\n{data.clean_body}"
    return bool(_TRANSACTIONAL_SIGNAL_RE.search(combined))


def apply_rules(data: CleanedPayload, cls: ClassificationOutput) -> RuleEvaluationResult:
    """Evaluate deterministic business routing rules with 0 LLM tokens.

    Decision Matrix:
    1. SPAM with confidence >= 0.85, no human review flag, no suspicious patterns, AND
       no conflicting deterministic customer transaction signals:
       should_draft_reply=False, escalate_to_human=False, assigned_queue="trash".
       (If SPAM conflicts with transactional customer signals, route to "spam_review" for human check.)
    2. Operational SLM unavailability/timeout ("SLM_UNAVAILABLE") without security flags:
       should_draft_reply=False, escalate_to_human=True, priority=MEDIUM, sla_hours=8,
       assigned_queue="tier1_triage" (prevents CRITICAL 1-hour SLA escalation floods on cold CPU).
    3. Confidence < 0.75 OR requires_human_review OR SECURITY_REPORT OR suspicious payload:
       should_draft_reply=False, escalate_to_human=True, priority=CRITICAL, sla_hours=1,
       assigned_queue="tier2_human".
    4. BILLING:
       should_draft_reply=True, priority=HIGH, sla_hours=4,
       allowed_tone="empathetic_formal", response_template_id="TPL_BILLING_01".
    5. TECHNICAL_SUPPORT:
       should_draft_reply=True, priority=MEDIUM, sla_hours=12,
       allowed_tone="concise_technical", response_template_id="TPL_TECH_01".
    6. Default (FEATURE_REQUEST):
       should_draft_reply=True, priority=LOW, sla_hours=48,
       allowed_tone="appreciative_brief", response_template_id="TPL_FEAT_01".
    """
    # Branch 1: High-confidence SPAM with deterministic corroboration guard
    if (
        cls.intent == IntentCategory.SPAM
        and cls.confidence >= 0.85
        and not cls.requires_human_review
        and not data.contains_suspicious_patterns
    ):
        if _has_transactional_customer_signals(data):
            return RuleEvaluationResult(
                event_id=data.event_id,
                should_draft_reply=False,
                escalate_to_human=True,
                priority=PriorityLevel.LOW,
                sla_hours=24,
                assigned_queue="spam_review",
                response_template_id="TPL_NONE",
                allowed_tone="strict_neutral",
            )
        return RuleEvaluationResult(
            event_id=data.event_id,
            should_draft_reply=False,
            escalate_to_human=False,
            priority=PriorityLevel.LOW,
            sla_hours=0,
            assigned_queue="trash",
            response_template_id="TPL_NONE",
            allowed_tone="none",
        )

    # Branch 2: Operational SLM timeout / empty input -> route to tier1_triage (MEDIUM, 8h SLA)
    # to avoid flooding tier2_human with CRITICAL 1-hour pages during local SLM cold starts.
    if (
        not data.contains_suspicious_patterns
        and cls.intent != IntentCategory.SECURITY_REPORT
        and ("SLM_UNAVAILABLE" in cls.detected_entities or "EMPTY_INPUT" in cls.detected_entities)
    ):
        return RuleEvaluationResult(
            event_id=data.event_id,
            should_draft_reply=False,
            escalate_to_human=True,
            priority=PriorityLevel.MEDIUM,
            sla_hours=8,
            assigned_queue="tier1_triage",
            response_template_id="TPL_ESC_TRIAGE",
            allowed_tone="strict_neutral",
        )

    # Branch 3: Low confidence, human review required, suspicious patterns, or Security Report
    if (
        cls.confidence < 0.75
        or cls.requires_human_review
        or data.contains_suspicious_patterns
        or cls.intent == IntentCategory.SECURITY_REPORT
    ):
        return RuleEvaluationResult(
            event_id=data.event_id,
            should_draft_reply=False,
            escalate_to_human=True,
            priority=PriorityLevel.CRITICAL,
            sla_hours=1,
            assigned_queue="tier2_human",
            response_template_id="TPL_ESC_SECURITY",
            allowed_tone="strict_neutral",
        )

    # Branch 4: Billing inquiries
    if cls.intent == IntentCategory.BILLING:
        return RuleEvaluationResult(
            event_id=data.event_id,
            should_draft_reply=True,
            escalate_to_human=False,
            priority=PriorityLevel.HIGH,
            sla_hours=4,
            assigned_queue="billing_support",
            response_template_id="TPL_BILLING_01",
            allowed_tone="empathetic_formal",
        )

    # Branch 5: Technical support inquiries
    if cls.intent == IntentCategory.TECHNICAL_SUPPORT:
        return RuleEvaluationResult(
            event_id=data.event_id,
            should_draft_reply=True,
            escalate_to_human=False,
            priority=PriorityLevel.MEDIUM,
            sla_hours=12,
            assigned_queue="tech_support",
            response_template_id="TPL_TECH_01",
            allowed_tone="concise_technical",
        )

    # Branch 6: Default (FEATURE_REQUEST)
    return RuleEvaluationResult(
        event_id=data.event_id,
        should_draft_reply=True,
        escalate_to_human=False,
        priority=PriorityLevel.LOW,
        sla_hours=48,
        assigned_queue="product_feedback",
        response_template_id="TPL_FEAT_01",
        allowed_tone="appreciative_brief",
    )
