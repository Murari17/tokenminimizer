"""Stage 4 [CODE — 0 Tokens]: Deterministic branching, SLA calculation, and routing matrix."""

from __future__ import annotations

from src.schemas import (
    ClassificationOutput,
    CleanedPayload,
    IntentCategory,
    PriorityLevel,
    RuleEvaluationResult,
)


def apply_rules(data: CleanedPayload, cls: ClassificationOutput) -> RuleEvaluationResult:
    """Evaluate deterministic business routing rules with 0 LLM tokens.

    Decision Matrix:
    1. SPAM with confidence >= 0.85 (and not flagged for human review/suspicious):
       should_draft_reply=False, escalate_to_human=False, assigned_queue="trash".
    2. Confidence < 0.75 OR requires_human_review OR SECURITY_REPORT OR suspicious payload:
       should_draft_reply=False, escalate_to_human=True, priority=CRITICAL, sla_hours=1,
       assigned_queue="tier2_human".
    3. BILLING:
       should_draft_reply=True, priority=HIGH, sla_hours=4,
       allowed_tone="empathetic_formal", response_template_id="TPL_BILLING_01".
    4. TECHNICAL_SUPPORT:
       should_draft_reply=True, priority=MEDIUM, sla_hours=12,
       allowed_tone="concise_technical", response_template_id="TPL_TECH_01".
    5. Default (FEATURE_REQUEST):
       should_draft_reply=True, priority=LOW, sla_hours=48,
       allowed_tone="appreciative_brief", response_template_id="TPL_FEAT_01".
    """
    # Branch 1: High-confidence SPAM drop
    if (
        cls.intent == IntentCategory.SPAM
        and cls.confidence >= 0.85
        and not cls.requires_human_review
        and not data.contains_suspicious_patterns
    ):
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

    # Branch 2: Low confidence, human review required, suspicious patterns, or Security Report
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

    # Branch 3: Billing inquiries
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

    # Branch 4: Technical support inquiries
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

    # Branch 5: Default (FEATURE_REQUEST)
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

