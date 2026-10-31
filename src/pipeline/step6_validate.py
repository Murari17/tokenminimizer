"""Stage 6 [CODE — 0 Tokens]: Output guardrails, PII leak check, SQLite audit save, and dispatch."""

from __future__ import annotations

from typing import Iterable, Optional

from src.config import DEFAULT_CONFIG
from src.database import Database
from src.schemas import (
    CleanedPayload,
    DispatchReceipt,
    DispatchStatus,
    DraftOutput,
    RuleEvaluationResult,
)
from src.security import (
    contains_disallowed_urls,
    contains_forbidden_commitments,
    contains_pii_or_secrets,
    contains_unescaped_html,
)


def is_draft_safe(
    draft: DraftOutput,
    allowed_domains: Iterable[str] = DEFAULT_CONFIG.allowed_domains,
) -> bool:
    """Run deterministic security and policy guardrails on an outbound draft.

    Rejects if `subject_line` or `body_text` contains:
    - Leaked PII/secrets (Luhn-valid credit card numbers, SSNs, API keys, private IPs)
    - Forbidden commitment phrases ("we guarantee a refund", "lawsuit", etc.)
    - URLs not on the explicit allowlist
    - Unescaped HTML / script tags
    """
    combined = f"{draft.subject_line}\n{draft.body_text}"
    if contains_pii_or_secrets(combined):
        return False
    if contains_forbidden_commitments(combined):
        return False
    if contains_unescaped_html(combined):
        return False
    if contains_disallowed_urls(combined, allowed_domains):
        return False
    return True


def validate_save_send(
    data: CleanedPayload,
    rules: RuleEvaluationResult,
    draft: Optional[DraftOutput],
    db: Database,
    tokens_used: int = 0,
    allowed_domains: Iterable[str] = DEFAULT_CONFIG.allowed_domains,
) -> DispatchReceipt:
    """Validate outbound draft, record audit trace in SQLite, and return `DispatchReceipt`.

    Tokens consumed: 0.
    """
    escalate_to_human = rules.escalate_to_human
    final_payload: Optional[DraftOutput] = None

    if draft is None:
        if rules.assigned_queue == "trash" and not rules.escalate_to_human:
            status = DispatchStatus.DROPPED_SPAM
        else:
            status = DispatchStatus.ESCALATED
            escalate_to_human = True
    else:
        if is_draft_safe(draft, allowed_domains=allowed_domains):
            status = DispatchStatus.SENT
            final_payload = draft
        else:
            status = DispatchStatus.BLOCKED_VALIDATION
            escalate_to_human = True
            final_payload = None

    audit_log_id = db.record_dispatch(
        data=data,
        rules=rules,
        status=status,
        draft=final_payload,
        escalate_to_human=escalate_to_human,
        tokens_used=tokens_used,
    )

    return DispatchReceipt(
        event_id=data.event_id,
        status=status,
        final_payload=final_payload,
        audit_log_id=audit_log_id,
        tokens_used=tokens_used,
    )

