"""Stage 6 [CODE — 0 Tokens]: Output guardrails, PII leak check, fencing-token CAS
transition (`PENDING` -> `DISPATCHING` -> `SENT`), and outbound delivery transport.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Iterable, Optional, Protocol
from uuid import UUID

from src.config import DEFAULT_CONFIG
from src.database import Database, LeaseReservation, StaleLeaseError
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


@dataclass(frozen=True)
class DispatchedMessage:
    """Immutable record of an outbound message delivered via a DispatchTransport."""

    event_id: UUID
    recipient: str
    subject_line: str
    body_text: str
    referenced_ticket_id: str
    dispatched_at: str


class DispatchTransport(Protocol):
    """Protocol for outbound message delivery (in-memory outbox, webhook, or SMTP)."""

    def send(self, data: CleanedPayload, draft: DraftOutput) -> DispatchedMessage:
        ...


class InMemoryOutboxTransport:
    """Thread-safe, memory-bounded outbox transport that records dispatched replies."""

    def __init__(self, max_items: int = 1000) -> None:
        self._lock = threading.Lock()
        self._outbox: Deque[DispatchedMessage] = deque(maxlen=max_items)

    def send(self, data: CleanedPayload, draft: DraftOutput) -> DispatchedMessage:
        msg = DispatchedMessage(
            event_id=data.event_id,
            recipient=data.sender,
            subject_line=draft.subject_line,
            body_text=draft.body_text,
            referenced_ticket_id=draft.referenced_ticket_id,
            dispatched_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._outbox.append(msg)
        return msg

    @property
    def sent_messages(self) -> tuple[DispatchedMessage, ...]:
        with self._lock:
            return tuple(self._outbox)

    def clear(self) -> None:
        with self._lock:
            self._outbox.clear()


class WebhookDispatchTransport:
    """Zero-dependency HTTP POST webhook dispatcher for outbound replies."""

    def __init__(self, webhook_url: str, timeout_seconds: float = 5.0) -> None:
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds

    def send(self, data: CleanedPayload, draft: DraftOutput) -> DispatchedMessage:
        dispatched_at = datetime.now(timezone.utc).isoformat()
        msg = DispatchedMessage(
            event_id=data.event_id,
            recipient=data.sender,
            subject_line=draft.subject_line,
            body_text=draft.body_text,
            referenced_ticket_id=draft.referenced_ticket_id,
            dispatched_at=dispatched_at,
        )
        payload_bytes = json.dumps(
            {
                "event_id": str(msg.event_id),
                "recipient": msg.recipient,
                "subject_line": msg.subject_line,
                "body_text": msg.body_text,
                "referenced_ticket_id": msg.referenced_ticket_id,
                "dispatched_at": msg.dispatched_at,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url=self.webhook_url,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": str(msg.event_id),
                "X-Event-ID": str(msg.event_id),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Webhook dispatch failed with HTTP status {resp.status}")
        return msg


DEFAULT_OUTBOX_TRANSPORT = InMemoryOutboxTransport(max_items=1000)


def is_draft_safe(
    draft: DraftOutput,
    allowed_domains: Iterable[str] = DEFAULT_CONFIG.allowed_domains,
) -> bool:
    """Run deterministic security and policy guardrails on an outbound draft."""
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
    transport: Optional[DispatchTransport] = None,
    reserved_log_id: Optional[int | LeaseReservation] = None,
    lease_token: Optional[str] = None,
) -> DispatchReceipt:
    """Atomically verify fencing `lease_token`, send draft, and finalize to `SENT`.

    - Compare-and-swaps `PENDING` -> `DISPATCHING` using `lease_token` immediately before
      calling `active_transport.send()`. If the worker's lease expired and was reclaimed,
      `begin_dispatch_reconciliation` raises `StaleLeaseError` and aborts with zero send.
    - Separates `active_transport.send()` error handling from `complete_dispatch_reconciliation()`
      so a post-send bookkeeping exception never mislabels a delivered message as `SEND_ERROR`.
    """
    reservation = (
        reserved_log_id
        if reserved_log_id is not None
        else db.reserve_event(
            event_id=data.event_id,
            sender=data.sender,
            clean_subject=data.clean_subject,
        )
    )
    effective_token = lease_token or getattr(reservation, "lease_token", None)

    try:
        active_transport = (
            transport if transport is not None else InMemoryOutboxTransport(max_items=100)
        )
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
                # CAS check (`WHERE id=? AND status='PENDING' AND lease_token=?`):
                # Raises StaleLeaseError BEFORE `send()` if another worker reclaimed the lease!
                rec_id = db.begin_dispatch_reconciliation(
                    audit_log_id=reservation,
                    event_id=data.event_id,
                    recipient=data.sender,
                    referenced_ticket_id=draft.referenced_ticket_id,
                    lease_token=effective_token,
                )
                try:
                    active_transport.send(data, draft)
                except Exception:
                    db.complete_dispatch_reconciliation(rec_id, state="SEND_ERROR")
                    status = DispatchStatus.ESCALATED
                    escalate_to_human = True
                    final_payload = None
                else:
                    status = DispatchStatus.SENT
                    final_payload = draft
                    db.complete_dispatch_reconciliation(rec_id, state="DELIVERED")
            else:
                status = DispatchStatus.BLOCKED_VALIDATION
                escalate_to_human = True
                final_payload = None

        db.finalize_dispatch(
            reservation,
            data=data,
            rules=rules,
            status=status,
            draft=final_payload,
            escalate_to_human=escalate_to_human,
            tokens_used=tokens_used,
            lease_token=effective_token,
        )

        return DispatchReceipt(
            event_id=data.event_id,
            status=status,
            final_payload=final_payload,
            audit_log_id=int(reservation),
            tokens_used=tokens_used,
        )
    except StaleLeaseError:
        # Stale worker lost its lease to a newer attempt; do not touch the newer attempt's row!
        raise
    except Exception as exc:
        db.mark_failed(
            reservation,
            error_detail=str(exc),
            tokens_used=tokens_used,
            lease_token=effective_token,
        )
        raise
