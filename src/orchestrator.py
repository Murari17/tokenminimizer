"""Main deterministic state machine connecting Steps 1 through 6 with fencing-token (`lease_token`)
compare-and-swap, automatic stale-lease sweeping, mid-pipeline heartbeat, and context-isolated
per-event token accounting.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import DEFAULT_CONFIG, EngineConfig
from src.database import Database, DuplicateEventError, StaleLeaseError
from src.pipeline.step1_trigger import ReplayAttackError, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step3_classify import read_and_classify
from src.pipeline.step4_rules import apply_rules
from src.pipeline.step5_draft import draft_response
from src.pipeline.step6_validate import (
    DispatchTransport,
    InMemoryOutboxTransport,
    validate_save_send,
)
from src.schemas import DispatchReceipt
from src.slm_client import LocalSLMClient


class WorkflowOrchestrator:
    """Deterministic 6-stage state machine with zero-token control flow and fencing-token replay defense."""

    def __init__(
        self,
        *,
        config: EngineConfig = DEFAULT_CONFIG,
        slm_client: Optional[LocalSLMClient] = None,
        db: Optional[Database] = None,
        transport: Optional[DispatchTransport] = None,
    ) -> None:
        self.config = config
        self.slm_client = slm_client or LocalSLMClient(config=config)
        self.db = db or Database(db_path=config.sqlite_db_path)
        self.transport: DispatchTransport = (
            transport if transport is not None else InMemoryOutboxTransport(max_items=1000)
        )
        # Sweep any orphaned PENDING/DISPATCHING rows left by prior process crashes on startup
        self.db.sweep_stale_pending(lease_seconds=self.config.pending_lease_seconds)

    def process_event(
        self,
        raw_dict: Dict[str, Any],
        secret_key: Optional[str] = None,
    ) -> DispatchReceipt:
        """Execute the 6-stage workflow with fencing `lease_token` CAS and inter-step heartbeats."""
        effective_secret = self.config.resolve_webhook_secret(secret_key)

        with self.slm_client.track_scope_tokens() as scoped_tokens:
            # Step 1 [CODE - 0 Tokens]: Verify HMAC, payload size, and timestamp freshness
            raw_payload = ingest_trigger(
                raw_dict=raw_dict,
                secret_key=effective_secret,
                max_payload_bytes=self.config.max_payload_bytes,
                max_timestamp_skew_seconds=self.config.max_timestamp_skew_seconds,
            )

            # Atomically reserve `event_id` with a unique `lease_token` (or reclaim if a prior
            # attempt failed mid-pipeline or its PENDING lease expired).
            try:
                reservation = self.db.reserve_event(
                    event_id=raw_payload.event_id,
                    sender=raw_payload.sender,
                    clean_subject=raw_payload.raw_subject[:500],
                    lease_seconds=self.config.pending_lease_seconds,
                )
            except DuplicateEventError as exc:
                raise ReplayAttackError(str(exc)) from exc

            try:
                # Step 2 [CODE - 0 Tokens]: Strip HTML, normalize Unicode, extract URLs, scan for injection
                cleaned_payload = fetch_and_clean(
                    payload=raw_payload,
                    max_body_chars=self.config.max_clean_body_chars,
                )

                # Step 3 [SMALL LLM - Minimal Cost]: Classify intent & urgency
                classification = read_and_classify(
                    data=cleaned_payload,
                    slm_client=self.slm_client,
                    model=self.config.classify_model,
                )

                # Heartbeat lease after Step 3 SLM call; abort immediately if lease was lost
                if not self.db.heartbeat_lease(reservation):
                    raise StaleLeaseError(
                        f"Lease superseded after Step 3 for event_id={raw_payload.event_id}; aborting stale worker."
                    )

                # Step 4 [CODE - 0 Tokens]: Apply deterministic routing, SLA, priority, and tone matrix
                rule_result = apply_rules(
                    data=cleaned_payload,
                    cls=classification,
                )

                # Step 5 [SMALL LLM - Minimal Cost]: Draft constrained reply
                draft = draft_response(
                    data=cleaned_payload,
                    rules=rule_result,
                    slm_client=self.slm_client,
                    model=self.config.draft_model,
                )

                # Exact context-isolated tokens consumed by this event only
                run_tokens_used = scoped_tokens[0] + scoped_tokens[1]

                # Step 6 [CODE - 0 Tokens]: CAS PENDING->DISPATCHING with lease_token, send, finalize
                receipt = validate_save_send(
                    data=cleaned_payload,
                    rules=rule_result,
                    draft=draft,
                    db=self.db,
                    tokens_used=run_tokens_used,
                    allowed_domains=self.config.allowed_domains,
                    transport=self.transport,
                    reserved_log_id=reservation,
                    lease_token=reservation.lease_token,
                )
                return receipt
            except StaleLeaseError as exc:
                raise ReplayAttackError(str(exc)) from exc
            except Exception as exc:
                run_tokens_used = scoped_tokens[0] + scoped_tokens[1]
                self.db.mark_failed(
                    reservation,
                    error_detail=f"{type(exc).__name__}: {exc}",
                    tokens_used=run_tokens_used,
                    lease_token=reservation.lease_token,
                )
                raise


def run_pipeline(
    raw_dict: Dict[str, Any],
    *,
    secret_key: Optional[str] = None,
    slm_client: Optional[LocalSLMClient] = None,
    db: Optional[Database] = None,
    transport: Optional[DispatchTransport] = None,
    config: EngineConfig = DEFAULT_CONFIG,
) -> DispatchReceipt:
    """Functional entry point to execute the full 6-stage workflow."""
    orchestrator = WorkflowOrchestrator(
        config=config,
        slm_client=slm_client,
        db=db,
        transport=transport,
    )
    return orchestrator.process_event(raw_dict=raw_dict, secret_key=secret_key)
