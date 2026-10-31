"""Main deterministic state machine connecting Steps 1 through 6 with replay protection
and context-isolated per-event token accounting.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import DEFAULT_CONFIG, EngineConfig
from src.database import Database
from src.pipeline.step1_trigger import ReplayAttackError, ingest_trigger
from src.pipeline.step2_clean import fetch_and_clean
from src.pipeline.step3_classify import read_and_classify
from src.pipeline.step4_rules import apply_rules
from src.pipeline.step5_draft import draft_response
from src.pipeline.step6_validate import (
    DEFAULT_OUTBOX_TRANSPORT,
    DispatchTransport,
    validate_save_send,
)
from src.schemas import DispatchReceipt
from src.slm_client import LocalSLMClient


class WorkflowOrchestrator:
    """Deterministic 6-stage state machine with zero-token control flow and replay defense."""

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
        self.transport = transport or DEFAULT_OUTBOX_TRANSPORT

    def process_event(
        self,
        raw_dict: Dict[str, Any],
        secret_key: Optional[str] = None,
    ) -> DispatchReceipt:
        """Execute the 6-stage workflow deterministically with thread-safe token accounting."""
        effective_secret = self.config.resolve_webhook_secret(secret_key)

        with self.slm_client.track_Scope_tokens() as scoped_tokens:
            # Step 1 [CODE - 0 Tokens]: Verify HMAC, payload size, and timestamp freshness
            raw_payload = ingest_trigger(
                raw_dict=raw_dict,
                secret_key=effective_secret,
                max_payload_bytes=self.config.max_payload_bytes,
                max_timestamp_skew_seconds=self.config.max_timestamp_skew_seconds,
            )

            # Replay check before spending any SLM tokens
            if self.db.has_event_id(raw_payload.event_id):
                raise ReplayAttackError(
                    f"Replay detected: event_id {raw_payload.event_id} has already been processed."
                )

            # Step 2 [CODE - 0 Tokens]: Strip HTML, normalize Unicode, extract URLs, scan for injection
            cleaned_payload = fetch_and_clean(
                payload=raw_payload,
                max_body_chars=self.config.max_clean_body_chars,
            )

            # Step 3 [SMALL LLM - Minimal Cost]: Classify intent & urgency (short-circuits on suspicious/empty)
            classification = read_and_classify(
                data=cleaned_payload,
                slm_client=self.slm_client,
                model=self.config.classify_model,
            )

            # Step 4 [CODE - 0 Tokens]: Apply deterministic routing, SLA, priority, and tone matrix
            rule_result = apply_rules(
                data=cleaned_payload,
                cls=classification,
            )

            # Step 5 [SMALL LLM - Minimal Cost]: Draft constrained reply (skipped if should_draft_reply=False)
            draft = draft_response(
                data=cleaned_payload,
                rules=rule_result,
                slm_client=self.slm_client,
                model=self.config.draft_model,
            )

            # Exact context-isolated tokens consumed by this event only
            run_tokens_used = scoped_tokens[0] + scoped_tokens[1]

            # Step 6 [CODE - 0 Tokens]: Run output guardrails, dispatch via transport, persist to SQLite
            receipt = validate_save_send(
                data=cleaned_payload,
                rules=rule_result,
                draft=draft,
                db=self.db,
                tokens_used=run_tokens_used,
                allowed_domains=self.config.allowed_domains,
                transport=self.transport,
            )
            return receipt


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
