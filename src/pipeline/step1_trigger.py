"""Stage 1 [CODE — 0 Tokens]: Webhook/email/form/file ingestion, HMAC signature verification,
and timestamp replay window enforcement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from src.config import ConfigurationError, DEFAULT_CONFIG
from src.database import DuplicateEventError
from src.schemas import RawTriggerPayload
from src.security import compute_hmac_signature, verify_hmac_signature


class TriggerIngestionError(ValueError):
    """Raised when incoming trigger payload fails size, signature, replay, or schema checks."""


class ReplayAttackError(TriggerIngestionError, DuplicateEventError):
    """Raised when a trigger payload is stale, future-skewed, or replays an existing event_id."""


def _canonicalize_for_signing(raw_dict: Dict[str, Any]) -> bytes:
    """Serialize trigger fields (excluding `signature`) into canonical UTF-8 JSON bytes."""
    signing_dict: Dict[str, Any] = {}
    for key in sorted(raw_dict.keys()):
        if key == "signature":
            continue
        val = raw_dict[key]
        if isinstance(val, datetime):
            signing_dict[key] = val.isoformat()
        elif hasattr(val, "value"):
            signing_dict[key] = val.value
        else:
            signing_dict[key] = str(val) if key == "event_id" else val
    return json.dumps(signing_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_payload_signature(raw_dict: Dict[str, Any], secret_key: str) -> str:
    """Compute canonical HMAC-SHA256 hex digest for a trigger payload dictionary."""
    canonical_bytes = _canonicalize_for_signing(raw_dict)
    return compute_hmac_signature(canonical_bytes, secret_key)


def _verify_timestamp_freshness(
    ts: datetime,
    max_past_skew_seconds: int = DEFAULT_CONFIG.max_timestamp_skew_seconds,
    max_future_skew_seconds: int = DEFAULT_CONFIG.max_future_skew_seconds,
    now: datetime | None = None,
) -> None:
    """Reject stale or future-skewed payload timestamps to prevent replay attacks."""
    reference_now = now or datetime.now(timezone.utc)
    normalized_ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    age_seconds = (reference_now - normalized_ts).total_seconds()

    if age_seconds > max_past_skew_seconds:
        raise ReplayAttackError(
            f"Stale payload rejected: timestamp is {int(age_seconds)}s old "
            f"(max allowed replay window is {max_past_skew_seconds}s)."
        )
    if age_seconds < -max_future_skew_seconds:
        raise ReplayAttackError(
            f"Future-skewed payload rejected: timestamp is {int(-age_seconds)}s in the future "
            f"(max allowed clock skew is {max_future_skew_seconds}s)."
        )


def ingest_trigger(
    raw_dict: Dict[str, Any],
    secret_key: str,
    max_payload_bytes: int = DEFAULT_CONFIG.max_payload_bytes,
    max_timestamp_skew_seconds: int = DEFAULT_CONFIG.max_timestamp_skew_seconds,
) -> RawTriggerPayload:
    """Verify secret strength, payload size, HMAC-SHA256 signature, schema, and replay freshness.

    Tokens consumed: 0.
    """
    if not isinstance(raw_dict, dict):
        raise TriggerIngestionError("Trigger payload must be a dictionary.")

    try:
        validated_secret = DEFAULT_CONFIG.resolve_webhook_secret(secret_key)
    except ConfigurationError as exc:
        raise TriggerIngestionError(str(exc)) from exc

    # 1. Enforce 50KB size limit before parsing/validating models to prevent memory exhaustion
    try:
        serialized_raw = json.dumps(raw_dict, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TriggerIngestionError(f"Malformed unserializable payload: {exc}") from exc

    if len(serialized_raw) > max_payload_bytes:
        raise TriggerIngestionError(
            f"Payload size ({len(serialized_raw)} bytes) exceeds maximum limit of {max_payload_bytes} bytes."
        )

    # 2. Verify required keys exist prior to HMAC verification
    required_fields = (
        "event_id",
        "source",
        "sender",
        "raw_subject",
        "raw_body",
        "timestamp",
        "signature",
    )
    missing = [f for f in required_fields if f not in raw_dict or raw_dict[f] is None]
    if missing:
        raise TriggerIngestionError(f"Missing required trigger fields: {', '.join(missing)}")

    provided_signature = str(raw_dict.get("signature", ""))
    canonical_bytes = _canonicalize_for_signing(raw_dict)

    if not verify_hmac_signature(canonical_bytes, provided_signature, validated_secret):
        raise TriggerIngestionError("HMAC-SHA256 signature verification failed.")

    # 3. Validate and coerce into immutable RawTriggerPayload
    try:
        payload = RawTriggerPayload.model_validate(raw_dict)
    except Exception as exc:
        raise TriggerIngestionError(f"Schema validation failed for trigger payload: {exc}") from exc

    # 4. Enforce timestamp replay window
    _verify_timestamp_freshness(
        payload.timestamp,
        max_past_skew_seconds=max_timestamp_skew_seconds,
    )

    return payload
