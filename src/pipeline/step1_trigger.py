"""Stage 1 [CODE — 0 Tokens]: Webhook/email/form/file ingestion & HMAC signature verification."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from src.config import DEFAULT_CONFIG
from src.schemas import RawTriggerPayload
from src.security import compute_hmac_signature, verify_hmac_signature


class TriggerIngestionError(ValueError):
    """Raised when incoming trigger payload fails size, signature, or schema checks."""


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


def ingest_trigger(
    raw_dict: Dict[str, Any],
    secret_key: str,
    max_payload_bytes: int = DEFAULT_CONFIG.max_payload_bytes,
) -> RawTriggerPayload:
    """Verify payload size, HMAC-SHA256 signature, and coerce into `RawTriggerPayload`.

    Tokens consumed: 0.
    """
    if not isinstance(raw_dict, dict):
        raise TriggerIngestionError("Trigger payload must be a dictionary.")
    if not secret_key or not isinstance(secret_key, str):
        raise TriggerIngestionError("A valid non-empty secret_key is required.")

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

    if not verify_hmac_signature(canonical_bytes, provided_signature, secret_key):
        raise TriggerIngestionError("HMAC-SHA256 signature verification failed.")

    # 3. Validate and coerce into immutable RawTriggerPayload
    try:
        return RawTriggerPayload.model_validate(raw_dict)
    except Exception as exc:
        raise TriggerIngestionError(f"Schema validation failed for trigger payload: {exc}") from exc

