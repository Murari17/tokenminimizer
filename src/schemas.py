"""Pydantic V2 immutable data contracts for all 6 pipeline stages."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
import threading
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, UUID4, field_validator, model_validator


class SourceType(str, Enum):
    """Supported ingestion trigger sources."""

    EMAIL = "EMAIL"
    FORM = "FORM"
    FILE = "FILE"


class IntentCategory(str, Enum):
    """Supported classification intents for SLM Node 1."""

    BILLING = "BILLING"
    TECHNICAL_SUPPORT = "TECHNICAL_SUPPORT"
    SECURITY_REPORT = "SECURITY_REPORT"
    FEATURE_REQUEST = "FEATURE_REQUEST"
    SPAM = "SPAM"


class PriorityLevel(str, Enum):
    """Deterministic routing priority levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DispatchStatus(str, Enum):
    """Terminal pipeline status codes for Stage 6."""

    SENT = "SENT"
    DROPPED_SPAM = "DROPPED_SPAM"
    ESCALATED = "ESCALATED"
    BLOCKED_VALIDATION = "BLOCKED_VALIDATION"


_SENDER_RE = re.compile(
    r"^(?:[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9_\-:.@]{3,128})$"
)


class RawTriggerPayload(BaseModel):
    """Stage 1 output contract: verified raw ingestion payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID4
    source: SourceType
    sender: str = Field(..., min_length=3, max_length=254)
    raw_subject: str = Field(..., max_length=1000)
    raw_body: str = Field(..., max_length=50000)
    timestamp: datetime
    signature: str = Field(..., min_length=64, max_length=128)

    @field_validator("sender")
    @classmethod
    def validate_sender(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(ch in cleaned for ch in ("\r", "\n", "\x00")):
            raise ValueError("Sender contains invalid control characters or is empty.")
        if not _SENDER_RE.match(cleaned):
            raise ValueError(f"Invalid sender email or identifier format: {cleaned!r}")
        return cleaned

    @field_validator("signature")
    @classmethod
    def validate_signature_hex(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned.startswith("sha256="):
            cleaned = cleaned[7:]
        if len(cleaned) != 64 or any(c not in "0123456789abcdef" for c in cleaned):
            raise ValueError("Signature must be a valid 64-character hex HMAC-SHA256 digest.")
        return cleaned


class CleanedPayload(BaseModel):
    """Stage 2 output contract: normalized and sanitized payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    sender: str = Field(..., min_length=3, max_length=254)
    clean_subject: str = Field(..., max_length=500)
    clean_body: str = Field(..., max_length=2000)
    extracted_urls: List[str] = Field(default_factory=list)
    word_count: int = Field(..., ge=0)
    contains_suspicious_patterns: bool

    @field_validator("clean_body")
    @classmethod
    def enforce_body_cap(cls, value: str) -> str:
        if len(value) > 2000:
            raise ValueError("clean_body must not exceed 2,000 characters.")
        return value


_CANONICAL_INTENTS = {e.value for e in IntentCategory}

_INTENT_ALIASES = {
    "BILLING_INQUIRY": "BILLING",
    "PAYMENT": "BILLING",
    "REFUND": "BILLING",
    "INVOICE": "BILLING",
    "SUBSCRIPTION": "BILLING",
    "TECH_SUPPORT": "TECHNICAL_SUPPORT",
    "TECHNICAL": "TECHNICAL_SUPPORT",
    "SUPPORT": "TECHNICAL_SUPPORT",
    "BUG": "TECHNICAL_SUPPORT",
    "OUTAGE": "TECHNICAL_SUPPORT",
    "SECURITY": "SECURITY_REPORT",
    "VULNERABILITY": "SECURITY_REPORT",
    "ABUSE": "SECURITY_REPORT",
    "FEATURE": "FEATURE_REQUEST",
    "FEEDBACK": "FEATURE_REQUEST",
    "ENHANCEMENT": "FEATURE_REQUEST",
    "JUNK": "SPAM",
    "PROMOTIONAL": "SPAM",
    "MARKETING": "SPAM",
}

_TELEMETRY_LOCK = threading.Lock()
_ALIAS_REMAP_COUNTS: Dict[str, int] = {}


def get_alias_remap_telemetry() -> Dict[str, Any]:
    """Return thread-safe telemetry of non-canonical intent label remaps (model drift indicator)."""
    with _TELEMETRY_LOCK:
        by_alias = dict(_ALIAS_REMAP_COUNTS)
    return {
        "total_alias_remaps": sum(by_alias.values()),
        "by_alias": by_alias,
    }


def reset_alias_remap_telemetry() -> None:
    """Reset the intent alias drift telemetry counters."""
    with _TELEMETRY_LOCK:
        _ALIAS_REMAP_COUNTS.clear()


class ClassificationOutput(BaseModel):
    """Stage 3 output contract (SLM Node 1): strict intent and urgency classification.

    Note: Scalar routing fields (`intent`, `confidence`, `requires_human_review`) are
    ordered BEFORE `detected_entities` so constrained decoding emits critical routing
    decisions within the first ~25 tokens across all local SLM architectures.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: IntentCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human_review: bool = False
    detected_entities: List[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def normalize_slm_output(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)

        # Normalize intent casing/separators; count non-canonical alias drift in telemetry
        raw_intent = normalized.get("intent")
        if isinstance(raw_intent, str):
            cleaned_intent = re.sub(r"[\s\-]+", "_", raw_intent.strip().upper())
            if cleaned_intent in _CANONICAL_INTENTS:
                normalized["intent"] = cleaned_intent
            elif cleaned_intent in _INTENT_ALIASES:
                mapped = _INTENT_ALIASES[cleaned_intent]
                normalized["intent"] = mapped
                with _TELEMETRY_LOCK:
                    key = f"{cleaned_intent}->{mapped}"
                    _ALIAS_REMAP_COUNTS[key] = _ALIAS_REMAP_COUNTS.get(key, 0) + 1

        # Normalize confidence: ONLY rescale by /100 when an explicit '%' suffix is present.
        # Out-of-range numeric values (e.g., 1.5 or -0.2) are NOT silently rescaled/clamped so
        # Pydantic's Field(ge=0.0, le=1.0) rejects them as schema violations.
        raw_conf = normalized.get("confidence")
        if isinstance(raw_conf, str):
            stripped_conf = raw_conf.strip()
            if stripped_conf.endswith("%"):
                try:
                    pct_val = float(stripped_conf[:-1].strip())
                    normalized["confidence"] = round(pct_val / 100.0, 4)
                except ValueError:
                    pass
            else:
                try:
                    normalized["confidence"] = round(float(stripped_conf), 4)
                except ValueError:
                    pass
        elif isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool):
            normalized["confidence"] = round(float(raw_conf), 4)

        # Ensure detected_entities is a list of strings even if null or single string
        raw_entities = normalized.get("detected_entities")
        if raw_entities is None:
            normalized["detected_entities"] = []
        elif isinstance(raw_entities, str):
            normalized["detected_entities"] = [raw_entities] if raw_entities.strip() else []

        # Default requires_human_review if omitted or coerced from string
        raw_review = normalized.get("requires_human_review")
        if raw_review is None:
            conf_check = normalized.get("confidence")
            intent_check = normalized.get("intent")
            normalized["requires_human_review"] = bool(
                intent_check == "SECURITY_REPORT"
                or (isinstance(conf_check, (int, float)) and conf_check < 0.75)
            )
        elif isinstance(raw_review, str):
            normalized["requires_human_review"] = raw_review.strip().lower() in ("true", "1", "yes")

        return normalized

    @field_validator("detected_entities")
    @classmethod
    def sanitize_entities(cls, entities: List[str]) -> List[str]:
        return [str(e).strip()[:100] for e in entities[:20] if str(e).strip()]


class RuleEvaluationResult(BaseModel):
    """Stage 4 output contract: deterministic routing and SLA matrix output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    should_draft_reply: bool
    escalate_to_human: bool
    priority: PriorityLevel
    sla_hours: int = Field(..., ge=0, le=720)
    assigned_queue: str = Field(..., min_length=1, max_length=64)
    response_template_id: str = Field(..., max_length=64)
    allowed_tone: str = Field(..., max_length=64)


class DraftOutput(BaseModel):
    """Stage 5 output contract (SLM Node 2): constrained response draft.

    Note: `referenced_ticket_id` and `subject_line` are ordered before `body_text` so
    constrained decoding emits short routing metadata before the multi-sentence body.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    referenced_ticket_id: str = Field(..., min_length=1, max_length=64)
    subject_line: str = Field(..., min_length=1, max_length=120)
    body_text: str = Field(..., min_length=1, max_length=600)

    @field_validator("subject_line")
    @classmethod
    def validate_subject_length(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("subject_line cannot be blank.")
        if len(cleaned) > 120:
            raise ValueError("subject_line exceeds 120 chars.")
        return cleaned

    @field_validator("body_text")
    @classmethod
    def validate_body_length(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("body_text cannot be blank.")
        if len(cleaned) > 600:
            raise ValueError("body_text exceeds 600 chars.")
        return cleaned


class DispatchReceipt(BaseModel):
    """Stage 6 output contract: immutable dispatch and audit receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    status: DispatchStatus
    final_payload: Optional[DraftOutput] = None
    audit_log_id: int = Field(..., ge=1)
    tokens_used: int = Field(..., ge=0)

