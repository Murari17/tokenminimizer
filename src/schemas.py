"""Pydantic V2 immutable data contracts for all 6 pipeline stages."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, UUID4, field_validator


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


class ClassificationOutput(BaseModel):
    """Stage 3 output contract (SLM Node 1): strict intent and urgency classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: IntentCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_entities: List[str] = Field(default_factory=list)
    requires_human_review: bool

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
    """Stage 5 output contract (SLM Node 2): constrained response draft."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_line: str = Field(..., min_length=1, max_length=120)
    body_text: str = Field(..., min_length=1, max_length=600)
    referenced_ticket_id: str = Field(..., min_length=1, max_length=64)

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

