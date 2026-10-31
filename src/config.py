"""Configuration settings, model endpoints, strict timeouts, replay windows, and token caps."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class ConfigurationError(RuntimeError):
    """Raised when a required security configuration (such as TM_WEBHOOK_SECRET) is missing."""


@dataclass(frozen=True)
class EngineConfig:
    """Immutable runtime configuration for the Token-Minimizing Workflow Engine."""

    # Webhook / Ingestion security — no hardcoded fallback secret!
    webhook_secret: str = field(
        default_factory=lambda: os.environ.get("TM_WEBHOOK_SECRET", "")
    )
    min_secret_length: int = 16
    max_payload_bytes: int = field(
        default_factory=lambda: int(os.environ.get("TM_MAX_PAYLOAD_BYTES", str(50 * 1024)))
    )
    max_timestamp_skew_seconds: int = field(
        default_factory=lambda: int(os.environ.get("TM_MAX_TIMESTAMP_SKEW_SECONDS", "300"))
    )
    max_future_skew_seconds: int = 60
    max_clean_body_chars: int = 2000

    # Local SLM HTTP endpoint settings (Ollama / vLLM compatible)
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("TM_OLLAMA_URL", "http://127.0.0.1:11434")
    )
    classify_model: str = field(
        default_factory=lambda: os.environ.get("TM_CLASSIFY_MODEL", "qwen2.5:1.5b")
    )
    draft_model: str = field(
        default_factory=lambda: os.environ.get("TM_DRAFT_MODEL", "llama3.2:3b")
    )

    # Hard token caps and deterministic generation parameters
    classify_max_tokens: int = 128
    draft_max_tokens: int = 200
    temperature: float = 0.0
    http_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("TM_HTTP_TIMEOUT_SECONDS", "15.0"))
    )
    max_retries: int = 1

    # Validation & Persistence
    sqlite_db_path: str = field(
        default_factory=lambda: os.environ.get("TM_SQLITE_DB_PATH", ":memory:")
    )
    allowed_domains: tuple[str, ...] = (
        "status.example.com",
        "docs.example.com",
        "support.example.com",
        "billing.example.com",
    )

    def resolve_webhook_secret(self, explicit_secret: str | None = None) -> str:
        """Return validated webhook secret or hard-fail if unset/weak."""
        candidate = explicit_secret if explicit_secret is not None else self.webhook_secret
        if not candidate or not isinstance(candidate, str) or not candidate.strip():
            raise ConfigurationError(
                "Webhook HMAC secret is not configured. Set TM_WEBHOOK_SECRET or pass secret_key explicitly."
            )
        if len(candidate.strip()) < self.min_secret_length:
            raise ConfigurationError(
                f"Webhook HMAC secret must be at least {self.min_secret_length} characters long."
            )
        return candidate


DEFAULT_CONFIG = EngineConfig()
