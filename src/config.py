"""Configuration settings, model endpoints, strict timeouts, and token caps."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineConfig:
    """Immutable runtime configuration for the Token-Minimizing Workflow Engine."""

    # Webhook / Ingestion security
    webhook_secret: str = field(
        default_factory=lambda: os.environ.get(
            "TM_WEBHOOK_SECRET", "default-local-dev-hmac-secret-key-2026"
        )
    )
    max_payload_bytes: int = field(
        default_factory=lambda: int(os.environ.get("TM_MAX_PAYLOAD_BYTES", str(50 * 1024)))
    )
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
        default_factory=lambda: float(os.environ.get("TM_HTTP_TIMEOUT_SECONDS", "5.0"))
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


DEFAULT_CONFIG = EngineConfig()

