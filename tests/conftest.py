"""Deterministic fixtures and mock local SLM HTTP server responses."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import pytest

from src.database import Database
from src.pipeline.step1_trigger import compute_payload_signature
from src.schemas import (
    CleanedPayload,
    SourceType,
)
from src.slm_client import LocalSLMClient

TEST_SECRET_KEY = "unit-test-hmac-secret-key-9988776655"


class _FakeHTTPResponse:
    """In-memory context manager mimicking `urllib.request.urlopen` response."""

    def __init__(self, body_bytes: bytes, status: int = 200) -> None:
        self._stream = io.BytesIO(body_bytes)
        self.status = status

    def read(self) -> bytes:
        return self._stream.read()

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stream.close()


class MockOllamaController:
    """Deterministic controller for intercepting `urllib.request.urlopen` calls to Ollama."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self._ResponderQueue: List[Callable[[Dict[str, Any]], bytes]] = []

    def queue_ollama_json(
        self,
        response_obj: Dict[str, Any] | str,
        *,
        prompt_eval_count: int = 45,
        eval_count: int = 25,
    ) -> None:
        """Queue a structured Ollama `/api/generate` envelope."""
        serialized_inner = (
            response_obj if isinstance(response_obj, str) else json.dumps(response_obj)
        )
        envelope = {
            "model": "mock-local-slm",
            "created_at": "2026-10-30T12:00:00Z",
            "response": serialized_inner,
            "done": True,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
        }
        raw_bytes = json.dumps(envelope).encode("utf-8")
        self._ResponderQueue.append(lambda _req: raw_bytes)

    def queue_timeout(self) -> None:
        """Queue a simulated socket timeout error."""

        def _raise_timeout(_req: Dict[str, Any]) -> bytes:
            raise TimeoutError("Simulated local SLM socket read timeout")

        self._ResponderQueue.append(_raise_timeout)

    def queue_http_error(self, code: int = 503) -> None:
        """Queue an HTTP error response."""

        def _raise_http(_req: Dict[str, Any]) -> bytes:
            raise urllib.error.HTTPError(
                url="http://127.0.0.1:11434/api/generate",
                code=code,
                msg="Service Unavailable",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )

        self._ResponderQueue.append(_raise_http)

    def handle_urlopen(self, request: urllib.request.Request, timeout: float = 5.0) -> _FakeHTTPResponse:
        body_str = (request.data or b"{}").decode("utf-8")
        parsed_body = json.loads(body_str)
        self.requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": parsed_body,
            }
        )
        if not self._ResponderQueue:
            raise AssertionError(
                f"Unexpected SLM call when MockOllamaController queue is empty: {parsed_body}"
            )
        handler = self._ResponderQueue.pop(0)
        return _FakeHTTPResponse(handler(parsed_body))


@pytest.fixture()
def secret_key() -> str:
    return TEST_SECRET_KEY


@pytest.fixture()
def memory_db() -> Database:
    db = Database(":memory:")
    yield db
    db.close()


@pytest.fixture()
def mock_ollama(monkeypatch: pytest.MonkeyPatch) -> MockOllamaController:
    controller = MockOllamaController()
    monkeypatch.setattr(urllib.request, "urlopen", controller.handle_urlopen)
    return controller


@pytest.fixture()
def slm_client(mock_ollama: MockOllamaController) -> LocalSLMClient:
    return LocalSLMClient(
        base_url="http://127.0.0.1:11434",
        timeout_seconds=2.0,
        max_retries=1,
    )


@pytest.fixture()
def make_signed_trigger(secret_key: str) -> Callable[..., Dict[str, Any]]:
    """Factory fixture that constructs a properly signed Stage 1 trigger dictionary."""

    def _factory(
        *,
        sender: str = "customer@acme.org",
        raw_subject: str = "Question about my monthly invoice",
        raw_body: str = "Hello, I was charged twice on invoice INV-2026-09. Please help.",
        source: str = SourceType.EMAIL.value,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        override_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "event_id": event_id or str(uuid4()),
            "source": source,
            "sender": sender,
            "raw_subject": raw_subject,
            "raw_body": raw_body,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        payload["signature"] = (
            override_signature
            if override_signature is not None
            else compute_payload_signature(payload, secret_key)
        )
        return payload

    return _factory


@pytest.fixture()
def sample_cleaned_payload() -> CleanedPayload:
    return CleanedPayload(
        event_id=uuid4(),
        sender="alice@company.org",
        clean_subject="Billing discrepancy on October statement",
        clean_body="Hi team, my credit card was billed twice for the Pro plan.",
        extracted_urls=["https://billing.example.com/invoices"],
        word_count=11,
        contains_suspicious_patterns=False,
    )

