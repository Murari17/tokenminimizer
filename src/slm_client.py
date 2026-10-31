"""Zero-dependency HTTP client for local SLMs (Ollama/vLLM) with strict JSON schema enforcement,
thread-safe token accounting, and self-correcting schema retries.
"""

from __future__ import annotations

import contextvars
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional, Type, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from src.config import DEFAULT_CONFIG, EngineConfig

T = TypeVar("T", bound=BaseModel)

_SCOPED_TOKEN_ACCUMULATOR: contextvars.ContextVar[Optional[List[int]]] = contextvars.ContextVar(
    "_SCOPED_TOKEN_ACCUMULATOR", default=None
)


class SLMClientError(RuntimeError):
    """Raised when the local SLM fails to respond or violates the schema contract."""


class SLMTransportError(SLMClientError):
    """Raised when the local SLM HTTP transport times out or is unreachable."""


@dataclass(frozen=True)
class SLMCallMetrics:
    """Token consumption metrics from a single SLM generation call."""

    model: str
    prompt_eval_count: int
    eval_count: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_eval_count + self.eval_count


class LocalSLMClient:
    """Thread-safe zero-dependency HTTP client for local SLMs via Ollama `/api/generate`."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        config: EngineConfig = DEFAULT_CONFIG,
    ) -> None:
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self._validate_base_url(self.base_url)
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else config.http_timeout_seconds
        )
        self.max_retries = max_retries if max_retries is not None else config.max_retries
        self._lock = threading.Lock()
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._call_history: list[SLMCallMetrics] = []

    @staticmethod
    def _validate_base_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid SLM base URL: {url!r}")

    @property
    def total_prompt_tokens(self) -> int:
        with self._lock:
            return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        with self._lock:
            return self._total_completion_tokens

    @property
    def total_tokens_used(self) -> int:
        with self._lock:
            return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def call_history(self) -> tuple[SLMCallMetrics, ...]:
        with self._lock:
            return tuple(self._call_history)

    def reset_token_counters(self) -> None:
        """Reset accumulated token counters cleanly."""
        with self._lock:
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._call_history.clear()

    @contextmanager
    def track_scope_tokens(self) -> Iterator[List[int]]:
        """Context-local token counter [prompt_tokens, eval_tokens] isolated per concurrent event."""
        bucket = [0, 0]
        token = _SCOPED_TOKEN_ACCUMULATOR.set(bucket)
        try:
            yield bucket
        finally:
            _SCOPED_TOKEN_ACCUMULATOR.reset(token)

    # Backwards-compatible alias
    track_Scope_tokens = track_scope_tokens

    def _record_tokens(self, model: str, prompt_tokens: int, eval_tokens: int) -> None:
        with self._lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += eval_tokens
            self._call_history.append(
                SLMCallMetrics(
                    model=model,
                    prompt_eval_count=prompt_tokens,
                    eval_count=eval_tokens,
                )
            )
        scoped_bucket = _SCOPED_TOKEN_ACCUMULATOR.get()
        if scoped_bucket is not None:
            scoped_bucket[0] += prompt_tokens
            scoped_bucket[1] += eval_tokens

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        """Perform a zero-dependency HTTP POST request with strict timeout."""
        url = f"{self.base_url}{endpoint}"
        body_bytes = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
                return json.loads(raw_response)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise SLMTransportError(f"Local SLM HTTP transport failure: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SLMClientError(f"Local SLM returned non-JSON HTTP envelope: {exc}") from exc

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[T],
        num_predict: int = 128,
        temperature: float = 0.0,
    ) -> T:
        """Invoke the local SLM with JSON schema enforcement, hard token caps, and self-correcting retry."""
        json_schema = response_schema.model_json_schema()
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        active_prompt = user_prompt

        for attempt_idx in range(attempts):
            payload = {
                "model": model,
                "system": system_prompt,
                "prompt": active_prompt,
                "stream": False,
                "format": json_schema,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(num_predict),
                },
            }
            try:
                envelope = self._post_json("/api/generate", payload)
                prompt_tokens = max(0, int(envelope.get("prompt_eval_count", 0) or 0))
                eval_tokens = max(0, int(envelope.get("eval_count", 0) or 0))
                self._record_tokens(model, prompt_tokens, eval_tokens)

                raw_content = envelope.get("response", "")
                if isinstance(raw_content, dict):
                    return response_schema.model_validate(raw_content)
                if not isinstance(raw_content, str) or not raw_content.strip():
                    raise SLMClientError("Empty or missing 'response' field in SLM output.")

                parsed_json = json.loads(raw_content)
                return response_schema.model_validate(parsed_json)
            except SLMTransportError as exc:
                last_error = exc
            except (SLMClientError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt_idx + 1 < attempts:
                    err_summary = str(exc).splitlines()[0][:240]
                    active_prompt = (
                        f"{user_prompt}\n\n"
                        f"[SCHEMA VALIDATION CORRECTION: Your previous output was invalid "
                        f"({err_summary}). Output ONLY valid JSON matching {response_schema.__name__}.]"
                    )

        if isinstance(last_error, SLMTransportError):
            raise SLMTransportError(
                f"SLM transport failed after {attempts} attempt(s) for {response_schema.__name__}: {last_error}"
            ) from last_error
        raise SLMClientError(
            f"SLM generation failed after {attempts} attempt(s) for {response_schema.__name__}: {last_error}"
        ) from last_error
