"""Zero-dependency HTTP client for local SLMs (Ollama/vLLM) with strict JSON schema enforcement."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Type, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from src.config import DEFAULT_CONFIG, EngineConfig

T = TypeVar("T", bound=BaseModel)


class SLMClientError(RuntimeError):
    """Raised when the local SLM fails to respond or violates the schema contract."""


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
    """Zero-dependency HTTP client for local small language models via Ollama `/api/generate`."""

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
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def total_tokens_used(self) -> int:
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def call_history(self) -> tuple[SLMCallMetrics, ...]:
        return tuple(self._call_history)

    def reset_token_counters(self) -> None:
        """Reset accumulated token counters between independent runs if desired."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._call_history.clear()

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
            raise SLMClientError(f"Local SLM HTTP transport failure: {exc}") from exc
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
        """Invoke the local SLM with JSON schema enforcement, hard token caps, and retry handling."""
        json_schema = response_schema.model_json_schema()
        payload = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": json_schema,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(num_predict),
            },
        }

        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None

        for _ in range(attempts):
            try:
                envelope = self._post_json("/api/generate", payload)
                prompt_tokens = max(0, int(envelope.get("prompt_eval_count", 0) or 0))
                eval_tokens = max(0, int(envelope.get("eval_count", 0) or 0))

                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += eval_tokens
                self._call_history.append(
                    SLMCallMetrics(
                        model=model,
                        prompt_eval_count=prompt_tokens,
                        eval_count=eval_tokens,
                    )
                )

                raw_content = envelope.get("response", "")
                if isinstance(raw_content, dict):
                    return response_schema.model_validate(raw_content)
                if not isinstance(raw_content, str) or not raw_content.strip():
                    raise SLMClientError("Empty or missing 'response' field in SLM output.")

                parsed_json = json.loads(raw_content)
                return response_schema.model_validate(parsed_json)
            except (SLMClientError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                last_error = exc

        raise SLMClientError(
            f"SLM generation failed after {attempts} attempt(s) for {response_schema.__name__}: {last_error}"
        ) from last_error

