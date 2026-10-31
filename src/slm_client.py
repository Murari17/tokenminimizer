"""Zero-dependency HTTP client for local SLMs (Ollama/vLLM) with strict JSON schema enforcement,
thread-safe token accounting, and self-correcting schema retries.
"""

from __future__ import annotations

import contextvars
import json
import re
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Type, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from src.config import DEFAULT_CONFIG, EngineConfig

T = TypeVar("T", bound=BaseModel)

_SCOPED_TOKEN_ACCUMULATOR: contextvars.ContextVar[Optional[List[int]]] = contextvars.ContextVar(
    "_SCOPED_TOKEN_ACCUMULATOR", default=None
)

_THINK_BLOCK_RE = re.compile(
    r"<(?:think|thought|reasoning)>.*?(?:</(?:think|thought|reasoning)>|$)",
    re.IGNORECASE | re.DOTALL,
)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def flatten_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline all Pydantic `$defs`/`$ref` references and strip metadata bloat for universal
    compatibility across Ollama, llama.cpp, vLLM, and LM Studio GBNF grammar engines.
    """
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs:
                    resolved = dict(defs[ref_name])
                    for k, v in node.items():
                        if k != "$ref":
                            resolved[k] = v
                    return _resolve(resolved)
            cleaned: Dict[str, Any] = {}
            for k, v in node.items():
                if k in ("$defs", "title"):
                    continue
                cleaned[k] = _resolve(v)
            if cleaned.get("type") == "object" and "properties" in cleaned:
                # Ensure all properties are listed in `required` so GBNF grammars emit every key in order
                cleaned["required"] = list(cleaned["properties"].keys())
            return cleaned
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


def extract_and_repair_json(raw_text: str) -> Dict[str, Any]:
    """Universally extract and parse a JSON object from any local SLM response.

    Handles:
    1. Direct valid JSON objects.
    2. Reasoning models emitting `<think>...</think>` or `<thought>...</thought>` prefixes.
    3. Markdown code blocks (` ```json ... ``` `).
    4. Truncated trailing array elements (e.g., if `num_predict` cuts off inside `detected_entities`
       after the required scalar routing fields were already emitted).
    """
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty SLM response text.")

    # Fast path: strict JSON object
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strip thinking blocks and markdown code fences
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Locate first '{'
    start_idx = cleaned.find("{")
    if start_idx == -1:
        raise ValueError(f"No JSON object start '{{' found in SLM response: {text[:120]!r}")

    candidate = cleaned[start_idx:]
    end_idx = candidate.rfind("}")
    if end_idx != -1:
        try:
            parsed = json.loads(candidate[: end_idx + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Repair path 1: if truncated inside a trailing array (such as `detected_entities`),
    # strip the incomplete tail back to the last complete key-value comma.
    if '"detected_entities"' in candidate:
        prefix, _, _ = candidate.partition('"detected_entities"')
        prefix_clean = prefix.rstrip().rstrip(",")
        repaired = prefix_clean + "}"
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                parsed.setdefault("detected_entities", [])
                return parsed
        except json.JSONDecodeError:
            pass

    # Repair path 2: if truncated inside a trailing `body_text` string after `referenced_ticket_id`
    # and `subject_line` were already emitted, close the string and object cleanly.
    if '"body_text"' in candidate and '"referenced_ticket_id"' in candidate:
        repaired_body = candidate.rstrip().rstrip("\\") + '"}'
        try:
            parsed = json.loads(repaired_body)
            if isinstance(parsed, dict) and parsed.get("body_text"):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Unable to parse or repair JSON from SLM output: {text[:160]!r}")


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
        """Invoke the local SLM with dereferenced JSON schema enforcement, thinking suppression,
        hard token caps, universal JSON extraction/repair, and self-correcting retry.
        """
        json_schema = flatten_json_schema(response_schema.model_json_schema())
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        active_prompt = user_prompt

        for attempt_idx in range(attempts):
            payload = {
                "model": model,
                "system": system_prompt,
                "prompt": active_prompt,
                "stream": False,
                "think": False,
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

                parsed_json = extract_and_repair_json(raw_content)
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
                        f"({err_summary}). Output ONLY a compact JSON object matching {response_schema.__name__} "
                        f"with no thinking, no markdown, and at most 2 short detected_entities.]"
                    )

        if isinstance(last_error, SLMTransportError):
            raise SLMTransportError(
                f"SLM transport failed after {attempts} attempt(s) for {response_schema.__name__}: {last_error}"
            ) from last_error
        raise SLMClientError(
            f"SLM generation failed after {attempts} attempt(s) for {response_schema.__name__}: {last_error}"
        ) from last_error
