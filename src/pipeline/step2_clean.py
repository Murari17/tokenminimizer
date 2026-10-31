"""Stage 2 [CODE — 0 Tokens]: HTML stripping, zero-width removal, Unicode normalization,
URL extraction, and deterministic prompt-injection scanning.
"""

from __future__ import annotations

import re
import unicodedata

from src.config import DEFAULT_CONFIG
from src.schemas import CleanedPayload, RawTriggerPayload
from src.security import (
    detect_prompt_injection,
    extract_urls_safely,
    strip_html_safely,
    strip_invisible_characters,
)

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b")


def _normalize_and_clean_text(text: str, max_chars: int) -> str:
    """Normalize Unicode (NFKC), strip zero-width chars, strip HTML safely, and cap length."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    visible_only = strip_invisible_characters(normalized)
    stripped = strip_html_safely(visible_only)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    return collapsed[:max_chars]


def fetch_and_clean(
    payload: RawTriggerPayload,
    max_body_chars: int = DEFAULT_CONFIG.max_clean_body_chars,
) -> CleanedPayload:
    """Clean and normalize raw input deterministically with 0 LLM tokens.

    - Normalizes Unicode (`NFKC`) and removes zero-width / format (`Cf`) characters
    - Strips HTML/script/style tags without dropping text after `< 5` or `x<y`
    - Extracts explicit-scheme, protocol-relative, `www.`, and bare-domain URLs safely
    - Scans both raw and cleaned text for prompt injection, leetspeak, and label-steering
    """
    raw_combined = f"{payload.raw_subject}\n{payload.raw_body}"
    normalized_raw = strip_invisible_characters(unicodedata.normalize("NFKC", raw_combined))

    # Extract URLs before stripping tags so anchor hrefs or plain-text URLs are captured
    extracted_urls = extract_urls_safely(normalized_raw)

    clean_subject = _normalize_and_clean_text(payload.raw_subject, max_chars=500)
    clean_body = _normalize_and_clean_text(payload.raw_body, max_chars=max_body_chars)

    # Scan raw text, normalized text, and cleaned text for injection / label steering
    clean_combined = f"{clean_subject}\n{clean_body}"
    suspicious = (
        detect_prompt_injection(raw_combined)
        or detect_prompt_injection(normalized_raw)
        or detect_prompt_injection(clean_combined)
    )

    word_count = len(_WORD_RE.findall(clean_body))

    return CleanedPayload(
        event_id=payload.event_id,
        sender=payload.sender,
        clean_subject=clean_subject,
        clean_body=clean_body,
        extracted_urls=extracted_urls,
        word_count=word_count,
        contains_suspicious_patterns=suspicious,
    )
