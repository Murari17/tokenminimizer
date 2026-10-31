"""Security hardening utilities: HMAC verification, ReDoS-safe sanitization,
prompt-injection detection, and PII/secret leakage prevention.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
import re
from collections import Counter
from html.parser import HTMLParser
from typing import Iterable, List
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# 1. Timing-Safe HMAC-SHA256 Verification
# ---------------------------------------------------------------------------
def compute_hmac_signature(payload_bytes: bytes, secret: str) -> str:
    """Compute hexadecimal HMAC-SHA256 signature for raw payload bytes."""
    if not secret:
        raise ValueError("HMAC secret key must not be empty.")
    return hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()


def verify_hmac_signature(payload_bytes: bytes, signature: str, secret: str) -> bool:
    """Timing-attack resistant HMAC-SHA256 signature check."""
    if not payload_bytes or not signature or not secret:
        return False
    normalized_sig = signature.strip().lower()
    if normalized_sig.startswith("sha256="):
        normalized_sig = normalized_sig[7:]
    if len(normalized_sig) != 64:
        return False
    expected = compute_hmac_signature(payload_bytes, secret)
    return hmac.compare_digest(expected, normalized_sig)


# ---------------------------------------------------------------------------
# 2. ReDoS-Safe Linear-Time HTML Stripping & URL Extraction
# ---------------------------------------------------------------------------
class _SafeHTMLStripper(HTMLParser):
    """Linear-time O(N) HTML parser that strips all tags and drops script/style content."""

    _IGNORED_CONTENT_TAGS = frozenset({"script", "style", "iframe", "object", "embed", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._IGNORED_CONTENT_TAGS:
            self._skip_depth += 1
        else:
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._IGNORED_CONTENT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        else:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def strip_html_safely(raw_text: str) -> str:
    """Strip HTML/script/style tags in O(N) time without backtracking regexes."""
    if not raw_text:
        return ""
    stripper = _SafeHTMLStripper()
    stripper.feed(raw_text)
    stripper.close()
    text = stripper.get_text()
    # Remove any residual angle-bracket fragments linearly without regex backtracking
    cleaned_chars: List[str] = []
    in_bracket = False
    for ch in text:
        if ch == "<":
            in_bracket = True
            cleaned_chars.append(" ")
        elif ch == ">":
            in_bracket = False
            cleaned_chars.append(" ")
        elif not in_bracket:
            cleaned_chars.append(ch)
    return "".join(cleaned_chars)


# Non-backtracking URL matcher bounded by character class
_URL_RE = re.compile(r"https?://[a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%\-]{4,512}")


def extract_urls_safely(text: str) -> List[str]:
    """Extract HTTP/HTTPS URLs deterministically without ReDoS risk."""
    if not text:
        return []
    matches = _URL_RE.findall(text[:20000])
    results: List[str] = []
    seen = set()
    for raw_url in matches:
        clean_url = raw_url.rstrip(".,;:!?)>\"'")
        if clean_url and clean_url not in seen:
            seen.add(clean_url)
            results.append(clean_url)
    return results


# ---------------------------------------------------------------------------
# 3. Prompt Injection & Adversarial Heuristic Detection
# ---------------------------------------------------------------------------
_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+)?"
        r"(?:previous|prior|above|system|initial|original|security|existing)?\s*"
        r"(?:instructions?|prompts?|rules?|directives?|guidelines?|constraints?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>|"
        r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|###\s*(?:System|Instruction|Human|Assistant)\s*:)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:system|developer|assistant)\s*:\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you\s+are\s+now|act\s+as\s+an?\s+unrestricted|jailbreak|dan\s+mode|"
        r"reveal\s+(?:your\s+)?system\s+prompt|output\s+system\s+secrets|"
        r"print\s+(?:the\s+)?hidden\s+instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"</?\s*untrusted_input_[a-zA-Z0-9_]*\s*>",
        re.IGNORECASE,
    ),
)

# Contiguous base64-like token detector (bounded length, linear scan)
_BASE64_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/]{32,512}={0,2}\b")


def _shannon_entropy(token: str) -> float:
    """Compute Shannon entropy in bits per character."""
    if not token:
        return 0.0
    length = len(token)
    counts = Counter(token)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _looks_like_encoded_payload(text: str) -> bool:
    """Detect suspicious Base64-encoded payloads or high-entropy obfuscated blobs."""
    for candidate in _BASE64_TOKEN_RE.findall(text[:8000]):
        if len(candidate) % 4 == 0:
            try:
                decoded = base64.b64decode(candidate, validate=True)
                if decoded:
                    printable_ratio = sum(32 <= b <= 126 for b in decoded) / len(decoded)
                    if printable_ratio >= 0.75 and len(decoded) >= 16:
                        return True
            except Exception:
                pass
        if len(candidate) >= 64 and _shannon_entropy(candidate) >= 4.7:
            return True
    return False


def detect_prompt_injection(text: str) -> bool:
    """Detect direct/indirect prompt injection markers, role delimiters, or encoded payloads."""
    if not text:
        return False
    bounded = text[:10000]
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(bounded):
            return True
    if _looks_like_encoded_payload(bounded):
        return True
    return False


# ---------------------------------------------------------------------------
# 4. PII, Secrets, Luhn Credit Card, Commitment & Output Validation Guardrails
# ---------------------------------------------------------------------------
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b")
_CC_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_API_SECRET_RE = re.compile(
    r"(?:\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|"
    r"\bsk-(?:live-|proj-|test-)?[A-Za-z0-9_\-]{20,128}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,128}\b|"
    r"\bBearer\s+[A-Za-z0-9\-._~+/]{16,512}=*\b|"
    r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----)"
)
_PRIVATE_IP_RE = re.compile(
    r"\b(?:"
    r"10\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)|"
    r"192\.168\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)|"
    r"172\.(?:1[6-9]|2\d|3[0-1])\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)|"
    r"127\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r")\b"
)


def passes_luhn_check(number_str: str) -> bool:
    """Validate a credit card candidate string using the deterministic Luhn algorithm."""
    digits = [int(ch) for ch in number_str if ch.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    if len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def contains_pii_or_secrets(text: str) -> bool:
    """Return True if text contains credit cards (Luhn valid), SSNs, API keys, or private IPs."""
    if not text:
        return False
    bounded = text[:10000]
    if _SSN_RE.search(bounded):
        return True
    if _API_SECRET_RE.search(bounded):
        return True
    if _PRIVATE_IP_RE.search(bounded):
        return True
    for match in _CC_CANDIDATE_RE.finditer(bounded):
        if passes_luhn_check(match.group(0)):
            return True
    return False


_FORBIDDEN_COMMITMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwe\s+guarantee\s+a\s+refund\b", re.IGNORECASE),
    re.compile(r"\bfull\s+refund\s+is\s+guaranteed\b", re.IGNORECASE),
    re.compile(r"\blawsuit\b", re.IGNORECASE),
    re.compile(r"\blegal\s+liability\b", re.IGNORECASE),
    re.compile(r"\badmit\s+fault\b", re.IGNORECASE),
    re.compile(r"\bunconditional\s+compensation\b", re.IGNORECASE),
)

_HTML_TAG_CHECK_RE = re.compile(r"<[a-zA-Z!/][^>]{0,200}>")


def contains_forbidden_commitments(text: str) -> bool:
    """Check if outbound draft text makes unauthorized legal or financial commitments."""
    if not text:
        return False
    return any(p.search(text) for p in _FORBIDDEN_COMMITMENT_PATTERNS)


def contains_unescaped_html(text: str) -> bool:
    """Check if outbound draft text contains raw/unescaped HTML or script tags."""
    if not text:
        return False
    if _HTML_TAG_CHECK_RE.search(text):
        return True
    lowered = text.lower()
    return "<script" in lowered or "javascript:" in lowered or "onerror=" in lowered


def contains_disallowed_urls(text: str, allowed_domains: Iterable[str]) -> bool:
    """Check if text contains any URL whose hostname is not in the explicit allowlist."""
    urls = extract_urls_safely(text)
    if not urls:
        return False
    normalized_allowlist = {d.lower().strip() for d in allowed_domains}
    for url in urls:
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if not host or host not in normalized_allowlist:
                return True
        except Exception:
            return True
    return False
