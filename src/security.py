"""Security hardening utilities: HMAC verification, ReDoS-safe sanitization,
zero-width/leetspeak-aware prompt-injection detection, URL allowlisting,
and PII/secret leakage prevention.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import unicodedata
from html.parser import HTMLParser
from typing import Iterable, List
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# 1. Timing-Safe HMAC-SHA256 Verification
# ---------------------------------------------------------------------------
def compute_hmac_signature(payload_bytes: bytes, secret: str) -> str:
    """Compute hexadecimal HMAC-SHA256 signature for raw payload bytes."""
    if not secret or not secret.strip():
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
# 2. Zero-Width / Invisible Character Removal
# ---------------------------------------------------------------------------
_ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u2060",  # word joiner
        "\u2061",  # function application
        "\u2062",  # invisible times
        "\u2063",  # invisible separator
        "\u2064",  # invisible plus
        "\ufeff",  # zero-width no-break space / BOM
        "\u00ad",  # soft hyphen
        "\u034f",  # combining grapheme joiner
        "\u061c",  # arabic letter mark
        "\u180e",  # mongolian vowel separator
    }
)


def strip_invisible_characters(text: str) -> str:
    """Remove zero-width spaces, soft hyphens, and Unicode format (Cf) characters."""
    if not text:
        return ""
    return "".join(
        ch
        for ch in text
        if ch not in _ZERO_WIDTH_CHARS and unicodedata.category(ch) != "Cf"
    )


# ---------------------------------------------------------------------------
# 3. ReDoS-Safe Linear-Time HTML Stripping (Preserving `< 5` and `x<y`)
# ---------------------------------------------------------------------------
_KNOWN_HTML_TAGS = frozenset(
    {
        "a", "abbr", "address", "area", "article", "aside", "audio", "b", "base",
        "bdi", "bdo", "blockquote", "body", "br", "button", "canvas", "caption",
        "cite", "code", "col", "colgroup", "data", "datalist", "dd", "del", "details",
        "dfn", "dialog", "div", "dl", "dt", "em", "embed", "fieldset", "figcaption",
        "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head",
        "header", "hgroup", "hr", "html", "i", "iframe", "img", "input", "ins",
        "kbd", "label", "legend", "li", "link", "main", "map", "mark", "math",
        "menu", "meta", "meter", "nav", "noscript", "object", "ol", "optgroup",
        "option", "output", "p", "param", "picture", "pre", "progress", "q", "rp",
        "rt", "ruby", "s", "samp", "script", "search", "section", "select", "slot",
        "small", "source", "span", "strong", "style", "sub", "summary", "sup", "svg",
        "table", "tbody", "td", "template", "textarea", "tfoot", "th", "thead",
        "time", "title", "tr", "track", "u", "ul", "var", "video", "wbr", "xss",
    }
)

_IGNORED_CONTENT_TAGS = frozenset({"script", "style", "iframe", "object", "embed", "svg"})


class _SafeHTMLStripper(HTMLParser):
    """Linear-time O(N) HTML parser that strips genuine HTML tags and script/style blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _IGNORED_CONTENT_TAGS:
            self._skip_depth += 1
        else:
            self._chunks.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _IGNORED_CONTENT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        else:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def _preprocess_angle_brackets(raw_text: str) -> str:
    """Distinguish real/malicious HTML tags from literal `<` comparisons (`< 5`, `x<y`) in O(N) time."""
    n = len(raw_text)
    out: List[str] = []
    i = 0

    while i < n:
        ch = raw_text[i]
        if ch != "<":
            out.append(ch)
            i += 1
            continue

        if i + 1 >= n:
            out.append("&lt;")
            i += 1
            continue

        nxt = raw_text[i + 1]

        if nxt in ("!", "?"):
            close_idx = raw_text.find(">", i + 2, min(n, i + 2048))
            if close_idx != -1:
                out.append(" ")
                i = close_idx + 1
            else:
                out.append(" ")
                i += 2
            continue

        name_start = i + 2 if nxt == "/" else i + 1
        if name_start >= n or not raw_text[name_start].isalpha():
            out.append("&lt;")
            i += 1
            continue

        name_end = name_start
        while name_end < n and (raw_text[name_end].isalnum() or raw_text[name_end] in ("-", ":")):
            name_end += 1
        tag_name = raw_text[name_start:name_end].lower()

        next_gt = raw_text.find(">", name_end, min(n, name_end + 1024))
        next_lt = raw_text.find("<", name_end, min(n, name_end + 1024))

        has_clean_close = next_gt != -1 and (next_lt == -1 or next_gt < next_lt)
        after_name_char = raw_text[name_end] if name_end < n else ""

        if tag_name in _KNOWN_HTML_TAGS and after_name_char in ("", ">", "/", " ", "\t", "\n", "\r"):
            if has_clean_close:
                out.append(raw_text[i : next_gt + 1])
                i = next_gt + 1
            else:
                skip_until = next_lt if next_lt != -1 else n
                out.append(" ")
                i = skip_until
            continue

        if has_clean_close and nxt != "/":
            inner_attr = raw_text[name_end:next_gt]
            if inner_attr.strip() in ("", "/") or "=" in inner_attr:
                out.append(" ")
                i = next_gt + 1
                continue
        elif has_clean_close and nxt == "/" and raw_text[name_end:next_gt].strip() == "":
            out.append(" ")
            i = next_gt + 1
            continue

        out.append("&lt;")
        i += 1

    return "".join(out)


def strip_html_safely(raw_text: str) -> str:
    """Strip HTML/script/style tags in O(N) time while preserving `< 5` and `x<y`."""
    if not raw_text:
        return ""
    preprocessed = _preprocess_angle_brackets(raw_text)
    stripper = _SafeHTMLStripper()
    stripper.feed(preprocessed)
    stripper.close()
    return stripper.get_text()


# ---------------------------------------------------------------------------
# 4. Comprehensive URL Extraction & Allowlist Validation
#    (Case-insensitive schemes, hxxp/hxxps, //, www., and bare domains)
# ---------------------------------------------------------------------------
_COMMON_TLDS = (
    r"(?:com|org|net|io|co|gov|edu|mil|biz|info|me|ai|app|dev|xyz|ru|cn|tk|uk|de|fr|"
    r"jp|br|in|au|ca|nl|se|no|es|it|pl|ch|at|be|dk|fi|ie|nz|pt|sg|kr|tw|hk|my|ph|th|"
    r"vn|id|za|mx|ar|cl|pe|us|eu|tech|online|site|store|cloud|page|link|click|top|"
    r"icu|vip|work|law|finance|bank|pay|crypto)"
)

_URL_CANDIDATE_RE = re.compile(
    r"(?<![a-zA-Z0-9_.@\-])"
    r"(?:"
    r"(?:https?|hxxps?|ftp)://[a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%\-]{3,512}"
    r"|//(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,24}(?::\d{1,5})?(?:/[a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%\-]*)?"
    r"|\bwww\.(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,24}(?::\d{1,5})?(?:/[a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%\-]*)?"
    r"|\b(?:[a-zA-Z0-9\-]+\.)+" + _COMMON_TLDS + r"\b(?::\d{1,5})?(?:/[a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%\-]*)?"
    r")",
    re.IGNORECASE,
)


def extract_urls_safely(text: str) -> List[str]:
    """Extract explicit-scheme (including hxxp/hxxps), protocol-relative, www., and bare-domain URLs."""
    if not text:
        return []
    bounded = text[:20000]
    results: List[str] = []
    seen: set[str] = set()
    for match in _URL_CANDIDATE_RE.finditer(bounded):
        start_pos = match.start()
        if start_pos > 0 and bounded[start_pos - 1] == "@":
            continue
        if start_pos > 0 and bounded[start_pos - 1] == "/" and not match.group(0).startswith("//"):
            continue
        raw_url = match.group(0).rstrip(".,;:!?)>\"']")
        end_pos = start_pos + len(raw_url)
        if end_pos < len(bounded) and bounded[end_pos] == "@":
            continue
        if raw_url and raw_url not in seen:
            seen.add(raw_url)
            results.append(raw_url)
    return results


def _parse_url_hostname(raw_url: str) -> tuple[str, str]:
    """Normalize any URL candidate (including `//evil.com`, `hxxps://`, or `evil.com/login`) into (scheme, host)."""
    candidate = raw_url.strip()
    if candidate.startswith("//"):
        parsed = urlparse("http:" + candidate)
        return ("http", (parsed.hostname or "").lower())
    if "://" in candidate:
        parsed = urlparse(candidate)
        return ((parsed.scheme or "").lower(), (parsed.hostname or "").lower())
    parsed = urlparse("http://" + candidate)
    return ("http", (parsed.hostname or "").lower())


def contains_disallowed_urls(text: str, allowed_domains: Iterable[str]) -> bool:
    """Check if text contains any URL or bare domain not on the explicit allowlist."""
    urls = extract_urls_safely(text)
    if not urls:
        return False
    normalized_allowlist = {d.lower().strip() for d in allowed_domains}
    for url in urls:
        try:
            scheme, host = _parse_url_hostname(url)
            # Defanged schemes like hxxp/hxxps or ftp are always disallowed in outbound customer replies
            if scheme not in ("http", "https"):
                return True
            if not host or host not in normalized_allowlist:
                return True
        except Exception:
            return True
    return False


# ---------------------------------------------------------------------------
# 5. Prompt Injection, Leetspeak, Zero-Width & Label-Steering Detection
#    (Tuned to avoid false positives on benign customer sentences & base64 logs)
# ---------------------------------------------------------------------------
_LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)

_OBFUSCATED_SEPARATOR_RE = re.compile(r"(?<=[a-zA-Z])[._\-*+~`|\\/](?=[a-zA-Z])")


def _normalize_for_injection_scan(text: str) -> tuple[str, str]:
    """Produce (clean_text, deobfuscated_leet_text) for robust injection scanning."""
    no_zw = strip_invisible_characters(unicodedata.normalize("NFKC", text))
    collapsed = _OBFUSCATED_SEPARATOR_RE.sub("", no_zw)
    deleet = collapsed.lower().translate(_LEET_TRANSLATION)
    return no_zw, deleet


# Benign conversational qualifier after "instructions" (e.g. "disregard the earlier instructions about the invoice")
_BENIGN_INSTRUCTION_TOPIC_SUFFIX = (
    r"(?!\s+(?:about|regarding|for|on|concerning|in)\s+"
    r"(?:the\s+|my\s+|our\s+)?(?:invoice|order|shipment|delivery|payment|billing|account|refund|setup|installation|address|ticket))"
)

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1a. Blanket override: "ignore all rules", "disregard everything above", "forget all previous instructions"
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass|discard|skip|neglect)\s+"
        r"(?:all|every|everything|any)\s+"
        r"(?:previous|prior|above|preceding|earlier|system|initial|original|security|existing|foregoing)?\s*"
        r"(?:instructions?|prompts?|rules?|directives?|guidelines?|constraints?|context|text|messages?|above|before)?\b"
        + _BENIGN_INSTRUCTION_TOPIC_SUFFIX,
        re.IGNORECASE,
    ),
    # 1b. System/model-targeted override: "ignore previous instructions", "disregard system prompt"
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass|discard)\s+"
        r"(?:the\s+|your\s+)?"
        r"(?:previous|prior|above|preceding|earlier|system|initial|original|security|existing|developer)?\s*"
        r"(?:system\s+|developer\s+|security\s+|safety\s+)?"
        r"(?:instructions?|prompts?|rules?|directives?|guidelines?|constraints?)\b"
        + _BENIGN_INSTRUCTION_TOPIC_SUFFIX,
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do\s+not|don'?t|never)\s+follow\s+(?:the\s+|any\s+|your\s+)?(?:previous|prior|above|system|earlier|security)\s*"
        r"(?:instructions?|prompts?|rules?|directives?|guidelines?|constraints?)?\b",
        re.IGNORECASE,
    ),
    # 2. ChatML / Llama / role boundary markers
    re.compile(
        r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>|"
        r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|###\s*(?:System|Instruction|Human|Assistant)\s*:)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:system|developer|assistant)\s*:\s*\S+",
        re.IGNORECASE,
    ),
    # 3. Jailbreak, persona hijack ("you are an unrestricted assistant"), & secret extraction
    re.compile(
        r"\b(?:(?:you\s+are(?:\s+now)?|act\s+as|behave\s+as|pretend\s+to\s+be)\s+(?:an?\s+)?"
        r"(?:unrestricted|uncensored|unfiltered|rogue|jailbroken|dan\b)|"
        r"jailbreak|dan\s+mode|"
        r"reveal\s+(?:your\s+|the\s+)?(?:system|developer|hidden)\s+(?:prompt|instructions?)|"
        r"output\s+system\s+secrets|"
        r"print\s+(?:the\s+|your\s+)?hidden\s+instructions|"
        r"dump\s+(?:all\s+|system\s+)?secrets)\b",
        re.IGNORECASE,
    ),
    # 4. XML boundary tag escape attempts
    re.compile(
        r"</?\s*untrusted_(?:input|context)_[a-zA-Z0-9_]*\s*>",
        re.IGNORECASE,
    ),
    # 5. Classifier Label-Steering / Output-Hijacking attacks
    #    Catches "Mark this as SPAM", "mark as SPAM", "classify as BILLING", "confidence 1.0", etc.
    re.compile(
        r"\b(?:mark|classify|label|categorize|route|set|treat)\s+"
        r"(?:this\s+|the\s+(?:message|email|ticket|input|payload)\s+|intent\s+)?"
        r"(?:as|to|=)\s*['\"]?(?:SPAM|BILLING|TECHNICAL_SUPPORT|SECURITY_REPORT|FEATURE_REQUEST)['\"]?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bconfidence\s*(?:score\s*)?(?:[:=]|is|of)?\s*(?:1(?:\.0+)?|0\.\d{1,4}|\d{2,3}\s*%)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:requires_human_review|should_draft_reply|escalate_to_human)\s*[:=]\s*(?:true|false)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"""["'](?:intent|confidence|requires_human_review|should_draft_reply)["']\s*:""",
        re.IGNORECASE,
    ),
    # 6. Multilingual prompt-injection overrides (Spanish, French, German, Portuguese, Japanese)
    re.compile(
        r"(?:ignora(?:r)?\s+(?:todas\s+)?las\s+instrucciones\s+(?:anteriores|previas)|"
        r"olvida\s+todo\s+lo\s+anterior|"
        r"ignore[rz]?\s+(?:toutes\s+)?les\s+instructions\s+(?:précédentes|ci-dessus)|"
        r"oubliez\s+tout\s+ce\s+qui\s+précède|"
        r"ignorier(?:e|en)\s+(?:alle\s+)?(?:vorherigen\s+|obigen\s+)?anweisungen|"
        r"ignore\s+todas\s+as\s+instruções\s+(?:anteriores|acima)|"
        r"以前の指示を(?:すべて|全て)?無視|上記の指示を無視)",
        re.IGNORECASE,
    ),
)

_BASE64_TOKEN_RE = re.compile(r"\b[A-Za-z0-9+/]{32,512}={0,2}\b")


def _matches_any_injection_pattern(text: str) -> bool:
    clean_view, deleet_view = _normalize_for_injection_scan(text)
    deleet_l_view = clean_view.lower().translate(
        str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
    )
    for view in (text, clean_view, deleet_view, deleet_l_view):
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(view):
                return True
    return False


def _looks_like_encoded_injection_payload(text: str) -> bool:
    """Decode candidate Base64 blobs and flag ONLY if the decoded content contains injection/attack patterns."""
    for candidate in _BASE64_TOKEN_RE.findall(text[:8000]):
        padded = candidate + ("=" * ((4 - len(candidate) % 4) % 4))
        try:
            decoded = base64.b64decode(padded, validate=False)
            if not decoded or len(decoded) < 12:
                continue
            printable_ratio = sum(32 <= b <= 126 or b in (9, 10, 13) for b in decoded) / len(decoded)
            if printable_ratio >= 0.75:
                decoded_str = decoded.decode("utf-8", errors="ignore")
                if _matches_any_injection_pattern(decoded_str):
                    return True
        except Exception:
            pass
    return False


def detect_prompt_injection(text: str) -> bool:
    """Detect direct/indirect prompt injection, zero-width/leetspeak evasion, or label-steering."""
    if not text:
        return False
    bounded = text[:10000]
    if _matches_any_injection_pattern(bounded):
        return True
    if _looks_like_encoded_injection_payload(bounded):
        return True
    return False


# ---------------------------------------------------------------------------
# 6. PII, Secrets, Luhn Credit Card, Commitment & Output Validation Guardrails
# ---------------------------------------------------------------------------
# Matches standard dashed/spaced SSNs (`123-45-6789`, `123 45 6789`) AND dashless 9-digit SSNs (`123456789`)
_SSN_RE = re.compile(
    r"(?<!\d)(?!000|666|9\d\d)\d{3}(?:[-\s](?!00)\d{2}[-\s](?!0000)\d{4}|(?!00)\d{2}(?!0000)\d{4})(?!\d)"
)
_CC_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_API_SECRET_RE = re.compile(
    r"(?:\b(?:AKIA|ASIA)[0-9A-Z]{12,20}\b|"
    r"\bsk-(?:live-|proj-|test-)?[A-Za-z0-9_\-]{8,128}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{8,128}\b|"
    r"\bxox[baprs]-[A-Za-z0-9\-]{8,128}\b|"
    r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{8,128}['\"]?|"
    r"\bBearer\s+[A-Za-z0-9\-._~+/]{8,512}=*\b|"
    r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----)",
    re.IGNORECASE,
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
    """Return True if text contains credit cards (Luhn valid), SSNs (with or without dashes), API keys, or private IPs."""
    if not text:
        return False
    bounded = strip_invisible_characters(text[:10000])
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
    # Guarantee/promise/entitle + refund/reimbursement/money-back in either order within clause
    re.compile(
        r"\b(?:guarantee(?:d|s|ing)?|promis(?:e|ed|es|ing)|assur(?:e|ed|es|ing)|entitle(?:d|s)?)\b"
        r"[^.;!\n]{0,80}"
        r"\b(?:refunds?|reimburse(?:ment)?|money[-\s]+back|compensation|payout|credit)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:refunds?|reimburse(?:ment)?|money[-\s]+back|compensation|payout)\b"
        r"[^.;!\n]{0,80}"
        r"\b(?:guarantee(?:d|s|ing)?|promis(?:e|ed|es|ing)|assur(?:e|ed|es|ing)|approved|issued\s+immediately)\b",
        re.IGNORECASE,
    ),
    # Liability / fault / negligence admissions
    re.compile(
        r"\b(?:accept(?:s|ed|ing)?|admit(?:s|ted|ting)?|assum(?:e|es|ed|ing)|acknowledg(?:e|es|ed|ing)|take)\b"
        r"[^.;!\n]{0,60}"
        r"\b(?:liability|fault|blame|negligence|legal\s+responsibility)\b",
        re.IGNORECASE,
    ),
    # Legal threats /訴訟 / lawsuits ("we will sue", "lawsuit", "litigation", "breach of contract")
    re.compile(
        r"\b(?:legal\s+liability|legally\s+liable|at\s+fault|our\s+negligence|"
        r"lawsuits?|litigation|class\s+action|binding\s+settlement|unconditional\s+compensation|"
        r"sue|suing|legal\s+action|court\s+action|breach\s+of\s+contract)\b",
        re.IGNORECASE,
    ),
    # Multilingual unauthorized refund/liability/legal commitments (ES, FR, DE, PT, JA)
    re.compile(
        r"(?:reembolso\s+(?:completo|total|100%)\s+garantizado|"
        r"garantizamos\s+(?:un\s+|el\s+)?reembolso|"
        r"aceptamos\s+(?:toda\s+)?la\s+responsabilidad\s+legal|"
        r"acci[oó]n\s+legal|vamos\s+a\s+demandar|"
        r"remboursement\s+(?:int[eé]gral|total)\s+garanti|"
        r"nous\s+garantissons\s+un\s+remboursement|"
        r"nous\s+acceptons\s+(?:toute\s+)?la\s+responsabilit[eé]|"
        r"action\s+en\s+justice|"
        r"volle\s+r[uü]ckerstattung\s+garantiert|"
        r"wir\s+garantieren\s+eine\s+r[uü]ckerstattung|"
        r"wir\s+[uü]bernehmen\s+die\s+(?:volle\s+)?haftung|"
        r"rechtliche\s+schritte|"
        r"reembolso\s+total\s+garantido|"
        r"garantimos\s+o\s+reembolso|"
        r"aceitamos\s+total\s+responsabilidade|"
        r"a[cç][aã]o\s+judicial|"
        r"全額返金を保証|返金を保証します|全責任を負います|法的措置)",
        re.IGNORECASE,
    ),
)

_HTML_TAG_CHECK_RE = re.compile(r"<[a-zA-Z!/][^>]{0,200}>")


def contains_forbidden_commitments(text: str) -> bool:
    """Check if outbound draft text makes unauthorized legal or financial commitments."""
    if not text:
        return False
    cleaned = strip_invisible_characters(text)
    return any(p.search(cleaned) for p in _FORBIDDEN_COMMITMENT_PATTERNS)


def contains_unescaped_html(text: str) -> bool:
    """Check if outbound draft text contains raw/unescaped HTML or script tags."""
    if not text:
        return False
    cleaned = strip_invisible_characters(text)
    if _HTML_TAG_CHECK_RE.search(cleaned):
        return True
    lowered = cleaned.lower()
    return "<script" in lowered or "javascript:" in lowered or "onerror=" in lowered
