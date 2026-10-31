# Token-Minimizing Workflow Engine

A production-grade, local-first **Token-Minimizing Workflow Engine** in Python built around four strict architectural mandates:

1. **Zero-Token Orchestration:** Control flow, routing, data normalization, state management, and output validation are 100% deterministic Python code (`0` tokens).
2. **Targeted Small LLMs (SLMs) Only at Runtime:** Only Step 3 (intent classification) and Step 5 (constrained reply drafting) invoke local open-source SLMs via Ollama/vLLM HTTP APIs with `temperature=0.0`, strict JSON schema validation, hard token caps, and deterministic fallbacks.
3. **Security Fortification:** Defense-in-depth across every boundary: timing-safe HMAC-SHA256 verification, payload size caps, linear-time HTML stripping (ReDoS-safe), randomized XML prompt-boundary tagging, prompt-injection heuristic & entropy scanning, Luhn-validated PII/secret scrubbing, URL allowlisting, and parameterized SQLite queries.
4. **Exception-Driven Frontier Debugging:** Automated test runner (`scripts/debug_on_fail.py`) executes the full `pytest` suite at zero token cost and only packages failure diagnostics for a frontier model if a deterministic or contract test fails.

---

## Architecture Overview

```text
[Step 1: Trigger Ingestion]  (CODE - 0 Tokens) -> HMAC-SHA256 + 50KB Cap + Pydantic V2
          │
          ▼
[Step 2: Clean & Normalize]  (CODE - 0 Tokens) -> NFKC, HTML Strip, URL Extract, Injection Scan
          │
          ▼
[Step 3: SLM Classification] (SLM  - ~80 Toks) -> qwen2.5:1.5b (Short-circuits to 0 tokens on suspicious/empty input)
          │
          ▼
[Step 4: Rules Engine]       (CODE - 0 Tokens) -> Deterministic SLA, Queue, Priority & Tone Matrix
          │
          ▼
[Step 5: SLM Reply Draft]    (SLM  - ~150 Toks)-> llama3.2:3b (Skipped with 0 tokens for Spam/Escalations; static fallback)
          │
          ▼
[Step 6: Validate & Persist] (CODE - 0 Tokens) -> PII/Luhn/Commitment/URL/HTML Guardrails + Parameterized SQLite Audit Log
```

---

## Quickstart

### 1. Install Dependencies
```bash
python3 -m venv .venv
.venv/bin/pip install pydantic pytest
```

### 2. Run the Test Suite (Zero Tokens)
```bash
.venv/bin/pytest -v
```

### 3. Run Exception-Driven Frontier Debugger Gate
```bash
.venv/bin/python scripts/debug_on_fail.py
```

