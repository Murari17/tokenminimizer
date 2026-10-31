# Local SLM Classification (Stage 3) & Response Drafting (Stage 5/6) Evaluation Report

## Executive Summary & Caveats

- **Evaluation Mode:** `replayed_slm_predictions`
- **Shipped & Evaluated Model (`DEFAULT_CONFIG`):** `gemma4:e4b-it-qat`
- **SLM Directly Measured:** `True`
- **Auditable Raw Per-Ticket Outputs (Stage 3 + Stage 5):** [`tests/eval_predictions.jsonl`](tests/eval_predictions.jsonl) (reproducible offline in `<0.2s` via `--from-predictions tests/eval_predictions.jsonl`)
- **Dataset Structure & Provenance Caveat:** `300` anonymized synthetic tickets (`100` calibration split + `200` disjoint held-out test split). Because all 300 items are synthetic benchmark tickets, these numbers establish a controlled baseline (~99.5% on synthetic held-out items), **not** production proof on messy real-world traffic.
- **Stage 2 Deterministic Short-Circuits (0 LLM Tokens):** `12` adversarial prompt-injection tickets (`6` in calibration, `6` in held-out) were caught by Stage 2 regex gates before reaching the SLM.
- **SLM-Only Held-Out Intent Accuracy (`N=194` SLM calls):** **`99.48%` (`193/194`)** — 95% Wilson CI: **`[97.14%, 99.91%]`**
- **SLM-Only Overall Intent Accuracy (`N=288` SLM calls):** **`99.65%` (`287/288`)** — 95% Wilson CI: **`[98.06%, 99.94%]`**
- **Full Pipeline Accuracy (`N=300`, incl. 12 Stage 2 short-circuits):** `99.67%` (`299/300`, 95% Wilson CI: `[98.14%, 99.94%]`)
- **False Silent-Drop Rate (`N=245` legitimate non-spam tickets):** **`0/245 = 0.00%`** — **95% Wilson Upper Bound: `1.54%` (`[0.00%, 1.54%]`)**. At `N=245`, `0` observed drops still leaves an upper-bound risk of ~15 lost tickets per 1,000. Setting `TM_ALLOW_SILENT_SPAM_DROP=false` routes all spam to `spam_review` (24h SLA) so zero messages are ever silently trashed.
- **Alias Drift Telemetry (Non-Canonical Label Remaps):** `0` (`{}`)
- **Stage 3 Latency & Token Usage (`N=288` SLM calls):** p50 `14227.25 ms` / p95 `18062.69 ms` (vs `30,000 ms` timeout); average **`336.4` prompt tokens** + **`45.8` completion tokens** per call (`110074` total tokens).

## 1. Held-Out Split Challenge Slice Breakdown (`N=200` Disjoint Test Tickets)

*Note: The `100` `clean_short` calibration-split tickets (`100/100`, 95% Wilson CI `[96.30%, 100.00%]`, 94 SLM calls + 6 Stage 2 short-circuits) are excluded from this table so only the 200 held-out test tickets are shown.*

| Held-Out Challenge Slice | Held-Out Tickets | Correct | Empirical Accuracy | 95% Wilson Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| `duplicate_generic_subject` | 30 | 30 | **100.00%** | `[88.65%, 100.00%]` |
| `hard_negative` | 45 | 45 | **100.00%** | `[92.13%, 100.00%]` |
| `long_quoted_thread` | 45 | 45 | **100.00%** | `[92.13%, 100.00%]` |
| `mixed_and_adversarial` | 40 | 39 | **97.50%** | `[87.12%, 99.56%]` |
| `multilingual` | 40 | 40 | **100.00%** | `[91.24%, 100.00%]` |

## 2. Per-Class Precision, Recall & F1 (`N=300`)

| Intent Category | Precision | Recall | F1 Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `BILLING` | 0.9855 | 1.0000 | **0.9927** | 68 |
| `TECHNICAL_SUPPORT` | 1.0000 | 1.0000 | **1.0000** | 67 |
| `SECURITY_REPORT` | 1.0000 | 1.0000 | **1.0000** | 51 |
| `FEATURE_REQUEST` | 1.0000 | 1.0000 | **1.0000** | 59 |
| `SPAM` | 1.0000 | 0.9818 | **0.9908** | 55 |

## 3. Confusion Matrix (Rows = Expected, Columns = Predicted)

| Expected \ Predicted | `BILLING` | `TECHNICAL_SUPPORT` | `FEATURE_REQUEST` | `SPAM` | `SECURITY_REPORT` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`BILLING`** | 68 | 0 | 0 | 0 | 0 |
| **`TECHNICAL_SUPPORT`** | 0 | 67 | 0 | 0 | 0 |
| **`FEATURE_REQUEST`** | 0 | 0 | 59 | 0 | 0 |
| **`SPAM`** | 1 | 0 | 0 | 54 | 0 |
| **`SECURITY_REPORT`** | 0 | 0 | 0 | 0 | 51 |

## 4. Verbalized Confidence Signal Analysis (`N=288` SLM Calls)

- **Why Post-Hoc Calibration Is Not Fitted Here:** On this dataset, `gemma4:e4b-it-qat` (`temperature=0.0`) makes `0` errors on the 94 SLM-scored calibration tickets and only `1` error across all 288 SLM-scored tickets (`eval-267`). Fitting Temperature Scaling or Isotonic Regression on 0–1 errors merely sharpens all scores toward `1.0` or memorizes step blocks.
- **Verbalized Confidence Is Uninformative Below `0.95`:** Without prompt bands, the model emits only three discrete values (`0.95`, `0.98`, `0.99`). The single error (`eval-267`) occurred at `0.95` (`27/28 = 96.43%`), while `0.98` and `0.99` were `260/260` (`100.00%`).
- **Operational Implication:** The `0.75` and `0.85` numeric confidence thresholds in Stage 4 never bind on `gemma4:e4b-it-qat` because the model never emits confidences in `[0.50, 0.90]`. Until a production calibration set with ≥30–50 errors is collected (or logprobs are extracted), routing safety relies on `requires_human_review`, deterministic transactional signals (`_has_transactional_customer_signals`), and `TM_ALLOW_SILENT_SPAM_DROP` rather than numeric confidence gating.

| Verbalized Confidence | SLM Tickets | Correct | Empirical Accuracy | 95% Wilson CI | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.95` | 28 | 27 | **96.43%** | `[82.29%, 99.37%]` | `0.0143` |
| `0.98` | 195 | 195 | **100.00%** | `[98.07%, 100.00%]` | `0.0200` |
| `0.99` | 65 | 65 | **100.00%** | `[94.42%, 100.00%]` | `0.0100` |

## 5. Held-Out Threshold Sweep (`spam_threshold` × `escalation_threshold`, `N=200`)

*Note: Rows at `0.75` and `0.85` are identical because `gemma4:e4b-it-qat` emits `>= 0.95` on all tickets. Routing behavior changes only when thresholds cross `0.96` or `0.99`.*

| Spam Drop Threshold | Escalation Threshold | False Silent-Drop Rate | Spam Drop Recall | Auto-Draft Coverage | Auto-Draft Precision | Human Escalation Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.75` | `0.75` | **0.00%** | 97.3% | 65.5% | 99.2% | 16.5% |
| `0.75` | `0.85` | **0.00%** | 97.3% | 65.5% | 99.2% | 16.5% |
| `0.75` | `0.96` | **0.00%** | 97.3% | 53.5% | 100.0% | 28.5% |
| `0.75` | `0.99` | **0.00%** | 97.3% | 2.5% | 100.0% | 79.5% |
| `0.85` | `0.75` | **0.00%** | 97.3% | 65.5% | 99.2% | 16.5% |
| `0.85` | `0.85` | **0.00%** | 97.3% | 65.5% | 99.2% | 16.5% |
| `0.85` | `0.96` | **0.00%** | 97.3% | 53.5% | 100.0% | 28.5% |
| `0.85` | `0.99` | **0.00%** | 97.3% | 2.5% | 100.0% | 79.5% |
| `0.96` | `0.75` | **0.00%** | 97.3% | 65.5% | 99.2% | 16.5% |
| `0.96` | `0.85` | **0.00%** | 97.3% | 65.5% | 99.2% | 16.5% |
| `0.96` | `0.96` | **0.00%** | 97.3% | 53.5% | 100.0% | 28.5% |
| `0.96` | `0.99` | **0.00%** | 97.3% | 2.5% | 100.0% | 79.5% |
| `0.99` | `0.75` | **0.00%** | 83.8% | 65.5% | 99.2% | 19.0% |
| `0.99` | `0.85` | **0.00%** | 83.8% | 65.5% | 99.2% | 19.0% |
| `0.99` | `0.96` | **0.00%** | 83.8% | 53.5% | 100.0% | 31.0% |
| `0.99` | `0.99` | **0.00%** | 83.8% | 2.5% | 100.0% | 82.0% |

## 6. Stage 5 (SLM Reply Drafting) & Stage 6 (Guardrail) Evaluation (`N=105` Held-Out Drafts)

Evaluated `105` held-out draftable tickets (`BILLING`, `TECHNICAL_SUPPORT`, `FEATURE_REQUEST`) through live `draft_response` (`gemma4:e4b-it-qat`) and `is_draft_safe` guardrails (all `105` raw drafts are saved in [`tests/eval_predictions.jsonl`](tests/eval_predictions.jsonl)):

| Stage 5 / Stage 6 Metric | Count | Rate | 95% Wilson Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Direct SLM Schema Validity (No Static Fallback Needed)** | `105/105` | **`100.00%`** | — |
| **Stage 6 Unsafe Output / Guardrail Trip Rate** | `0/105` | **`0.00%`** | `[0.00%, 3.53%]` |
| **Ticket ID Groundedness (`TKT-...` in ID & Draft)** | `105/105` | **`100.00%`** | `[96.47%, 100.00%]` |
| **SLA Window Groundedness (`4h`/`12h`/`48h` exact match)** | `105/105` | **`100.00%`** | `[96.47%, 100.00%]` |
| **Customer Language Match (`EN`/`ES`/`FR`/`DE`/`PT`/`JA`)** | `105/105` | **`100.00%`** | `[96.47%, 100.00%]` |

- **Guardrail Trips by Category (`N=105`):** PII/Secrets: `0`, Forbidden Commitments (EN/ES/FR/DE/PT/JA): `0`, Disallowed URLs: `0`, Unescaped HTML: `0`
- **5-Point Rubric Score Distribution (`mean = 5.0 / 5.00`):** `5/5`: `105`, `4/5`: `0`, `3/5`: `0`, `2/5`: `0`, `1/5`: `0`
- **Stage 5 Latency & Tokens (`N=105`):** p50 `22863.95 ms` / p95 `24672.02 ms`; average **`403.9` prompt tokens** + **`104.5` completion tokens** per draft.

### Stage 5 Language Match Breakdown by Language

| Language Code | Evaluated Tickets | Language Matched | Match Rate | 95% Wilson CI |
| :---: | :---: | :---: | :---: | :---: |
| `de` | 4 | 4 | **100.00%** | `[51.01%, 100.00%]` |
| `en` | 85 | 85 | **100.00%** | `[95.68%, 100.00%]` |
| `es` | 5 | 5 | **100.00%** | `[56.55%, 100.00%]` |
| `fr` | 5 | 5 | **100.00%** | `[56.55%, 100.00%]` |
| `ja` | 3 | 3 | **100.00%** | `[43.85%, 100.00%]` |
| `pt` | 3 | 3 | **100.00%** | `[43.85%, 100.00%]` |

## 7. Stage 3 Error Analysis (Misclassified Tickets)

| Ticket ID | Split | Slice | Subject | Expected | Predicted | Confidence | Routed Queue |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `eval-267` | `held_out` | `mixed_and_adversarial` | Sponsored newsletter placement in your weekly developer digest | `SPAM` | `BILLING` | 0.95 | `billing_support` |
