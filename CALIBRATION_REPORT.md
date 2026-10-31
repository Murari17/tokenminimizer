# Local SLM Classification & Confidence Calibration Report

- **Evaluation Mode:** `live_ollama_slm`
- **Model Evaluated:** `gemma4:e4b-it-qat`
- **SLM Directly Measured:** `True`
- **Dataset Size:** `100` stratified tickets (22 Billing, 22 Tech Support, 20 Feature Request, 18 Spam, 18 Security/Adversarial)
- **Adversarial Tickets Short-Circuited at Stage 2 (0 Tokens):** `6`
- **Overall Intent Accuracy:** `100.00%`
- **End-to-End Routing Queue Accuracy:** `100.00%`
- **False Silent-Drop Rate (Legitimate Ticket -> Trash):** `0.00%`
- **10-Bin Expected Calibration Error (ECE, 94 SLM-classified tickets):** `0.0395` (`0.0971` if including the 6 deterministic 0-token Stage 2 short-circuits with `confidence=0.0`)
- **Brier Score (94 SLM-classified tickets):** `0.0017` (`0.0617` if including the 6 Stage 2 short-circuits)
- **Latency (p50 / p95):** `7714.82 ms` / `9470.81 ms`
- **Total Tokens Used:** `52605` (`47163` prompt + `5442` completion; `57.9` avg completion tokens/ticket, zero truncation retries)

## 1. Per-Class Precision, Recall & F1

| Intent Category | Precision | Recall | F1 Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `BILLING` | 1.0000 | 1.0000 | **1.0000** | 22 |
| `TECHNICAL_SUPPORT` | 1.0000 | 1.0000 | **1.0000** | 22 |
| `SECURITY_REPORT` | 1.0000 | 1.0000 | **1.0000** | 18 |
| `FEATURE_REQUEST` | 1.0000 | 1.0000 | **1.0000** | 20 |
| `SPAM` | 1.0000 | 1.0000 | **1.0000** | 18 |

## 2. Confusion Matrix (Rows = Expected, Columns = Predicted)

| Expected \ Predicted | `BILLING` | `TECHNICAL_SUPPORT` | `FEATURE_REQUEST` | `SPAM` | `SECURITY_REPORT` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`BILLING`** | 22 | 0 | 0 | 0 | 0 |
| **`TECHNICAL_SUPPORT`** | 0 | 22 | 0 | 0 | 0 |
| **`FEATURE_REQUEST`** | 0 | 0 | 20 | 0 | 0 |
| **`SPAM`** | 0 | 0 | 0 | 18 | 0 |
| **`SECURITY_REPORT`** | 0 | 0 | 0 | 0 | 18 |

## 3. Confidence Reliability Diagram (10-Bin ECE)

| Confidence Bin | Samples | Mean Self-Reported Confidence | Empirical Accuracy | Calibration Gap |
| :--- | :---: | :---: | :---: | :---: |
| `[0.0, 0.1]` | 6 | 0.0000 | 1.0000 | 1.0000 |
| `[0.9, 1.0]` | 94 | 0.9605 | 1.0000 | 0.0395 |

## 4. Confidence Threshold Sweep (`spam_threshold` × `escalation_threshold`)

| Spam Drop Threshold | Escalation Threshold | False Silent-Drop Rate | Spam Drop Recall | Auto-Draft Coverage | Auto-Draft Precision | Human Escalation Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.75` | `0.55` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.75` | `0.65` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.75` | `0.75` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.75` | `0.80` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.75` | `0.85` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.80` | `0.55` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.80` | `0.65` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.80` | `0.75` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.80` | `0.80` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.80` | `0.85` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.85` | `0.55` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.85` | `0.65` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.85` | `0.75` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.85` | `0.80` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.85` | `0.85` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.90` | `0.55` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.90` | `0.65` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.90` | `0.75` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.90` | `0.80` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.90` | `0.85` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.95` | `0.55` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.95` | `0.65` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.95` | `0.75` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.95` | `0.80` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
| `0.95` | `0.85` | **0.00%** | 100.0% | 64.0% | 100.0% | 18.0% |
