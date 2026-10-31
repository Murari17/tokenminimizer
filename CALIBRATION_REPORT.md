# Local SLM Classification, Held-Out Slice & Post-Hoc Calibration Report

- **Evaluation Mode:** `replayed_slm_predictions`
- **Shipped & Evaluated Model (`DEFAULT_CONFIG`):** `gemma4:e4b-it-qat`
- **SLM Directly Measured:** `True`
- **Auditable Per-Ticket Raw Outputs:** [`tests/eval_predictions.jsonl`](tests/eval_predictions.jsonl) (reproducible offline via `--from-predictions tests/eval_predictions.jsonl`)
- **Dataset Size:** `300` tickets (`100` calibration split + `200` held-out test split)
- **Adversarial Injections Short-Circuited at Stage 2 (0 Tokens):** `12`
- **Overall Intent Accuracy (N=300):** `99.67%` (95% Wilson CI: `[98.14%, 99.94%]`)
- **Held-Out Split Intent Accuracy (N=200):** `99.50%` (95% Wilson CI: `[97.22%, 99.91%]`)
- **Held-Out End-to-End Routing Queue Accuracy:** `99.50%`
- **False Silent-Drop Rate (Legitimate Ticket -> Trash):** `0.00%`
- **Alias Drift Telemetry (Non-Canonical Label Remaps):** `0` (`{}`)
- **Latency (p50 / p95 vs 30,000 ms Timeout):** `14052.14 ms` / `18062.69 ms`
- **Total Tokens Used:** `110074` (`96886` prompt + `13188` completion)

## 1. Held-Out Challenge Slice Breakdown (Wilson 95% CIs)

| Challenge Slice | Tickets | Correct | Empirical Accuracy | 95% Wilson Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| `clean_short` | 100 | 100 | **100.00%** | `[96.30%, 100.00%]` |
| `duplicate_generic_subject` | 30 | 30 | **100.00%** | `[88.65%, 100.00%]` |
| `hard_negative` | 45 | 45 | **100.00%** | `[92.13%, 100.00%]` |
| `long_quoted_thread` | 45 | 45 | **100.00%** | `[92.13%, 100.00%]` |
| `mixed_and_adversarial` | 40 | 39 | **97.50%** | `[87.12%, 99.56%]` |
| `multilingual` | 40 | 40 | **100.00%** | `[91.24%, 100.00%]` |

## 2. Post-Hoc Confidence Calibration on Held-Out Split (`N=200`)

Fitted on the `100`-item calibration split (with unscripted prompt probabilities) and evaluated on the disjoint `200`-item held-out split:

| Calibration Method | Parameter / Complexity | Held-Out 10-Bin ECE | Held-Out Brier Score |
| :--- | :--- | :---: | :---: |
| **Uncalibrated Raw SLM Confidence** | None ($T = 1.0$) | `0.0164` | `0.0052` |
| **Temperature Scaling** | $T^* = 0.74$ (1 scalar parameter) | **`0.001`** | **`0.005`** |
| **Isotonic Regression (PAVA)** | `96` monotonic step blocks | **`0.0052`** | **`0.0052`** |

## 3. Per-Class Precision, Recall & F1 (`N=300`)

| Intent Category | Precision | Recall | F1 Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `BILLING` | 0.9855 | 1.0000 | **0.9927** | 68 |
| `TECHNICAL_SUPPORT` | 1.0000 | 1.0000 | **1.0000** | 67 |
| `SECURITY_REPORT` | 1.0000 | 1.0000 | **1.0000** | 51 |
| `FEATURE_REQUEST` | 1.0000 | 1.0000 | **1.0000** | 59 |
| `SPAM` | 1.0000 | 0.9818 | **0.9908** | 55 |

## 4. Confusion Matrix (Rows = Expected, Columns = Predicted)

| Expected \ Predicted | `BILLING` | `TECHNICAL_SUPPORT` | `FEATURE_REQUEST` | `SPAM` | `SECURITY_REPORT` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`BILLING`** | 68 | 0 | 0 | 0 | 0 |
| **`TECHNICAL_SUPPORT`** | 0 | 67 | 0 | 0 | 0 |
| **`FEATURE_REQUEST`** | 0 | 0 | 59 | 0 | 0 |
| **`SPAM`** | 1 | 0 | 0 | 54 | 0 |
| **`SECURITY_REPORT`** | 0 | 0 | 0 | 0 | 51 |

## 5. Discrete Verbalized Confidence Breakdown (`N=288` SLM Predictions)

Without prompt-scripted confidence bands, `gemma4:e4b-it-qat` (`temperature=0.0`) emits three discrete verbalized confidence levels (`0.95`, `0.98`, `0.99`). Notably, the single misclassified ticket (`eval-267`) occurred in the lowest verbalized level (`0.95`, empirical accuracy `96.43%` vs `95.00%` confidence, gap `1.43%`), whereas `0.98` and `0.99` achieved `100.00%` (`260/260`):

| Verbalized Confidence | Tickets | Correct | Empirical Accuracy | 95% Wilson CI | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.95` | 28 | 27 | **96.43%** | `[82.29%, 99.37%]` | `0.0143` |
| `0.98` | 195 | 195 | **100.00%** | `[98.07%, 100.00%]` | `0.0200` |
| `0.99` | 65 | 65 | **100.00%** | `[94.42%, 100.00%]` | `0.0100` |

### Standard 10-Bin Reliability Diagram

| Confidence Bin | Samples | Mean Self-Reported Confidence | Empirical Accuracy | Calibration Gap |
| :--- | :---: | :---: | :---: | :---: |
| `[0.9, 1.0]` | 288 | 0.9793 | 0.9965 | 0.0172 |

## 6. Held-Out Threshold Sweep (`spam_threshold` × `escalation_threshold`)

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

## 7. Error Analysis (Misclassified Tickets)

| Ticket ID | Split | Slice | Subject | Expected | Predicted | Confidence | Routed Queue |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `eval-267` | `held_out` | `mixed_and_adversarial` | Sponsored newsletter placement in your weekly developer digest | `SPAM` | `BILLING` | 0.95 | `billing_support` |
