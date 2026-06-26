# realtime-fraud-detection, Design Spec

- **Date:** 2026-06-25
- **Status:** Approved

## Overview

A real-time fraud-detection system: a synthetic transaction stream is scored by
a gradient-boosted model plus an unsupervised anomaly detector, with online
drift monitoring and a FastAPI scoring service. Self-contained and reproducible
- no external data or broker required to run the demo.

## What it demonstrates

| Capability | How |
|---|---|
| "$4.2M real-time fraud detection system" | End-to-end streaming detection demo |
| Kafka streaming / real-time intelligence | `MessageSource` with Kafka + in-memory backends |
| Anomaly detection | IsolationForest complementary signal |
| XGBoost/LightGBM, "33% detection / 28% fewer FP" | Tuned XGBoost + LightGBM baseline; detection-rate & FPR metrics |
| Model monitoring / drift detection | PSI + KS `DriftMonitor` over the live stream |

## Data

Synthetic generator (`generator.py`): legitimate transactions plus injected
fraud signatures, amount spikes / card-testing micro-charges, night-skewed
timing, more foreign + higher distance, riskier merchant categories. Drift knobs
(`amount_scale`, `foreign_boost`, `fraud_rate`) let the demo stream shift
distribution partway through to exercise the monitor.

## Key design decisions

- **Shared `FeatureState`** for train, stream, and API, eliminates train/serve skew.
- **Pluggable transport**: Kafka for production, in-memory for demo/tests, so
  everything verifies without a broker download.
- **Custom PSI + KS drift** instead of a heavyweight dependency, lighter and
  demonstrates the underlying statistics.
- **Time-ordered feature build**: velocity/recency computed by replaying
  transactions in timestamp order; no future leakage.

## Components

`generator` → `dataset` (build + split) → `features.FeatureState` → `train`
(XGBoost + Optuna + LightGBM baseline + IsolationForest + MLflow) →
`evaluate` (PR-AUC, detection rate, FPR, plots) ; `stream` (source + scorer +
drift) ; `api` (`/score`, `/health`).

## Testing

Unit tests for the generator (schema, fraud separability, drift segment),
feature state (velocity window, spend ratio, leak-free), drift (PSI/KS),
training smoke (tiny run), streaming (FIFO + scorer), and the API
(health / validation / score-or-503).

## Deliverable

Verified (tests + a real training run + a streaming demo with drift detection)
→ `realtime-fraud-detection.zip`, including the trained model so it runs out of
the box. Docker (`python:3.12-slim`) is the canonical runtime; CI runs ruff + pytest.
