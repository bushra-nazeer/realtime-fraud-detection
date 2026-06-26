"""Real-time scoring over a message stream, with online drift detection.

``StreamScorer`` is the shared scoring core (also used by the API). ``run_demo``
wires the synthetic generator through an in-memory source and reports detection
stats plus any drift the monitor catches when the stream distribution shifts.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import joblib
import pandas as pd

from ..anomaly import anomaly_score
from ..config import Config, load_config
from ..drift import DriftMonitor
from ..features import FeatureState
from ..generator import generate_stream_frame
from .source import InMemorySource


class StreamScorer:
    """Scores one transaction at a time, maintaining per-customer feature state."""

    def __init__(self, cfg: Config, model=None, anomaly_model=None, metadata: dict | None = None):
        self.cfg = cfg
        self.model = model if model is not None else joblib.load(cfg.paths.model_path)
        if metadata is None:
            metadata = json.loads(Path(cfg.paths.model_metadata).read_text())
        self.metadata = metadata
        self.feature_names = metadata["feature_names"]
        self.threshold = float(metadata["metrics"]["threshold"])
        self.state = FeatureState(metadata["merchant_categories"])
        self.anomaly_model = anomaly_model
        if self.anomaly_model is None and model is None and Path(cfg.paths.anomaly_path).exists():
            self.anomaly_model = joblib.load(cfg.paths.anomaly_path)

    def score(self, tx: dict) -> dict:
        features = self.state.transform(tx)
        row = pd.DataFrame([features])[self.feature_names]
        proba = float(self.model.predict_proba(row)[:, 1][0])
        result = {
            "transaction_id": tx.get("transaction_id"),
            "fraud_probability": proba,
            "is_flagged": bool(proba >= self.threshold),
        }
        if self.anomaly_model is not None:
            result["anomaly_score"] = float(anomaly_score(self.anomaly_model, row)[0])
        return result


def run_demo(cfg: Config | None = None) -> dict:
    """In-memory streaming simulation: score the drifting demo stream and
    report detection stats + drift events. Requires a trained model on disk."""
    cfg = cfg or load_config()
    scorer = StreamScorer(cfg)

    source = InMemorySource()
    for record in generate_stream_frame(cfg).to_dict("records"):
        source.produce(record)

    cut = cfg.stream.drift_after
    window: deque[float] = deque(maxlen=cfg.drift.window_size)
    reference_scores: list[float] = []
    monitor: DriftMonitor | None = None
    drift_events: list[dict] = []
    total = flagged = true_positives = false_positives = actual_fraud = 0

    for record in source.consume():
        result = scorer.score(record)
        total += 1
        window.append(result["fraud_probability"])

        if total <= cut:
            reference_scores.append(result["fraud_probability"])
        elif monitor is None:
            monitor = DriftMonitor(
                reference_scores, cfg.drift.psi_threshold, cfg.drift.ks_pvalue_threshold
            )

        if record.get("is_fraud") == 1:
            actual_fraud += 1
        if result["is_flagged"]:
            flagged += 1
            if record.get("is_fraud") == 1:
                true_positives += 1
            else:
                false_positives += 1

        if monitor is not None and total % cfg.drift.window_size == 0:
            check = monitor.check(list(window))
            if check["drift_detected"]:
                drift_events.append({"at_event": total, **check})

    if monitor is not None and len(window) >= 50:
        final = monitor.check(list(window))
        if final["drift_detected"] and (not drift_events or drift_events[-1]["at_event"] != total):
            drift_events.append({"at_event": total, **final})

    summary = {
        "events": total,
        "flagged": flagged,
        "actual_fraud": actual_fraud,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "detection_rate": (true_positives / actual_fraud) if actual_fraud else 0.0,
        "drift_events_detected": len(drift_events),
        "drift_events": drift_events,
    }
    return summary


def main() -> None:
    summary = run_demo()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
