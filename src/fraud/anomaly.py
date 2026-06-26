"""Unsupervised anomaly detection (IsolationForest) as a complementary signal
to the supervised fraud model."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_anomaly(X, random_state: int, contamination: float | str = "auto") -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def anomaly_score(model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous (negated IsolationForest score)."""
    return -model.score_samples(X)
