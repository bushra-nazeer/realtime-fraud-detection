"""Stateful feature engineering.

``FeatureState`` maintains per-customer running statistics (spend mean, recent
timestamps for velocity, last-seen time). The *same* object computes features
for a batch (replayed in time order to build the training matrix) and for the
live stream, guaranteeing zero train/serve skew.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "log_amount",
    "hour",
    "is_night",
    "is_foreign",
    "distance_from_home_km",
    "amount_to_cust_mean",
    "tx_count_1h",
    "secs_since_last",
]
_VELOCITY_WINDOW_SECONDS = 3600
_NO_PRIOR_TX = 1_000_000.0


class FeatureState:
    """Incremental feature builder with per-customer memory."""

    def __init__(self, merchant_categories: list[str]):
        self.categories = list(merchant_categories)
        self._count: dict[int, int] = defaultdict(int)
        self._sum_amount: dict[int, float] = defaultdict(float)
        self._recent: dict[int, deque] = defaultdict(deque)
        self._last_ts: dict[int, float] = {}

    def feature_names(self) -> list[str]:
        return NUMERIC_FEATURES + [f"merchant_{c}" for c in self.categories]

    def transform(self, tx: dict, update: bool = True) -> dict:
        cid = int(tx["customer_id"])
        amount = float(tx["amount"])
        ts = float(tx["timestamp"])
        hour = int(tx["hour"])

        prior_count = self._count[cid]
        prior_mean = (self._sum_amount[cid] / prior_count) if prior_count else amount
        amount_to_mean = amount / prior_mean if prior_mean > 0 else 1.0

        recent = self._recent[cid]
        while recent and ts - recent[0] > _VELOCITY_WINDOW_SECONDS:
            recent.popleft()
        tx_count_1h = len(recent)

        last = self._last_ts.get(cid)
        secs_since_last = (ts - last) if last is not None else _NO_PRIOR_TX

        features = {
            "log_amount": float(np.log1p(amount)),
            "hour": float(hour),
            "is_night": float(hour < 6 or hour >= 22),
            "is_foreign": float(tx["is_foreign"]),
            "distance_from_home_km": float(tx["distance_from_home_km"]),
            "amount_to_cust_mean": float(amount_to_mean),
            "tx_count_1h": float(tx_count_1h),
            "secs_since_last": float(min(secs_since_last, _NO_PRIOR_TX)),
        }
        for category in self.categories:
            features[f"merchant_{category}"] = 1.0 if tx["merchant_category"] == category else 0.0

        if update:
            self._count[cid] += 1
            self._sum_amount[cid] += amount
            recent.append(ts)
            self._last_ts[cid] = ts

        return features


def build_feature_matrix(
    df: pd.DataFrame, merchant_categories: list[str]
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Replay transactions in time order to produce the feature matrix + label."""
    state = FeatureState(merchant_categories)
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    rows = [state.transform(record) for record in ordered.to_dict("records")]
    X = pd.DataFrame(rows, columns=state.feature_names())
    y = ordered["is_fraud"].astype(int) if "is_fraud" in ordered.columns else None
    return X, y
