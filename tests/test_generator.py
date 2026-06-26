import numpy as np

from fraud.config import load_config
from fraud.generator import generate_stream_frame, generate_transactions


def test_generate_transactions_schema_and_fraud_rate():
    cfg = load_config()
    df = generate_transactions(cfg, n=5000, seed=0)
    assert len(df) == 5000
    expected = {
        "transaction_id", "customer_id", "timestamp", "amount", "hour",
        "merchant_category", "is_foreign", "distance_from_home_km", "is_fraud",
    }
    assert expected <= set(df.columns)
    # Fraud rate roughly matches the configured rate.
    assert abs(df["is_fraud"].mean() - cfg.generator.fraud_rate) < 0.01
    assert df["timestamp"].is_monotonic_increasing


def test_fraud_has_distinct_distribution():
    cfg = load_config()
    df = generate_transactions(cfg, n=20000, seed=1)
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]
    # Fraud is more often foreign and travels further from home.
    assert fraud["is_foreign"].mean() > legit["is_foreign"].mean()
    assert fraud["distance_from_home_km"].mean() > legit["distance_from_home_km"].mean()


def test_stream_frame_has_drift_segment():
    cfg = load_config()
    stream = generate_stream_frame(cfg, seed=3)
    assert len(stream) == cfg.stream.demo_events
    cut = cfg.stream.drift_after
    before = stream.iloc[:cut]["is_fraud"].mean()
    after = stream.iloc[cut:]["is_fraud"].mean()
    # The drifted tail has a higher fraud rate than the normal segment.
    assert after > before
    assert np.all(np.diff(stream["timestamp"].to_numpy()) >= 0)
