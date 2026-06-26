import pandas as pd

from fraud.features import FeatureState, build_feature_matrix


def _tx(cid, ts, amount, hour=12, foreign=0, dist=5.0, cat="grocery"):
    return {
        "customer_id": cid, "timestamp": ts, "amount": amount, "hour": hour,
        "is_foreign": foreign, "distance_from_home_km": dist, "merchant_category": cat,
    }


def test_feature_state_velocity_and_ratio():
    state = FeatureState(["grocery", "online"])
    f1 = state.transform(_tx(1, 0.0, 100.0))
    # First transaction: no prior history.
    assert f1["amount_to_cust_mean"] == 1.0
    assert f1["tx_count_1h"] == 0.0
    assert f1["secs_since_last"] == 1_000_000.0
    assert f1["merchant_grocery"] == 1.0 and f1["merchant_online"] == 0.0

    f2 = state.transform(_tx(1, 100.0, 300.0))
    # Second transaction 100s later: one prior tx in the last hour, mean was 100.
    assert f2["tx_count_1h"] == 1.0
    assert f2["secs_since_last"] == 100.0
    assert abs(f2["amount_to_cust_mean"] - 3.0) < 1e-6


def test_velocity_window_expires_old_events():
    state = FeatureState(["grocery"])
    state.transform(_tx(1, 0.0, 10.0))
    # 2 hours later, the earlier event has aged out of the 1h window.
    f = state.transform(_tx(1, 7200.0, 10.0))
    assert f["tx_count_1h"] == 0.0


def test_build_feature_matrix_columns_and_labels():
    df = pd.DataFrame(
        [
            {**_tx(1, 0.0, 50.0), "is_fraud": 0},
            {**_tx(2, 1.0, 9000.0, hour=3, foreign=1, dist=900.0, cat="online"), "is_fraud": 1},
        ]
    )
    X, y = build_feature_matrix(df, ["grocery", "online"])
    assert list(X.columns) == FeatureState(["grocery", "online"]).feature_names()
    assert len(X) == 2 and list(y) == [0, 1]
