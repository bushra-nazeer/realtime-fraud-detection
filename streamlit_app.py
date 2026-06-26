"""Streamlit demo for the real-time fraud detector.

Run locally with `streamlit run streamlit_app.py`, or deploy free on Streamlit
Community Cloud by pointing it at this file.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from fraud.config import load_config
from fraud.explain import make_explainer, top_factors
from fraud.features import FeatureState

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")


@st.cache_resource
def _load():
    cfg = load_config()
    model = joblib.load(cfg.paths.model_path)
    with open(cfg.paths.model_metadata) as fh:
        meta = json.load(fh)
    return cfg, model, meta, make_explainer(model)


cfg, model, meta, explainer = _load()
threshold = float(meta["metrics"]["threshold"])

st.title("Real-Time Fraud Detection")
st.caption("Scores a card transaction for fraud risk. Synthetic data; educational demo.")

left, right = st.columns(2)
amount = left.number_input("Amount ($)", min_value=0.0, max_value=10000.0, value=1299.0, step=10.0)
hour = right.slider("Hour of day", 0, 23, 3)
merchant = left.selectbox("Merchant category", meta["merchant_categories"])
distance = right.number_input("Distance from home (km)", min_value=0.0, max_value=3000.0, value=740.0)
is_foreign = 1 if left.checkbox("Foreign transaction", value=True) else 0

if st.button("Score transaction", type="primary"):
    state = FeatureState(meta["merchant_categories"])
    tx = {
        "customer_id": 0,
        "amount": amount,
        "hour": hour,
        "merchant_category": merchant,
        "is_foreign": is_foreign,
        "distance_from_home_km": distance,
        "timestamp": 1.0,
    }
    row = pd.DataFrame([state.transform(tx)])[meta["feature_names"]]
    proba = float(model.predict_proba(row)[:, 1][0])

    flagged = proba >= threshold
    st.metric("Fraud probability", f"{proba * 100:.1f}%", "FLAGGED" if flagged else "cleared")
    st.progress(min(proba, 1.0))

    st.subheader("Top contributing factors")
    for feature, value in top_factors(explainer, row, k=6):
        arrow = "raises risk" if value > 0 else "lowers risk"
        st.write(f"- `{feature}`: {arrow}  ({value:+.3f})")
