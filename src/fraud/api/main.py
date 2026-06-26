"""FastAPI real-time fraud scoring service.

Maintains per-customer feature state in-process, so repeated calls for the same
customer build up velocity/spend context exactly as the streaming consumer does.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..anomaly import anomaly_score
from ..config import load_config
from ..explain import make_explainer, top_factors
from ..features import FeatureState
from .schemas import FactorContribution, ScoreResponse, TransactionIn

cfg = load_config()
_state: dict = {
    "model": None,
    "anomaly": None,
    "metadata": {},
    "features": None,
    "explainer": None,
    "clock": 0.0,
}


def _load() -> None:
    model_path = Path(cfg.paths.model_path)
    if not model_path.exists():
        return
    _state["model"] = joblib.load(model_path)
    metadata = json.loads(Path(cfg.paths.model_metadata).read_text())
    _state["metadata"] = metadata
    _state["features"] = FeatureState(metadata["merchant_categories"])
    if Path(cfg.paths.anomaly_path).exists():
        _state["anomaly"] = joblib.load(cfg.paths.anomaly_path)
    _state["explainer"] = make_explainer(_state["model"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)


def _risk_band(p: float) -> str:
    if p >= 0.8:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionIn) -> ScoreResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first (`make train`).")

    metadata = _state["metadata"]
    feature_names = metadata["feature_names"]
    threshold = float(metadata["metrics"]["threshold"])

    data = tx.model_dump()
    if data.get("timestamp") is None:
        _state["clock"] += 1.0
        data["timestamp"] = _state["clock"]

    row = pd.DataFrame([_state["features"].transform(data)])[feature_names]
    proba = float(model.predict_proba(row)[:, 1][0])

    anomaly = None
    if _state["anomaly"] is not None:
        anomaly = float(anomaly_score(_state["anomaly"], row)[0])

    try:
        factors = [
            FactorContribution(feature=name, contribution=value)
            for name, value in top_factors(_state["explainer"], row)
        ]
    except Exception:
        factors = []

    return ScoreResponse(
        fraud_probability=proba,
        is_flagged=proba >= threshold,
        risk_band=_risk_band(proba),
        anomaly_score=anomaly,
        top_factors=factors,
    )
