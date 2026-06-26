"""Pydantic request/response models for the scoring API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    customer_id: int
    amount: float = Field(..., ge=0)
    hour: int = Field(default=12, ge=0, le=23)
    merchant_category: str = "online"
    is_foreign: int = Field(default=0, ge=0, le=1)
    distance_from_home_km: float = 0.0
    timestamp: float | None = None


class FactorContribution(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_flagged: bool
    risk_band: str
    anomaly_score: float | None = None
    top_factors: list[FactorContribution]
