"""Typed configuration loaded from ``config/config.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = "config/config.yaml"


class Paths(BaseModel):
    model_path: str
    anomaly_path: str
    model_metadata: str
    reports_dir: str
    figures_dir: str


class GeneratorCfg(BaseModel):
    n_customers: int = 2000
    n_transactions: int = 60000
    fraud_rate: float = 0.012
    merchant_categories: list[str] = Field(
        default_factory=lambda: [
            "grocery", "restaurant", "travel", "electronics",
            "fuel", "online", "atm", "healthcare",
        ]
    )


class OptunaCfg(BaseModel):
    n_trials: int = 20
    timeout_seconds: int = 600


class ThresholdCfg(BaseModel):
    target_precision: float = 0.90


class DriftCfg(BaseModel):
    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.01
    window_size: int = 2000


class StreamCfg(BaseModel):
    topic: str = "transactions"
    bootstrap_servers: str = "localhost:9092"
    demo_events: int = 5000
    drift_after: int = 3000


class Config(BaseModel):
    paths: Paths
    random_state: int = 42
    test_size: float = 0.25
    generator: GeneratorCfg = GeneratorCfg()
    optuna: OptunaCfg = OptunaCfg()
    threshold: ThresholdCfg = ThresholdCfg()
    drift: DriftCfg = DriftCfg()
    stream: StreamCfg = StreamCfg()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(**data)
