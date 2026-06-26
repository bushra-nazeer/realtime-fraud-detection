"""Build and split the training dataset (shared by train and evaluate so both
see the identical, deterministic split)."""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from .config import Config
from .features import build_feature_matrix
from .generator import generate_transactions


def make_dataset(cfg: Config):
    df = generate_transactions(cfg)
    X, y = build_feature_matrix(df, cfg.generator.merchant_categories)
    return X, y


def split_dataset(X, y, cfg: Config):
    return train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )
