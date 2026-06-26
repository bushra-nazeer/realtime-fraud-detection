"""SHAP-based per-prediction explanations (the top drivers behind a score)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_explainer(model):
    import shap  # imported here to keep module import light

    return shap.TreeExplainer(model)


def top_factors(explainer, X: pd.DataFrame, k: int = 6):
    """Return the top-k (feature, signed_shap_value) drivers for one row."""
    values = np.asarray(explainer.shap_values(X))
    if values.ndim == 3:  # (n, features, classes) in some versions
        values = values[..., -1]
    row = values[0]
    order = np.argsort(np.abs(row))[::-1][:k]
    columns = list(X.columns)
    return [(str(columns[i]), float(row[i])) for i in order]
