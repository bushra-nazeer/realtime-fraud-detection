"""Drift detection via Population Stability Index (PSI) and the
Kolmogorov-Smirnov two-sample test."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

_EPS = 1e-6


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between a reference and current sample.

    Rule of thumb: <0.1 no shift, 0.1-0.2 moderate, >0.2 significant.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)
    ref_prop = np.clip(ref_counts / max(ref_counts.sum(), 1), _EPS, None)
    cur_prop = np.clip(cur_counts / max(cur_counts.sum(), 1), _EPS, None)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def ks_pvalue(reference, current) -> float:
    return float(ks_2samp(np.asarray(reference), np.asarray(current)).pvalue)


class DriftMonitor:
    """Compares a live window against a fixed reference distribution."""

    def __init__(self, reference, psi_threshold: float = 0.2, ks_pvalue_threshold: float = 0.01):
        self.reference = np.asarray(reference, dtype=float)
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold

    def check(self, window) -> dict:
        window = np.asarray(window, dtype=float)
        psi_value = psi(self.reference, window)
        ks_p = ks_pvalue(self.reference, window)
        return {
            "psi": psi_value,
            "ks_pvalue": ks_p,
            "drift_detected": bool(
                psi_value > self.psi_threshold or ks_p < self.ks_pvalue_threshold
            ),
        }
