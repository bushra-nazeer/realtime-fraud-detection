import numpy as np

from fraud.drift import DriftMonitor, ks_pvalue, psi


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.1


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    shifted = rng.normal(3, 1, 5000)
    assert psi(a, shifted) > 0.2


def test_ks_pvalue_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 3000)
    same = rng.normal(0, 1, 3000)
    shifted = rng.normal(2, 1, 3000)
    assert ks_pvalue(a, same) > 0.01
    assert ks_pvalue(a, shifted) < 0.01


def test_drift_monitor_flags_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 4000)
    monitor = DriftMonitor(reference, psi_threshold=0.2, ks_pvalue_threshold=0.01)
    assert monitor.check(rng.normal(0, 1, 2000))["drift_detected"] is False
    assert monitor.check(rng.normal(2.5, 1, 2000))["drift_detected"] is True
