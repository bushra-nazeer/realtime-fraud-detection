from pathlib import Path

from fraud.config import load_config
from fraud.train import train


def test_train_smoke(tmp_path):
    cfg = load_config()
    cfg.generator.n_transactions = 4000
    cfg.optuna.n_trials = 1
    cfg.paths.model_path = str(tmp_path / "model.pkl")
    cfg.paths.anomaly_path = str(tmp_path / "anomaly.pkl")
    cfg.paths.model_metadata = str(tmp_path / "meta.json")

    metrics = train(cfg)

    assert Path(cfg.paths.model_path).exists()
    assert Path(cfg.paths.anomaly_path).exists()
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "detection_rate" in metrics and "false_positive_rate" in metrics
