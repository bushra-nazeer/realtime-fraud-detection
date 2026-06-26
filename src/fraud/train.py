"""Train the supervised fraud model (XGBoost, Optuna-tuned) plus a LightGBM
baseline and an IsolationForest anomaly detector; log to MLflow; persist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .anomaly import fit_anomaly
from .config import Config, load_config
from .dataset import make_dataset, split_dataset
from .evaluate import compute_metrics, pick_threshold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _scale_pos_weight(y) -> float:
    positives = int(y.sum())
    return float((len(y) - positives) / max(positives, 1))


def train(cfg: Config, n_trials: int | None = None) -> dict:
    X, y = make_dataset(cfg)
    X_train, X_test, y_train, y_test = split_dataset(X, y, cfg)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=cfg.random_state
    )
    spw = _scale_pos_weight(y_tr)
    n_trials = cfg.optuna.n_trials if n_trials is None else n_trials

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBClassifier(
            **params, scale_pos_weight=spw, random_state=cfg.random_state,
            n_jobs=-1, eval_metric="aucpr", tree_method="hist",
        )
        model.fit(X_tr, y_tr)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=cfg.optuna.timeout_seconds)
    best_params = dict(study.best_params)

    model = XGBClassifier(
        **best_params, scale_pos_weight=_scale_pos_weight(y_train),
        random_state=cfg.random_state, n_jobs=-1, eval_metric="aucpr", tree_method="hist",
    )
    model.fit(X_train, y_train)

    test_proba = model.predict_proba(X_test)[:, 1]
    threshold = pick_threshold(y_test.to_numpy(), test_proba, cfg.threshold.target_precision)
    metrics = compute_metrics(y_test.to_numpy(), test_proba, threshold)
    metrics["fraud_rate"] = float(y.mean())

    baseline = LGBMClassifier(
        n_estimators=300, scale_pos_weight=_scale_pos_weight(y_train),
        random_state=cfg.random_state, n_jobs=-1, verbose=-1,
    )
    baseline.fit(X_train, y_train)
    metrics["baseline_roc_auc"] = float(roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1]))

    anomaly_model = fit_anomaly(X_train, random_state=cfg.random_state)

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="xgboost-optuna"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    Path(cfg.paths.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.paths.model_path)
    joblib.dump(anomaly_model, cfg.paths.anomaly_path)
    metadata = {
        "best_params": best_params,
        "metrics": metrics,
        "feature_names": list(X.columns),
        "merchant_categories": list(cfg.generator.merchant_categories),
        "dataset": "synthetic transactions (fraud.generator)",
    }
    Path(cfg.paths.model_metadata).write_text(json.dumps(metadata, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud-detection model.")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    metrics = train(cfg, n_trials=args.trials)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
