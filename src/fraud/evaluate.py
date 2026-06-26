"""Evaluation: fraud-relevant metrics (detection rate, false-positive rate,
PR-AUC), threshold selection, and diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Config, load_config
from .dataset import make_dataset, split_dataset

# Non-interactive backend so figures render in headless containers/CI.
matplotlib.use("Agg")


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "detection_rate": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pick_threshold(y_true, y_proba, target_precision: float = 0.9) -> float:
    """Smallest threshold meeting ``target_precision`` with the best recall; else 0.5."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    best_threshold, best_recall = 0.5, -1.0
    for p, r, t in zip(prec[:-1], rec[:-1], thr, strict=False):
        if p >= target_precision and r > best_recall:
            best_recall, best_threshold = r, float(t)
    return best_threshold


def evaluate(cfg: Config) -> dict:
    model = joblib.load(cfg.paths.model_path)
    metadata_path = Path(cfg.paths.model_metadata)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    feature_names = metadata.get("feature_names")

    X, y = make_dataset(cfg)
    _, X_test, _, y_test = split_dataset(X, y, cfg)
    if feature_names:
        X_test = X_test[feature_names]
    proba = model.predict_proba(X_test)[:, 1]

    threshold = metadata.get("metrics", {}).get("threshold")
    if threshold is None:
        threshold = pick_threshold(y_test.to_numpy(), proba, cfg.threshold.target_precision)
    metrics = compute_metrics(y_test.to_numpy(), proba, threshold)

    reports = Path(cfg.paths.reports_dir)
    figures = Path(cfg.paths.figures_dir)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#0E7C66")
    plt.xlabel("Recall (detection rate)")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (PR-AUC = {metrics['pr_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(figures / "precision_recall.png", dpi=120)
    plt.close()

    # Score distribution by class
    plt.figure(figsize=(7, 4))
    plt.hist(proba[y_test == 0], bins=40, alpha=0.6, label="legitimate", color="#46688E")
    plt.hist(proba[y_test == 1], bins=40, alpha=0.7, label="fraud", color="#C0392B")
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.title("Score distribution by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "score_distribution.png", dpi=120)
    plt.close()

    return metrics


def main() -> None:
    cfg = load_config()
    metrics = evaluate(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
