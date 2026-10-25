"""Masked binary classification metrics for one Tox21 assay endpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate assay metrics and stratified bootstrap CIs when requested."""
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels et probabilities doivent avoir la même forme (n,).")
    if not len(labels):
        raise ValueError("Impossible de calculer les métriques sans labels observés.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Les labels observés doivent être binaires {0,1}.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Les probabilités doivent être finies et dans [0,1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Le seuil de décision doit être dans [0,1].")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap doit être positif ou nul.")
    if np.unique(labels).size != 2:
        raise ValueError("ROC-AUC nécessite des labels positifs et négatifs.")

    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": _expected_calibration_error(
            labels,
            probabilities,
            bins=10,
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    result: dict[str, Any] = {
        "sample_count": int(len(labels)),
        "class_counts": {
            "negative": int(np.sum(labels == 0)),
            "positive": int(np.sum(labels == 1)),
        },
        "positive_prevalence": float(np.mean(labels)),
        "decision_threshold": float(threshold),
        "metrics": metrics,
        "confidence_intervals_95_percent": None,
    }
    if n_bootstrap:
        result["confidence_intervals_95_percent"] = _bootstrap_intervals(
            labels,
            probabilities,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return result


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> float:
    bin_indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        selected = bin_indices == bin_index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(labels[selected].mean())
                - float(probabilities[selected].mean())
            )
    return error


def _bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float] | None]:
    """Bootstrap within each class to retain both classes in each replicate."""
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    samples: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "brier_score": [],
    }
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        sample_y = labels[indices]
        sample_p = probabilities[indices]
        samples["roc_auc"].append(float(roc_auc_score(sample_y, sample_p)))
        samples["average_precision"].append(
            float(average_precision_score(sample_y, sample_p))
        )
        samples["brier_score"].append(float(brier_score_loss(sample_y, sample_p)))
    return {
        metric: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        if values
        else None
        for metric, values in samples.items()
    }
