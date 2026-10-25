"""Held-out evaluation helpers for a single masked Tox21 endpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
from deepchem.data import Dataset

from src.data.tox21 import TASKS
from src.evaluation.tox21_endpoint import binary_classification_metrics
from src.models.tox21_endpoint import (
    Tox21EndpointPredictor,
    _parse_and_sanitize,
)


def evaluate_tox21_endpoint(
    predictor: Tox21EndpointPredictor,
    dataset: Dataset,
    *,
    split_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate only observed, valid structures in the caller-selected split."""
    if predictor.endpoint not in TASKS:
        raise ValueError(f"Endpoint Tox21 inconnu : {predictor.endpoint}.")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap doit être positif ou nul.")
    labels = np.asarray(dataset.y)
    weights = np.asarray(dataset.w)
    ids = np.asarray(dataset.ids)
    if labels.ndim != 2 or labels.shape != weights.shape:
        raise ValueError("Les labels y et les masques w doivent avoir la même forme 2D.")
    if labels.shape[1] != len(TASKS) or len(ids) != len(labels):
        raise ValueError("Formes Dataset incompatibles avec les tâches Tox21.")
    if not np.isin(weights, [0, 1]).all():
        raise ValueError("Les poids Tox21 doivent être 0 (masqué) ou 1 (observé).")

    endpoint_index = TASKS.index(predictor.endpoint)
    task_labels = labels[:, endpoint_index]
    task_weights = weights[:, endpoint_index]
    observed = task_weights > 0
    if not np.isin(task_labels[observed], [0, 1]).all():
        raise ValueError("Les labels Tox21 observés doivent être binaires {0,1}.")

    molecules = []
    evaluated_labels: list[int] = []
    prediction_ids: list[str] = []
    invalid_structure_count = 0
    observed_invalid_structure_count = 0
    for row, identifier in enumerate(ids):
        molecule = _parse_and_sanitize(identifier)
        if molecule is None:
            invalid_structure_count += 1
            if observed[row]:
                observed_invalid_structure_count += 1
            continue
        if not observed[row]:
            continue
        molecules.append(molecule)
        prediction_ids.append(str(identifier))
        evaluated_labels.append(int(task_labels[row]))

    probabilities = predictor.predict_active_probability(molecules)
    y = np.asarray(evaluated_labels, dtype=np.int8)
    metrics = binary_classification_metrics(
        y,
        probabilities,
        predictor.decision_threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    report = {
        "split": split_name,
        "endpoint": predictor.endpoint,
        "outcome_label": "predicted Tox21 assay activity",
        "biological_interpretation": predictor.biological_interpretation,
        "compound_count": int(len(ids)),
        "observed_label_count": int(np.sum(observed)),
        "masked_label_count": int(np.sum(~observed)),
        "invalid_structure_count": invalid_structure_count,
        "observed_invalid_structure_count": observed_invalid_structure_count,
        "excluded_observed_invalid_count": observed_invalid_structure_count,
        **metrics,
        "interpretation_caveat": (
            "Predictions estimate activity in the named in-vitro assay and do "
            "not establish toxicity or safety in organisms or humans."
        ),
    }
    prediction_rows = [
        {
            "smiles": identifier,
            "endpoint": predictor.endpoint,
            "observed_assay_label": int(label),
            "predicted_probability_active": float(probability),
            "predicted_assay_active": bool(
                probability >= predictor.decision_threshold
            ),
            "decision_threshold": predictor.decision_threshold,
        }
        for identifier, label, probability in zip(
            prediction_ids,
            evaluated_labels,
            probabilities,
            strict=True,
        )
    ]
    return report, prediction_rows
