"""A masked Morgan-fingerprint classifier for one named Tox21 assay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from deepchem.data import Dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.tox21 import TASKS
from src.evaluation.tox21_endpoint import binary_classification_metrics

ENDPOINT_DESCRIPTIONS = {
    "NR-AR": (
        "Androgen receptor (AR) pathway reporter-assay activity. A positive "
        "label means assay activity in this Tox21 in-vitro assay; it is not a "
        "clinical androgenic-effect determination or a general toxicity label."
    )
}


def _fingerprint(molecule: Chem.Mol, generator, size: int) -> np.ndarray:
    vector = np.zeros((size,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), vector)
    return vector


def extract_endpoint_data(
    dataset: Dataset,
    endpoint: str,
    *,
    fingerprint_size: int = 2048,
    fingerprint_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return only valid, observed labels and counts; masked labels are excluded."""
    if endpoint not in TASKS:
        raise ValueError(f"Endpoint Tox21 inconnu : {endpoint}.")
    labels = np.asarray(dataset.y)
    weights = np.asarray(dataset.w)
    ids = np.asarray(dataset.ids)
    if labels.ndim != 2 or labels.shape != weights.shape:
        raise ValueError("Les labels y et les masques w doivent avoir la même forme 2D.")
    if labels.shape[1] != len(TASKS):
        raise ValueError(f"Dataset attendu avec {len(TASKS)} endpoints Tox21.")
    if len(ids) != len(labels):
        raise ValueError("Le nombre d'IDs doit correspondre au nombre de labels.")
    if not np.isin(weights, [0, 1]).all():
        raise ValueError("Les poids Tox21 doivent être 0 (masqué) ou 1 (observé).")

    task_index = TASKS.index(endpoint)
    task_labels = labels[:, task_index]
    task_weights = weights[:, task_index]
    if not np.isin(task_labels[task_weights > 0], [0, 1]).all():
        raise ValueError("Les labels observés doivent être binaires {0,1}.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprint_radius,
        fpSize=fingerprint_size,
    )
    features: list[np.ndarray] = []
    observed_labels: list[int] = []
    invalid_structures = 0
    observed_invalid_structures = 0
    observed_count = int(np.sum(task_weights > 0))
    missing_count = int(np.sum(task_weights == 0))

    for row, raw_smiles in enumerate(ids):
        molecule = _parse_and_sanitize(raw_smiles)
        if molecule is None:
            invalid_structures += 1
            if task_weights[row] > 0:
                observed_invalid_structures += 1
            continue
        if task_weights[row] == 0:
            continue
        features.append(_fingerprint(molecule, generator, fingerprint_size))
        observed_labels.append(int(task_labels[row]))

    x = (
        np.asarray(features, dtype=np.uint8)
        if features
        else np.empty((0, fingerprint_size), dtype=np.uint8)
    )
    y = np.asarray(observed_labels, dtype=np.int8)
    counts = {
        "compound_count": int(len(ids)),
        "observed_label_count": observed_count,
        "masked_label_count": missing_count,
        "invalid_structure_count": invalid_structures,
        "observed_invalid_structure_count": observed_invalid_structures,
        "usable_label_count": int(len(y)),
        "negative_count": int(np.sum(y == 0)),
        "positive_count": int(np.sum(y == 1)),
    }
    return x, y, counts


def _parse_and_sanitize(value: Any) -> Chem.Mol | None:
    if isinstance(value, Chem.Mol):
        molecule = Chem.Mol(value)
    elif isinstance(value, str) and value.strip():
        molecule = Chem.MolFromSmiles(value, sanitize=False)
    else:
        return None
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    return molecule


@dataclass
class Tox21EndpointPredictor:
    """Random forest predicting activity in a single Tox21 assay endpoint."""

    endpoint: str
    estimator: RandomForestClassifier
    fingerprint_size: int
    fingerprint_radius: int
    decision_threshold: float
    random_seed: int
    selected_estimators: int
    training_counts: dict[str, int]
    validation_counts: dict[str, int]
    validation_metrics: dict[str, Any]
    candidate_validation_auc: dict[str, float]
    outcome_label: str = "predicted Tox21 assay activity"
    fingerprint_generator: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.endpoint not in ENDPOINT_DESCRIPTIONS:
            raise ValueError(
                f"Endpoint non pris en charge par ce prédicteur : {self.endpoint}."
            )
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("Le seuil de décision doit être dans [0,1].")
        self.fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fingerprint_radius,
            fpSize=self.fingerprint_size,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("fingerprint_generator", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__post_init__()

    @classmethod
    def fit(
        cls,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        *,
        endpoint: str = "NR-AR",
        estimator_candidates: Sequence[int] = (100, 300),
        fingerprint_size: int = 2048,
        fingerprint_radius: int = 2,
        seed: int = 42,
    ) -> "Tox21EndpointPredictor":
        """Fit on observed training rows and select only by validation ROC-AUC.

        No test dataset is accepted by this API. Missing assay labels are masked
        with ``Dataset.w == 0`` and never used as negative examples.
        """
        if endpoint not in ENDPOINT_DESCRIPTIONS:
            raise ValueError(
                f"Endpoint non pris en charge. Disponibles : "
                f"{', '.join(ENDPOINT_DESCRIPTIONS)}."
            )
        candidates = tuple(dict.fromkeys(int(value) for value in estimator_candidates))
        if not candidates or any(value < 1 for value in candidates):
            raise ValueError("estimator_candidates doit contenir des entiers positifs.")

        x_train, y_train, train_counts = extract_endpoint_data(
            train_dataset,
            endpoint,
            fingerprint_size=fingerprint_size,
            fingerprint_radius=fingerprint_radius,
        )
        x_valid, y_valid, validation_counts = extract_endpoint_data(
            validation_dataset,
            endpoint,
            fingerprint_size=fingerprint_size,
            fingerprint_radius=fingerprint_radius,
        )
        if np.unique(y_train).size != 2:
            raise ValueError(
                f"Train {endpoint}: les deux classes doivent être présentes."
            )
        if np.unique(y_valid).size != 2:
            raise ValueError(
                f"Validation {endpoint}: les deux classes observées sont requises "
                "pour sélectionner le modèle."
            )

        candidate_auc: dict[str, float] = {}
        best_estimator: RandomForestClassifier | None = None
        best_probabilities: np.ndarray | None = None
        best_auc = -np.inf
        for n_estimators in candidates:
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
            estimator.fit(x_train, y_train)
            probabilities = _positive_probabilities(estimator, x_valid)
            auc = float(roc_auc_score(y_valid, probabilities))
            candidate_auc[str(n_estimators)] = auc
            if auc > best_auc:
                best_auc = auc
                best_estimator = estimator
                best_probabilities = probabilities

        if best_estimator is None or best_probabilities is None:
            raise RuntimeError("Aucun modèle n'a été retenu par la validation.")
        threshold = _select_youden_threshold(y_valid, best_probabilities)
        validation_metrics = binary_classification_metrics(
            y_valid,
            best_probabilities,
            threshold,
        )
        return cls(
            endpoint=endpoint,
            estimator=best_estimator,
            fingerprint_size=fingerprint_size,
            fingerprint_radius=fingerprint_radius,
            decision_threshold=threshold,
            random_seed=seed,
            selected_estimators=best_estimator.n_estimators,
            training_counts=train_counts,
            validation_counts=validation_counts,
            validation_metrics=validation_metrics,
            candidate_validation_auc=candidate_auc,
        )

    @property
    def biological_interpretation(self) -> str:
        return ENDPOINT_DESCRIPTIONS[self.endpoint]

    def predict_active_probability(
        self,
        molecules: Sequence[Chem.Mol],
    ) -> np.ndarray:
        """Predict positive-assay probability for sanitized RDKit molecules."""
        probabilities, _ = self.predict_activity_with_uncertainty(molecules)
        return probabilities

    def predict_activity_with_uncertainty(
        self,
        molecules: Sequence[Chem.Mol],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return assay probabilities and normalized tree-disagreement estimates.

        The uncertainty is the standard deviation of individual forest-tree
        probabilities, scaled to [0, 1]. It is a model-disagreement proxy, not
        a calibrated confidence interval.
        """
        molecules = list(molecules)
        if not molecules:
            empty = np.empty((0,), dtype=float)
            return empty, empty.copy()
        vectors = []
        for index, molecule in enumerate(molecules):
            sanitized = _parse_and_sanitize(molecule)
            if sanitized is None:
                raise ValueError(
                    f"Molécule RDKit invalide ou non sanitisable à l'index {index}."
                )
            vectors.append(
                _fingerprint(
                    sanitized,
                    self.fingerprint_generator,
                    self.fingerprint_size,
                )
            )
        features = np.asarray(vectors)
        probabilities = _positive_probabilities(self.estimator, features)
        tree_probabilities = np.asarray(
            [
                _positive_probabilities(tree, features)
                for tree in self.estimator.estimators_
            ],
            dtype=float,
        )
        uncertainty = np.clip(np.std(tree_probabilities, axis=0) * 2.0, 0.0, 1.0)
        return probabilities, uncertainty

    def reference_fingerprints(
        self,
        training_dataset: Dataset,
    ) -> tuple[list[Any], int]:
        """Fingerprint observed-label training structures for support checks."""
        labels = np.asarray(training_dataset.y)
        weights = np.asarray(training_dataset.w)
        ids = np.asarray(training_dataset.ids)
        if labels.shape != weights.shape or labels.ndim != 2:
            raise ValueError("Les labels y et masques w doivent avoir la même forme 2D.")
        if len(ids) != len(labels) or labels.shape[1] != len(TASKS):
            raise ValueError("Le dataset de référence n'a pas la forme Tox21 attendue.")
        endpoint_index = TASKS.index(self.endpoint)
        fingerprints = []
        invalid_count = 0
        for row, raw_smiles in enumerate(ids):
            if weights[row, endpoint_index] == 0:
                continue
            molecule = _parse_and_sanitize(raw_smiles)
            if molecule is None:
                invalid_count += 1
                continue
            fingerprints.append(
                self.fingerprint_generator.GetFingerprint(molecule)
            )
        if not fingerprints:
            raise ValueError(
                f"Aucune structure de référence valide avec label observé "
                f"pour {self.endpoint}."
            )
        return fingerprints, invalid_count

    def predict(self, molecules: Sequence[Chem.Mol]) -> list[dict[str, Any]]:
        """Return probabilities and assay-activity labels, never a safety claim."""
        probabilities = self.predict_active_probability(molecules)
        return [
            {
                "endpoint": self.endpoint,
                "outcome_label": self.outcome_label,
                "predicted_probability_active": float(probability),
                "predicted_assay_active": bool(
                    probability >= self.decision_threshold
                ),
                "decision_threshold": self.decision_threshold,
                "interpretation": self.biological_interpretation,
            }
            for probability in probabilities
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "Tox21EndpointPredictor":
        predictor = joblib.load(path)
        if not isinstance(predictor, cls):
            raise TypeError(f"Fichier non reconnu comme prédicteur Tox21 : {path}")
        return predictor


def _positive_probabilities(
    estimator: RandomForestClassifier,
    features: np.ndarray,
) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("Le classifieur n'a pas exactement une classe positive 1.")
    return np.asarray(
        estimator.predict_proba(features)[:, int(positive_columns[0])],
        dtype=float,
    )


def _select_youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        probabilities,
    )
    finite = np.isfinite(thresholds)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    if not len(scores):
        raise ValueError("Impossible de choisir un seuil de décision sur validation.")
    return float(thresholds[finite][int(np.argmax(scores))])
