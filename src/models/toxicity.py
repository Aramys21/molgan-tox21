"""Morgan-fingerprint toxicity models trained on the Tox21 assay labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data.tox21 import TASKS


@dataclass
class Tox21ToxicityPredictor:
    """Per-task classifiers with an aggregate probability of any Tox21 activity."""

    models: dict
    validation_aucs: dict[str, float]
    fingerprint_size: int = 2048
    fingerprint_radius: int = 2
    training_compounds: int = 0

    def __post_init__(self):
        self.fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fingerprint_radius,
            fpSize=self.fingerprint_size,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("fingerprint_generator", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()

    @staticmethod
    def _fingerprints(
        smiles: list[str],
        fingerprint_generator,
        fingerprint_size: int,
    ) -> tuple[list[int], np.ndarray]:
        valid_indices = []
        vectors = []
        for index, smile in enumerate(smiles):
            molecule = Chem.MolFromSmiles(smile)
            if molecule is None:
                continue
            vector = np.zeros((fingerprint_size,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(
                fingerprint_generator.GetFingerprint(molecule),
                vector,
            )
            valid_indices.append(index)
            vectors.append(vector)
        return valid_indices, np.asarray(vectors, dtype=np.uint8)

    @classmethod
    def fit(
        cls,
        train_dataset,
        validation_dataset=None,
        fingerprint_size: int = 2048,
        fingerprint_radius: int = 2,
        estimators: int = 100,
        seed: int = 42,
    ) -> "Tox21ToxicityPredictor":
        """Fit using only the official training partition.

        Validation labels are used only for ROC-AUC reporting. This method does
        not accept the test partition, preventing test-label leakage into model
        fitting or model selection.
        """
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fingerprint_radius,
            fpSize=fingerprint_size,
        )
        smiles = [str(smile) for smile in train_dataset.ids]
        valid_indices, x = cls._fingerprints(smiles, generator, fingerprint_size)
        if len(valid_indices) == 0:
            raise ValueError("Aucun SMILES valide pour entraîner les prédicteurs Tox21.")
        train_y = np.asarray(train_dataset.y)
        train_w = np.asarray(train_dataset.w)
        if train_y.shape != train_w.shape or train_y.ndim != 2:
            raise ValueError("Les tableaux y et w d'entraînement doivent avoir forme (n, tâches).")
        if train_y.shape[1] != len(TASKS):
            raise ValueError(
                f"Les données d'entraînement doivent avoir {len(TASKS)} tâches."
            )
        train_y = train_y[valid_indices]
        train_w = train_w[valid_indices]

        validation_x = None
        validation_y = None
        validation_w = None
        if validation_dataset is not None:
            validation_indices, validation_x = cls._fingerprints(
                [str(smile) for smile in validation_dataset.ids],
                generator,
                fingerprint_size,
            )
            validation_y = np.asarray(validation_dataset.y)[validation_indices]
            validation_w = np.asarray(validation_dataset.w)[validation_indices]

        models = {}
        aucs = {}
        for task in TASKS:
            task_index = TASKS.index(task)
            train_labeled = train_w[:, task_index] > 0
            y_train = train_y[train_labeled, task_index].astype(int)
            x_train = x[train_labeled]
            if (
                len(np.unique(y_train)) != 2
                or np.bincount(y_train, minlength=2).min() < 2
            ):
                continue

            model = RandomForestClassifier(
                n_estimators=estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(x_train, y_train)
            models[task] = model
            if validation_dataset is not None:
                validation_labeled = validation_w[:, task_index] > 0
                y_valid = validation_y[validation_labeled, task_index].astype(int)
                if len(np.unique(y_valid)) == 2:
                    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
                    aucs[task] = float(
                        roc_auc_score(
                            y_valid,
                            model.predict_proba(
                                validation_x[validation_labeled]
                            )[:, positive_column],
                        )
                    )

        if not models:
            raise ValueError("Aucune cible Tox21 ne permet un apprentissage binaire.")
        return cls(
            models,
            aucs,
            fingerprint_size,
            fingerprint_radius,
            len(valid_indices),
        )

    def predict_task_probabilities(self, molecules) -> dict[str, np.ndarray]:
        molecules = list(molecules)
        if not molecules:
            return {task: np.empty(0, dtype=float) for task in self.models}
        vectors = []
        for molecule in molecules:
            vector = np.zeros((self.fingerprint_size,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(
                self.fingerprint_generator.GetFingerprint(molecule),
                vector,
            )
            vectors.append(vector)
        x = np.asarray(vectors)
        predictions = {}
        for task, model in self.models.items():
            positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
            predictions[task] = model.predict_proba(x)[:, positive_column]
        return predictions

    def predict_risk(self, molecules) -> np.ndarray:
        probabilities = self.predict_task_probabilities(molecules)
        if not probabilities:
            return np.empty(0, dtype=float)
        return np.mean(np.stack(list(probabilities.values())), axis=0)

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        import joblib

        predictor = joblib.load(path)
        if not isinstance(predictor, cls):
            raise TypeError(f"Fichier non reconnu comme prédicteur Tox21 : {path}")
        return predictor
