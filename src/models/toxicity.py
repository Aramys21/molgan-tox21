"""Morgan-fingerprint toxicity models trained on the Tox21 assay labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.tox21 import TASKS


@dataclass
class Tox21ToxicityPredictor:
    """Per-task classifiers with an aggregate probability of any Tox21 activity."""

    models: dict
    validation_aucs: dict[str, float]
    fingerprint_size: int = 2048
    fingerprint_radius: int = 2

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
        frame: pd.DataFrame,
        fingerprint_size: int = 2048,
        fingerprint_radius: int = 2,
        estimators: int = 100,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> "Tox21ToxicityPredictor":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fingerprint_radius,
            fpSize=fingerprint_size,
        )
        smiles = frame["smiles"].astype(str).tolist()
        valid_indices, x = cls._fingerprints(smiles, generator, fingerprint_size)
        if len(valid_indices) == 0:
            raise ValueError("Aucun SMILES valide pour entraîner les prédicteurs Tox21.")
        valid_frame = frame.iloc[valid_indices].reset_index(drop=True)

        models = {}
        aucs = {}
        for task in TASKS:
            labels = pd.to_numeric(valid_frame[task], errors="coerce").to_numpy()
            labeled = np.isfinite(labels)
            x_task = x[labeled]
            y_task = labels[labeled].astype(int)
            if len(np.unique(y_task)) != 2 or np.bincount(y_task, minlength=2).min() < 2:
                continue

            x_train, x_valid, y_train, y_valid = train_test_split(
                x_task,
                y_task,
                test_size=test_size,
                random_state=seed,
                stratify=y_task,
            )
            validation_model = RandomForestClassifier(
                n_estimators=estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
            validation_model.fit(x_train, y_train)
            positive_column = int(np.flatnonzero(validation_model.classes_ == 1)[0])
            aucs[task] = float(
                roc_auc_score(
                    y_valid,
                    validation_model.predict_proba(x_valid)[:, positive_column],
                )
            )

            final_model = RandomForestClassifier(
                n_estimators=estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
            final_model.fit(x_task, y_task)
            models[task] = final_model

        if not models:
            raise ValueError("Aucune cible Tox21 ne permet un apprentissage binaire.")
        return cls(models, aucs, fingerprint_size, fingerprint_radius)

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
