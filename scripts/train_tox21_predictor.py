"""Train one masked-label Tox21 assay-activity predictor."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import deepchem as dc
import sklearn
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tox21 import load_tox21_splits
from src.evaluation.tox21_assay import evaluate_tox21_endpoint
from src.models.tox21_endpoint import (
    ENDPOINT_DESCRIPTIONS,
    Tox21EndpointPredictor,
)

DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne un prédicteur d'activité d'essai Tox21; les labels test "
            "ne sont ni ajustés ni utilisés pour choisir le modèle."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--endpoint",
        choices=tuple(ENDPOINT_DESCRIPTIONS),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    predictor_config = config["tox21_predictor"]
    endpoint = args.endpoint or predictor_config["endpoint"]

    splits = load_tox21_splits(
        cache_dir=ROOT / config["data"]["molnet_cache"],
        featurizer=config["data"]["featurizer"],
        splitter=config["data"]["splitter"],
        reload=True,
    )
    if endpoint not in splits.tasks:
        raise ValueError(f"Endpoint {endpoint} absent des tâches chargées.")
    predictor = Tox21EndpointPredictor.fit(
        splits.train,
        splits.valid,
        endpoint=endpoint,
        estimator_candidates=predictor_config["estimator_candidates"],
        fingerprint_size=predictor_config["fingerprint_size"],
        fingerprint_radius=predictor_config["fingerprint_radius"],
        seed=config["seed"],
    )

    output_dir = ROOT / predictor_config["artifacts_dir"] / endpoint.lower().replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = output_dir / "predictor.joblib"
    predictor.save(predictor_path)
    validation_report, _ = evaluate_tox21_endpoint(
        predictor,
        splits.valid,
        split_name="validation",
        n_bootstrap=predictor_config["bootstrap_samples"],
        seed=config["seed"],
    )
    metadata = {
        "model": "sklearn.ensemble.RandomForestClassifier",
        "model_parameters": {
            "n_estimators": predictor.selected_estimators,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": predictor.random_seed,
        },
        "feature_representation": {
            "type": "RDKit Morgan/ECFP bit fingerprint",
            "radius": predictor.fingerprint_radius,
            "size_bits": predictor.fingerprint_size,
        },
        "endpoint": predictor.endpoint,
        "outcome_label": predictor.outcome_label,
        "biological_interpretation": predictor.biological_interpretation,
        "validation_selected_estimators": predictor.selected_estimators,
        "validation_candidate_roc_auc": predictor.candidate_validation_auc,
        "validation_selected_threshold": predictor.decision_threshold,
        "random_seed": predictor.random_seed,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "rdkit_version": rdBase.rdkitVersion,
        "deepchem_version": dc.__version__,
        "training_split_counts": predictor.training_counts,
        "validation_split_counts": predictor.validation_counts,
        "validation_evaluation": validation_report,
        "test_labels_used": False,
        "training_split": "official MoleculeNet Tox21 train scaffold split",
        "model_selection_split": "official MoleculeNet Tox21 validation scaffold split",
        "caveat": (
            "Predicted assay activity is an in-vitro endpoint estimate, not a "
            "determination of organism-level toxicity or safety."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    effective_config = {
        **predictor_config,
        "endpoint": endpoint,
        "seed": config["seed"],
        "selected_estimators": predictor.selected_estimators,
        "decision_threshold": predictor.decision_threshold,
    }
    (output_dir / "config.json").write_text(
        json.dumps(effective_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Endpoint {endpoint} | activité biologique prédite : "
        f"{predictor.biological_interpretation}"
    )
    print(
        f"Train observé: {predictor.training_counts['negative_count']} négatifs, "
        f"{predictor.training_counts['positive_count']} positifs, "
        f"{predictor.training_counts['masked_label_count']} masqués"
    )
    print(
        f"Validation ROC-AUC: "
        f"{validation_report['metrics']['roc_auc']:.4f} | "
        f"threshold={predictor.decision_threshold:.4f} | "
        f"n_estimators={predictor.selected_estimators}"
    )
    print(f"Modèle, configuration et rapports sauvegardés dans {output_dir}")


if __name__ == "__main__":
    main()
