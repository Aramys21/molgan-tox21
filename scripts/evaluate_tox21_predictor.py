"""Evaluate the saved Tox21 endpoint predictor on the untouched test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tox21 import load_tox21_splits
from src.evaluation.tox21_assay import evaluate_tox21_endpoint
from src.models.tox21_endpoint import Tox21EndpointPredictor

DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Évalue le modèle Tox21 sauvegardé une seule fois sur la partition "
            "test officielle; aucun réajustement n'est effectué."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--endpoint")
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    predictor_config = config["tox21_predictor"]
    endpoint = args.endpoint or predictor_config["endpoint"]
    endpoint_dir = endpoint.lower().replace("-", "_")
    output_dir = args.output_dir or (
        ROOT / predictor_config["artifacts_dir"] / endpoint_dir
    )
    predictor_path = args.predictor or output_dir / "predictor.joblib"
    if not predictor_path.is_file():
        raise FileNotFoundError(
            f"Prédicteur absent : {predictor_path}. Exécutez d'abord "
            "scripts/train_tox21_predictor.py."
        )
    predictor = Tox21EndpointPredictor.load(predictor_path)
    if predictor.endpoint != endpoint:
        raise ValueError(
            f"Le modèle prédit {predictor.endpoint}, endpoint demandé {endpoint}."
        )

    splits = load_tox21_splits(
        cache_dir=ROOT / config["data"]["molnet_cache"],
        featurizer=config["data"]["featurizer"],
        splitter=config["data"]["splitter"],
        reload=True,
    )
    report, prediction_rows = evaluate_tox21_endpoint(
        predictor,
        splits.test,
        split_name="test",
        n_bootstrap=predictor_config["bootstrap_samples"],
        seed=config["seed"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "test_evaluation.json"
    predictions_path = output_dir / "test_predictions.csv"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    print(
        f"Évaluation test finale — {endpoint} (activité d'essai prédite), "
        f"n={report['sample_count']}, "
        f"négatifs={report['class_counts']['negative']}, "
        f"positifs={report['class_counts']['positive']}"
    )
    for metric, value in report["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    print(
        f"Ce résultat prédit une activité biologique dans l'essai {endpoint}; "
        "il ne constitue pas une preuve de toxicité ni de sécurité."
    )
    print(f"Rapport JSON : {report_path}")
    print(f"Prédictions CSV : {predictions_path}")


if __name__ == "__main__":
    main()
