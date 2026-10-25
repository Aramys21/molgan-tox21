"""Compare a preserved MolGAN baseline with two NR-AR reward settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import deepchem as dc
import numpy as np
import pandas as pd
import sklearn
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.molgan_graphs import featurize_smiles, reconstruct_graphs
from src.data.tox21 import load_tox21_splits
from src.evaluation.tox21_guided import evaluate_tox21_guided_samples
from src.models.molgan import (
    build_model,
    generate_graphs,
    load_generator,
    resolve_device,
    seed_everything,
)
from src.models.tox21_endpoint import Tox21EndpointPredictor
from src.rewards.tox21_guided import reinforce_tox21_generator
from src.visualization.molecules import save_molecule_grid

DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _evaluate_arm(
    name: str,
    graphs,
    featurizer,
    predictor: Tox21EndpointPredictor,
    training_smiles: set[str],
    training_fingerprints,
    output_dir: Path,
    similarity_threshold: float,
) -> dict[str, Any]:
    reconstruction = reconstruct_graphs(graphs, featurizer)
    report, samples = evaluate_tox21_guided_samples(
        reconstruction,
        predictor,
        training_smiles,
        training_fingerprints,
        out_of_domain_similarity_threshold=similarity_threshold,
    )
    report["arm"] = name
    report["endpoint"] = predictor.endpoint
    report["biological_interpretation"] = predictor.biological_interpretation
    report["prediction_target"] = (
        "predicted in-vitro NR-AR reporter-assay activity; not organism-level "
        "toxicity or safety"
    )
    arm_dir = output_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    _save_json(arm_dir / "evaluation.json", report)
    pd.DataFrame(samples).to_csv(arm_dir / "samples.csv", index=False)
    valid_molecules = [
        reconstruction.molecules[index]
        for index in range(reconstruction.input_count)
        if reconstruction.molecules[index] is not None
    ]
    valid_smiles = [
        reconstruction.canonical_smiles[index]
        for index in range(reconstruction.input_count)
        if reconstruction.molecules[index] is not None
    ]
    if valid_molecules:
        save_molecule_grid(
            valid_molecules[:16],
            arm_dir / "molecule_grid.png",
            legends=[str(value) for value in valid_smiles[:16]],
        )
    return report


def _diagnose_arms(
    reports: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Flag possible issues for review without treating heuristic cutoffs as proof."""
    baseline = reports[0]
    diagnostics = []
    for report in reports:
        unique_count = int(report["unique_valid_count"])
        if unique_count < 2:
            mode_collapse: bool | None = None
            exploitation: bool | None = None
            reason = "insufficient unique valid molecules"
        else:
            diversity = report["mean_pairwise_tanimoto_distance_unique"]
            mode_collapse = bool(
                report["uniqueness_rate"] < thresholds["low_uniqueness_rate"]
                or (
                    diversity is not None
                    and diversity < thresholds["low_tanimoto_distance"]
                )
            )
            baseline_has_samples = int(baseline["unique_valid_count"]) >= 2
            if report["arm"] == "baseline" or not baseline_has_samples:
                exploitation = None
                reason = (
                    "baseline comparison unavailable"
                    if not baseline_has_samples
                    else "unmodified baseline"
                )
            else:
                activity_reduced = (
                    report["unique_sample_activity_probability_mean"]
                    < baseline["unique_sample_activity_probability_mean"]
                )
                support_degraded = (
                    report["out_of_domain_valid_rate"]
                    - baseline["out_of_domain_valid_rate"]
                    >= thresholds["ood_fraction_increase"]
                )
                uncertainty_increased = (
                    report["mean_forest_tree_disagreement_unique"]
                    - baseline["mean_forest_tree_disagreement_unique"]
                    >= thresholds["uncertainty_increase"]
                )
                exploitation = bool(
                    activity_reduced and (support_degraded or uncertainty_increased)
                )
                reason = "relative to baseline on unique valid samples"
        diagnostics.append(
            {
                "arm": report["arm"],
                "possible_mode_collapse_flag": mode_collapse,
                "possible_predictor_exploitation_flag": exploitation,
                "assessment": reason,
            }
        )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MolGAN baseline et deux réglages REINFORCE pour réduire "
            "l'activité prédite de l'essai Tox21 NR-AR."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--baseline-generator", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = _load_config(args.config)
    experiment = config["tox21_guided_comparison"]
    endpoint = experiment["endpoint"]
    settings = experiment["settings"]
    if endpoint != "NR-AR":
        parser.error("Cette comparaison expérimentale est explicitement configurée pour NR-AR.")
    if len(settings) != 2:
        parser.error("La comparaison exige exactement deux réglages de récompense.")
    steps = args.steps if args.steps is not None else int(experiment["steps"])
    sample_count = (
        args.samples if args.samples is not None else int(experiment["sample_count"])
    )
    if steps < 1 or sample_count < 1:
        parser.error("--steps et --samples doivent être strictement positifs.")

    artifact_root = ROOT / config["artifacts_dir"]
    baseline_path = args.baseline_generator or (
        artifact_root / "molgan_baseline" / "generator_weights.pt"
    )
    predictor_path = (
        artifact_root
        / "tox21_predictors"
        / endpoint.lower().replace("-", "_")
        / "predictor.joblib"
    )
    output_dir = args.output_dir or (
        artifact_root / f"tox21_guided_{endpoint.lower().replace('-', '_')}"
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Générateur baseline absent : {baseline_path}. "
            "Exécutez d'abord scripts/train_baseline.py."
        )
    predictor = Tox21EndpointPredictor.load(predictor_path)
    if predictor.endpoint != endpoint:
        raise ValueError(
            f"Le prédicteur sauvegardé couvre {predictor.endpoint}, pas {endpoint}."
        )

    splits = load_tox21_splits(
        cache_dir=ROOT / config["data"]["molnet_cache"],
        featurizer=config["data"]["featurizer"],
        splitter=config["data"]["splitter"],
        reload=True,
    )
    molgan_config = config["molgan"]
    preparation = featurize_smiles(
        (str(smiles) for smiles in splits.train.ids),
        max_atom_count=molgan_config["vertices"],
    )
    training_smiles = set(preparation.canonical_smiles)
    training_fingerprints, invalid_reference_count = predictor.reference_fingerprints(
        splits.train
    )
    requested_device = args.device or config["baseline"]["device"]
    device = resolve_device(requested_device)
    seed = int(config["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_reports = []
    seed_everything(seed)
    baseline_model = build_model(
        config,
        seed=seed,
        device=device.type,
    )
    load_generator(baseline_model, baseline_path)
    baseline_graphs = generate_graphs(baseline_model, sample_count)
    arm_reports.append(
        _evaluate_arm(
            "baseline",
            baseline_graphs,
            preparation.featurizer,
            predictor,
            training_smiles,
            training_fingerprints,
            output_dir,
            float(experiment["out_of_domain_similarity_threshold"]),
        )
    )

    for setting in settings:
        name = str(setting["name"])
        if not name.isidentifier() or name == "baseline":
            parser.error(f"Nom de réglage d'artefact non valide : {name!r}.")
        seed_everything(seed)
        model = build_model(config, seed=seed, device=device.type)
        load_generator(model, baseline_path)
        logs = reinforce_tox21_generator(
            model,
            preparation.featurizer,
            predictor,
            training_fingerprints,
            weights=setting["weights"],
            steps=steps,
            batch_size=int(experiment["batch_size"]),
            learning_rate=float(experiment["learning_rate"]),
        )
        pd.DataFrame(logs).to_csv(output_dir / f"{name}_reward_components.csv", index=False)
        generated_graphs = generate_graphs(model, sample_count)
        arm_reports.append(
            _evaluate_arm(
                name,
                generated_graphs,
                preparation.featurizer,
                predictor,
                training_smiles,
                training_fingerprints,
                output_dir,
                float(experiment["out_of_domain_similarity_threshold"]),
            )
        )

    comparison = {
        "endpoint": endpoint,
        "biological_interpretation": predictor.biological_interpretation,
        "objective": (
            "Reduce predicted NR-AR Tox21 reporter-assay activity; this is a "
            "narrow assay objective and not a general safety objective."
        ),
        "seed": seed,
        "device": str(device),
        "baseline_generator": str(baseline_path.relative_to(ROOT))
        if baseline_path.is_relative_to(ROOT)
        else str(baseline_path),
        "predictor": str(predictor_path.relative_to(ROOT)),
        "train_reference_molecule_count": len(training_smiles),
        "train_reference_invalid_observed_structure_count": invalid_reference_count,
        "train_smiles_source": "MolGAN-featurizable molecules in official train split",
        "support_reference_source": "observed NR-AR labels in official train split",
        "steps_per_guided_arm": steps,
        "batch_size": int(experiment["batch_size"]),
        "generated_graphs_per_arm": sample_count,
        "settings": settings,
        "arms": arm_reports,
        "diagnostic_thresholds": experiment["diagnostic_thresholds"],
        "diagnostics": _diagnose_arms(
            arm_reports,
            experiment["diagnostic_thresholds"],
        ),
        "package_versions": {
            "python": __import__("platform").python_version(),
            "deepchem": dc.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
            "torch": torch.__version__,
        },
        "uncertainty_limitation": (
            "Forest-tree probability disagreement is an uncalibrated proxy; "
            "it is not a probability that a prediction is wrong."
        ),
        "interpretation_limitation": (
            "All generated-sample outcomes are computational predictions, not "
            "experimental toxicology results."
        ),
    }
    _save_json(output_dir / "comparison.json", comparison)
    print(f"Comparaison terminée : {output_dir}")
    for report in arm_reports:
        activity_rate = report["unique_sample_predicted_active_rate"]
        activity_text = (
            "n/a" if activity_rate is None else f"{activity_rate:.3f}"
        )
        print(
            f"{report['arm']}: validité={report['validity_rate']:.3f}, "
            f"unicité={report['uniqueness_rate']:.3f}, "
            f"activité NR-AR={activity_text}"
        )


if __name__ == "__main__":
    main()
