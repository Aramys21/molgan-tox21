from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tox21 import (
    MOLGAN_COMPATIBLE_FILE,
    PROCESSED_FILE,
    analyze_and_prepare,
)
from src.evaluation.metrics import evaluate_molecules
from src.models.molgan import (
    build_model,
    featurize_compatible_smiles,
    generate_graphs,
    save_generator,
    train_adversarial,
)
from src.models.toxicity import Tox21ToxicityPredictor
from src.rewards.safety import reinforce_safe_generation
from src.visualization.molecules import save_molecule_grid

DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne MolGAN et guide les échantillons vers un risque Tox21 faible."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--candidate-pool", type=int)
    parser.add_argument("--toxicity-weight", type=float)
    parser.add_argument("--reward-steps", type=int)
    parser.add_argument("--limit", type=int, help="Limiter les graphes MolGAN pour test rapide.")
    args = parser.parse_args()
    config = _load_config(args.config)
    molgan_config = config["molgan"]
    reward_config = config["reward"]
    generation_config = config["generation"]

    epochs = args.epochs or molgan_config["epochs"]
    count = args.n_samples or generation_config["n_samples"]
    candidate_pool = args.candidate_pool or generation_config["candidate_pool"]
    toxicity_weight = (
        args.toxicity_weight
        if args.toxicity_weight is not None
        else reward_config["toxicity_weight"]
    )
    reward_steps = (
        args.reward_steps if args.reward_steps is not None else reward_config["steps"]
    )
    if epochs < 1 or count < 1 or candidate_pool < 1 or reward_steps < 0:
        parser.error("epochs/samples/pool doivent être > 0 et reward-steps >= 0.")
    if toxicity_weight < 0:
        parser.error("--toxicity-weight doit être positif ou nul.")

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    raw_file = ROOT / config["data"]["raw_file"]
    processed_dir = ROOT / config["data"]["processed_dir"]
    clean_file = processed_dir / PROCESSED_FILE.name
    compatible_file = processed_dir / MOLGAN_COMPATIBLE_FILE.name
    if not clean_file.is_file() or not compatible_file.is_file():
        print("Préparation des données nettoyées...")
        analyze_and_prepare(raw_file, processed_dir)

    clean_frame = pd.read_csv(clean_file)
    compatible_frame = pd.read_csv(compatible_file)
    print(f"Données Tox21 nettoyées : {len(clean_frame)} structures.")
    predictor_config = config["toxicity"]
    predictor = Tox21ToxicityPredictor.fit(
        clean_frame,
        fingerprint_size=predictor_config["fingerprint_size"],
        fingerprint_radius=predictor_config["fingerprint_radius"],
        estimators=predictor_config["estimators"],
        test_size=predictor_config["test_size"],
        seed=config["seed"],
    )
    aucs_path = ROOT / config["artifacts_dir"] / "toxicity_validation_auc.json"
    aucs_path.parent.mkdir(parents=True, exist_ok=True)
    aucs_path.write_text(
        json.dumps(predictor.validation_aucs, indent=2),
        encoding="utf-8",
    )
    predictor.save(aucs_path.parent / "tox21_toxicity.joblib")
    print("ROC-AUC de validation :")
    for task, auc in predictor.validation_aucs.items():
        print(f"  {task}: {auc:.3f}")

    featurizer, features = featurize_compatible_smiles(
        compatible_frame["smiles"].astype(str).tolist(),
        max_molecules=args.limit or 0,
    )
    print(f"Graphes MolGAN chargés : {len(features)}")
    model = build_model(config, seed=config["seed"])
    print(f"Pré-entraînement adversarial : {epochs} époques.")
    train_adversarial(
        model,
        features,
        epochs=epochs,
        generator_steps=molgan_config["generator_steps"],
    )

    print(
        f"Fine-tuning récompense sûreté : poids={toxicity_weight:g}, "
        f"étapes={reward_steps}."
    )
    reinforce_safe_generation(
        model,
        featurizer,
        predictor,
        toxicity_weight=toxicity_weight,
        steps=reward_steps,
        batch_size=reward_config["batch_size"],
        learning_rate=reward_config["learning_rate"],
    )

    candidates = []
    unique = set()
    for _ in range(0, candidate_pool, min(128, candidate_pool)):
        batch_count = min(128, candidate_pool - len(candidates))
        if batch_count <= 0:
            break
        for graph in generate_graphs(model, batch_count):
            molecule = featurizer.defeaturize([graph])[0]
            if molecule is None:
                continue
            smiles = Chem.MolToSmiles(molecule, canonical=True)
            if smiles and smiles not in unique:
                unique.add(smiles)
                candidates.append((smiles, molecule))
    risks = predictor.predict_risk([molecule for _, molecule in candidates])
    ranked_candidates = sorted(
        [
            (smiles, molecule, float(risk))
            for (smiles, molecule), risk in zip(candidates, risks)
        ],
        key=lambda item: item[2],
    )
    selected = ranked_candidates[:count]
    artifacts_dir = ROOT / config["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_generator(model, artifacts_dir / "molgan_generator.pt")
    predictor.save(artifacts_dir / "tox21_toxicity.joblib")

    results = []
    for smiles, molecule, risk in selected:
        results.append(
            {
                "smiles": smiles,
                "predicted_toxicity_risk": risk,
            }
        )
    results_path = artifacts_dir / "generated_molecules.csv"
    pd.DataFrame(
        results,
        columns=["smiles", "predicted_toxicity_risk"],
    ).to_csv(results_path, index=False)
    if selected:
        save_molecule_grid(
            [molecule for _, molecule, _ in selected],
            artifacts_dir / "generated_molecules.png",
            legends=[
                f"risque={result['predicted_toxicity_risk']:.2f}"
                for result in results
            ],
        )
    generated_molecules = [molecule for _, molecule, _ in selected]
    metrics = evaluate_molecules(generated_molecules, predictor)
    metrics_path = artifacts_dir / "evaluation.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(
        f"Molécules valides et uniques dans le pool : {len(candidates)}/{candidate_pool}."
    )
    print(f"Molécules retenues : {len(selected)}/{count}.")
    print(f"Évaluation : {metrics}")
    print(f"Résultats sauvegardés dans {artifacts_dir}")


if __name__ == "__main__":
    main()
