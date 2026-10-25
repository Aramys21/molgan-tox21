"""Train the unmodified DeepChem MolGAN baseline on official Tox21 train data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import deepchem as dc
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.molgan_graphs import featurize_smiles, reconstruct_graphs
from src.data.tox21 import load_tox21_splits
from src.models.molgan import (
    build_model,
    resolve_device,
    save_generator,
    seed_everything,
    train_adversarial,
)

DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _generated_metrics(
    model,
    featurizer,
    sample_count: int,
    training_smiles: set[str],
) -> dict[str, Any]:
    graphs = model.predict_gan_generator(sample_count)
    reconstruction = reconstruct_graphs(graphs, featurizer)
    valid_smiles = [
        smile for smile in reconstruction.canonical_smiles if smile is not None
    ]
    unique_smiles = set(valid_smiles)
    return {
        "generated_graph_count": len(graphs),
        "decoded_and_sanitized_count": reconstruction.sanitized_count,
        "validity_rate": (
            reconstruction.sanitized_count / len(graphs) if graphs else 0.0
        ),
        "unique_valid_count": len(unique_smiles),
        "uniqueness_rate": (
            len(unique_smiles) / reconstruction.sanitized_count
            if reconstruction.sanitized_count
            else 0.0
        ),
        "novel_valid_count": len(unique_smiles - training_smiles),
        "novelty_rate": (
            len(unique_smiles - training_smiles) / len(unique_smiles)
            if unique_smiles
            else 0.0
        ),
        "reconstruction": reconstruction.summary(),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne le baseline MolGAN/WGAN-GP natif de DeepChem, sans "
            "récompense de propriété ni optimisation de toxicité."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Force un entraînement court et réduit avant l'entraînement complet.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restaure le dernier checkpoint DeepChem du dossier d'artefacts.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    molgan_config = dict(config["molgan"])
    baseline_config = config["baseline"]
    smoke_config = baseline_config["smoke_test"]
    epochs = (
        args.epochs
        if args.epochs is not None
        else smoke_config["epochs"]
        if args.smoke_test
        else molgan_config["epochs"]
    )
    max_molecules = args.limit
    if max_molecules is None and args.smoke_test:
        max_molecules = smoke_config["max_molecules"]
    sample_count = (
        args.samples
        if args.samples is not None
        else smoke_config["generation_samples"]
        if args.smoke_test
        else baseline_config["generation_samples_per_epoch"]
    )
    if args.smoke_test:
        molgan_config["batch_size"] = smoke_config["batch_size"]
        molgan_config["generator_steps"] = smoke_config["generator_steps"]
    if epochs < 1 or sample_count < 1 or (max_molecules is not None and max_molecules < 1):
        parser.error("epochs, samples et limit doivent être strictement positifs.")

    seed = int(config["seed"])
    seed_everything(seed)
    device_name = args.device or baseline_config["device"]
    device = resolve_device(device_name)
    cache_dir = ROOT / config["data"]["molnet_cache"]
    splits = load_tox21_splits(
        cache_dir=cache_dir,
        featurizer=config["data"]["featurizer"],
        splitter=config["data"]["splitter"],
        reload=True,
    )

    preparation = featurize_smiles(
        (str(smile) for smile in splits.train.ids),
        max_atom_count=molgan_config["vertices"],
        max_molecules=max_molecules,
    )
    if not preparation.graphs:
        raise RuntimeError(
            "Aucune molécule de train ne respecte le featurizer MolGAN; "
            "consultez les statistiques de préparation."
        )
    if len(preparation.featurizer.atom_labels) != molgan_config["nodes"]:
        raise ValueError("molgan.nodes ne correspond pas aux labels d'atome.")
    if len(preparation.featurizer.bond_labels) != molgan_config["edges"]:
        raise ValueError("molgan.edges ne correspond pas aux labels de liaison.")

    artifacts_dir = ROOT / config["artifacts_dir"] / "molgan_baseline"
    checkpoint_dir = artifacts_dir / "checkpoints"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        {**config, "molgan": molgan_config},
        seed=seed,
        device=device.type,
        model_dir=checkpoint_dir,
    )
    if args.resume and not model.get_checkpoints(model_dir=str(checkpoint_dir)):
        raise FileNotFoundError(
            f"Aucun checkpoint DeepChem à reprendre dans {checkpoint_dir}."
        )

    training_smiles = set(preparation.canonical_smiles)
    epoch_records: list[dict[str, Any]] = []
    loss_records: list[dict[str, Any]] = []
    print(
        f"Baseline MolGAN DeepChem {dc.__version__}: device={device}, "
        f"train graphs={len(preparation.graphs)}, epochs={epochs}, "
        f"batch={molgan_config['batch_size']}, seed={seed}"
    )
    print(
        "Objectifs natifs WGAN-GP; aucune BCE, récompense de propriété ou "
        "optimisation Tox21 n'est ajoutée."
    )

    checkpoint_interval = int(baseline_config["checkpoint_interval"])
    keep_checkpoints = int(baseline_config["max_checkpoints_to_keep"])
    for epoch in range(1, epochs + 1):
        epoch_loss_start = len(model.training_losses)
        epoch_losses = train_adversarial(
            model,
            preparation.graphs,
            epochs=1,
            generator_steps=float(molgan_config["generator_steps"]),
            checkpoint_interval=checkpoint_interval,
            max_checkpoints_to_keep=keep_checkpoints,
            restore=args.resume and epoch == 1,
        )
        if not epoch_losses:
            raise RuntimeError(
                "DeepChem n'a exposé aucune perte de générateur/discriminateur "
                "pour cette époque."
            )
        if any(not math.isfinite(float(record["loss"])) for record in epoch_losses):
            raise FloatingPointError(
                f"Perte non finie détectée à l'époque {epoch}."
            )
        loss_records.extend(
            {"epoch": epoch, **record}
            for record in model.training_losses[epoch_loss_start:]
        )

        metrics = _generated_metrics(
            model,
            preparation.featurizer,
            sample_count,
            training_smiles,
        )
        epoch_record = {
            "epoch": epoch,
            "global_step": int(model.get_global_step()),
            "training_graph_count": len(preparation.graphs),
            "device": str(device),
            "seed": seed,
            "property_reward": None,
            "generator_loss_mean": _mean_loss(epoch_losses, "generator"),
            "discriminator_loss_mean": _mean_loss(epoch_losses, "discriminator"),
            **metrics,
        }
        epoch_records.append(epoch_record)
        print(
            f"epoch={epoch}/{epochs} "
            f"G_loss={epoch_record['generator_loss_mean']} "
            f"D_loss={epoch_record['discriminator_loss_mean']} "
            f"valid={metrics['decoded_and_sanitized_count']}/"
            f"{metrics['generated_graph_count']} "
            f"unique={metrics['unique_valid_count']}"
        )

        # fit_gan writes a native checkpoint at the end of each epoch if enabled.
        if checkpoint_interval == 0:
            model.save_checkpoint(keep_checkpoints, model_dir=str(checkpoint_dir))
        pd.DataFrame(loss_records).to_csv(
            artifacts_dir / "training_losses.csv",
            index=False,
        )
        pd.DataFrame(epoch_records).to_csv(
            artifacts_dir / "epoch_metrics.csv",
            index=False,
        )
        _save_json(
            artifacts_dir / "training_metrics.json",
            {
                "run_type": "smoke_test" if args.smoke_test else "baseline",
                "epochs_completed": epoch,
                "training_graph_preparation": preparation.summary(),
                "epochs": epoch_records,
                "loss_updates": loss_records,
            },
        )

    save_generator(model, artifacts_dir / "generator_weights.pt")
    effective_config = {
        **config,
        "seed": seed,
        "molgan": molgan_config,
        "baseline": {
            **baseline_config,
            "device": str(device),
            "resume_requested": args.resume,
            "smoke_test": args.smoke_test,
        },
    }
    _save_json(artifacts_dir / "config.json", effective_config)
    metadata = {
        "model": "deepchem.models.torch_models.BasicMolGANModel",
        "deepchem_version": dc.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": seed,
        "data_source": "MoleculeNet Tox21 official scaffold train split",
        "feature": {
            "featurizer": "deepchem.feat.MolGanFeaturizer",
            "vertices": molgan_config["vertices"],
            "atom_categories": preparation.featurizer.atom_labels,
            "bond_categories": [
                str(label) for label in preparation.featurizer.bond_labels
            ],
            "training_graphs": len(preparation.graphs),
        },
        "training": {
            "epochs_completed": epochs,
            "generator_steps_per_discriminator_batch": molgan_config[
                "generator_steps"
            ],
            "gradient_penalty_coefficient": baseline_config["gradient_penalty"],
            "resume_requested": args.resume,
            "loss_objectives": {
                "generator": "DeepChem WGANModel: mean critic score of generated graphs",
                "discriminator": (
                    "DeepChem WGANModel: mean(real critic score - fake critic "
                    "score) plus gradient penalty"
                ),
            },
            "property_reward": None,
        },
        "artifacts": {
            "checkpoint_directory": str(checkpoint_dir.relative_to(ROOT)),
            "generator_weights": str(
                (artifacts_dir / "generator_weights.pt").relative_to(ROOT)
            ),
            "losses_csv": str(
                (artifacts_dir / "training_losses.csv").relative_to(ROOT)
            ),
            "metrics_json": str(
                (artifacts_dir / "training_metrics.json").relative_to(ROOT)
            ),
        },
    }
    _save_json(artifacts_dir / "metadata.json", metadata)
    print(f"Baseline terminé. Checkpoints et métriques : {artifacts_dir}")


def _mean_loss(
    records: list[dict[str, float | int | str]],
    component: str,
) -> float | None:
    values = [
        float(record["loss"])
        for record in records
        if record["component"] == component
    ]
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    main()
