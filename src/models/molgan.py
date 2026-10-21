"""DeepChem MolGAN training, sampling, and checkpoint persistence."""

from __future__ import annotations

from pathlib import Path

import deepchem as dc
import numpy as np
import torch
import torch.nn.functional as F
from deepchem.models.optimizers import ExponentialDecay
from deepchem.models.torch_models import BasicMolGANModel
from rdkit import Chem


def build_model(config: dict, seed: int = 42) -> BasicMolGANModel:
    model_config = config["molgan"]
    torch.manual_seed(seed)
    return BasicMolGANModel(
        learning_rate=ExponentialDecay(
            model_config["learning_rate"],
            0.9,
            5000,
        ),
        vertices=model_config["vertices"],
        nodes=model_config["nodes"],
        edges=model_config["edges"],
        embedding_dim=model_config["embedding_dim"],
        batch_size=model_config["batch_size"],
    )


def featurize_compatible_smiles(smiles: list[str], max_molecules: int = 0):
    featurizer = dc.feat.MolGanFeaturizer(max_atom_count=9)
    features = []
    for smile in smiles:
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None or molecule.GetNumAtoms() > featurizer.max_atom_count:
            continue
        graph = featurizer.featurize([molecule])[0]
        if graph is not None and hasattr(graph, "adjacency_matrix"):
            features.append(graph)
            if max_molecules > 0 and len(features) >= max_molecules:
                break
    return featurizer, features


def _training_batches(model, features, epochs: int):
    dataset = dc.data.NumpyDataset(
        [graph.adjacency_matrix for graph in features],
        [graph.node_features for graph in features],
    )
    for _ in range(epochs):
        for batch in dataset.iterbatches(
            batch_size=model.batch_size,
            pad_batches=True,
        ):
            adjacency = F.one_hot(
                torch.as_tensor(batch[0], dtype=torch.int64),
                model.edges,
            ).to(torch.float32)
            nodes = F.one_hot(
                torch.as_tensor(batch[1], dtype=torch.int64),
                model.nodes,
            ).to(torch.float32)
            yield {model.data_inputs[0]: adjacency, model.data_inputs[1]: nodes}


def train_adversarial(
    model: BasicMolGANModel,
    features,
    epochs: int,
    generator_steps: float,
) -> None:
    if not features:
        raise ValueError("MolGAN exige au moins un graphe moléculaire.")
    if epochs < 1:
        raise ValueError("Le nombre d'époques doit être strictement positif.")
    model.fit_gan(
        _training_batches(model, features, epochs),
        generator_steps=generator_steps,
        checkpoint_interval=0,
    )


def generate_graphs(model: BasicMolGANModel, count: int):
    if count < 1:
        raise ValueError("Le nombre de graphes à générer doit être > 0.")
    return model.predict_gan_generator(count)


def save_generator(model: BasicMolGANModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state_dict": model.generators[0].state_dict(),
            "model_config": {
                "vertices": model.vertices,
                "nodes": model.nodes,
                "edges": model.edges,
                "embedding_dim": model.embedding_dim,
            },
        },
        path,
    )
