"""DeepChem MolGAN training, sampling, and checkpoint persistence."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import deepchem as dc
import numpy as np
import torch
import torch.nn.functional as F
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
from deepchem.models.optimizers import ExponentialDecay
from deepchem.models.torch_models import BasicMolGANModel


class TrackedMolGANModel(BasicMolGANModel):
    """DeepChem MolGAN with observed losses and unchanged native updates.

    Its generator maps random noise to differentiable categorical node/edge
    distributions. The discriminator is a scalar Wasserstein critic, not a
    sigmoid classifier. This baseline has no separate property-reward network.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.training_losses: list[dict[str, float | int | str]] = []
        self._loss_update = 0

    def restore(
        self,
        checkpoint: str | None = None,
        model_dir: str | None = None,
        strict: bool | None = True,
    ) -> None:
        """Restore DeepChem weights, resizing its batch-shaped noise parameter.

        DeepChem 2.8.0 stores the last training batch in ``GAN.noise_input``.
        A freshly built model initializes that parameter with batch size one,
        causing the stock restore to fail on a fixed-size training checkpoint.
        """
        self._ensure_built()
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if not checkpoints:
                return super().restore(
                    checkpoint=checkpoint,
                    model_dir=model_dir,
                    strict=strict,
                )
            checkpoint = checkpoints[0]

        checkpoint_data = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        state_dict = checkpoint_data.get("model_state_dict", {})
        saved_noise = state_dict.get("noise_input")
        if saved_noise is not None:
            current_noise = self.model.noise_input
            if current_noise.shape != saved_noise.shape:
                if (
                    current_noise.ndim != saved_noise.ndim
                    or current_noise.shape[1:] != saved_noise.shape[1:]
                ):
                    raise ValueError(
                        "Le checkpoint MolGAN a une dimension de bruit "
                        f"incompatible: {tuple(saved_noise.shape)}."
                    )
                current_noise.data = torch.empty_like(
                    saved_noise,
                    device=self.device,
                )

        super().restore(
            checkpoint=checkpoint,
            model_dir=model_dir,
            strict=strict,
        )

    def fit_generator(
        self,
        generator: Iterable[tuple[Any, Any, Any]],
        max_checkpoints_to_keep: int = 5,
        checkpoint_interval: int = 1000,
        restore: bool = False,
        variables: Any = None,
        loss: Callable | None = None,
        callbacks: Any = (),
        all_losses: list[float] | None = None,
    ) -> float:
        """Call DeepChem's optimizer unchanged and retain its measured loss."""
        loss_value = super().fit_generator(
            generator=generator,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            checkpoint_interval=checkpoint_interval,
            restore=restore,
            variables=variables,
            loss=loss,
            callbacks=callbacks,
            all_losses=all_losses,
        )
        self._loss_update += 1
        loss_name = getattr(loss, "__name__", "unknown")
        if loss_name == "discrim_loss_fn":
            component = "discriminator"
        elif loss_name == "gen_loss_fn":
            component = "generator"
        else:
            component = loss_name
        self.training_losses.append(
            {
                "update": self._loss_update,
                "component": component,
                "loss": float(loss_value),
            }
        )
        return float(loss_value)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve and validate the requested CPU/CUDA/MPS training device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA a été demandé mais PyTorch ne le détecte pas.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS a été demandé mais PyTorch ne le détecte pas.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Type de périphérique non pris en charge : {device.type}.")
    return device


def build_model(
    config: dict,
    seed: int = 42,
    *,
    device: str = "auto",
    model_dir: Path | None = None,
) -> TrackedMolGANModel:
    """Build DeepChem's WGAN-backed MolGAN with configured graph dimensions."""
    model_config = config["molgan"]
    seed_everything(seed)
    training_config = config.get("baseline", {})
    resolved_device = resolve_device(device)
    model_kwargs: dict[str, Any] = {
        "learning_rate": ExponentialDecay(
            model_config["learning_rate"],
            training_config.get("learning_rate_decay_rate", 0.9),
            training_config.get("learning_rate_decay_steps", 5000),
        ),
        "gradient_penalty": training_config.get("gradient_penalty", 10.0),
        "vertices": model_config["vertices"],
        "nodes": model_config["nodes"],
        "edges": model_config["edges"],
        "embedding_dim": model_config["embedding_dim"],
        "batch_size": model_config["batch_size"],
        "device": resolved_device,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "log_frequency": training_config.get("log_frequency", 100),
    }
    return TrackedMolGANModel(**model_kwargs)


def _training_batches(
    model: BasicMolGANModel,
    features: Sequence[GraphMatrix],
    epochs: int,
    *,
    deterministic: bool = True,
):
    """One-hot encode DeepChem MolGAN graphs for its native discriminator."""
    dataset = dc.data.NumpyDataset(
        [graph.adjacency_matrix for graph in features],
        [graph.node_features for graph in features],
    )
    for _ in range(epochs):
        for batch in dataset.iterbatches(
            batch_size=model.batch_size,
            pad_batches=True,
            deterministic=deterministic,
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
    features: Sequence[GraphMatrix],
    epochs: int,
    generator_steps: float,
    *,
    checkpoint_interval: int = 1000,
    max_checkpoints_to_keep: int = 3,
    restore: bool = False,
) -> list[dict[str, float | int | str]]:
    """Train with DeepChem's native WGAN-GP objectives; no property reward."""
    if not features:
        raise ValueError("MolGAN exige au moins un graphe moléculaire.")
    if epochs < 1:
        raise ValueError("Le nombre d'époques doit être strictement positif.")
    if generator_steps <= 0:
        raise ValueError("generator_steps doit être strictement positif.")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval doit être positif ou nul.")
    if max_checkpoints_to_keep < 1:
        raise ValueError("max_checkpoints_to_keep doit être strictement positif.")

    start = len(getattr(model, "training_losses", []))
    for epoch in range(epochs):
        model.fit_gan(
            _training_batches(model, features, 1),
            generator_steps=generator_steps,
            checkpoint_interval=checkpoint_interval,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            restore=restore and epoch == 0,
        )
    return list(getattr(model, "training_losses", [])[start:])


def generate_graphs(model: BasicMolGANModel, count: int):
    """Sample graph matrices from DeepChem's trained generator."""
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


def load_generator(model: BasicMolGANModel, path: Path) -> None:
    """Load generator-only weights after checking the MolGAN graph dimensions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint générateur MolGAN introuvable : {path}")
    checkpoint = torch.load(path, map_location=model.device, weights_only=True)
    expected_config = {
        "vertices": model.vertices,
        "nodes": model.nodes,
        "edges": model.edges,
        "embedding_dim": model.embedding_dim,
    }
    if checkpoint.get("model_config") != expected_config:
        raise ValueError(
            "Les dimensions du générateur sauvegardé ne correspondent pas au modèle : "
            f"{checkpoint.get('model_config')} != {expected_config}."
        )
    state_dict = checkpoint.get("generator_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Le fichier ne contient pas de poids générateur MolGAN.")
    model.generators[0].load_state_dict(state_dict, strict=True)
