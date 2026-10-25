"""Policy-gradient reward for a named, predicted Tox21 assay endpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
from rdkit import DataStructs

from src.data.molgan_graphs import reconstruct_graphs
from src.models.molgan import TrackedMolGANModel
from src.models.tox21_endpoint import Tox21EndpointPredictor

_REWARD_COMPONENTS = (
    "activity",
    "diversity",
    "support",
    "uncertainty",
)


def compute_tox21_guided_rewards(
    activity_probabilities: Sequence[float],
    uncertainty: Sequence[float],
    support_similarity: Sequence[float],
    diversity: Sequence[float],
    valid_mask: Sequence[bool],
    weights: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build bounded per-graph REINFORCE rewards from normalized components.

    Each component is in [0, 1]. Activity, diversity, and support are reward
    scores; uncertainty is a penalty. Invalid molecules receive a fixed -1
    validity-gate reward and are not rescued by other components.
    """
    arrays = {
        "activity_probability": np.asarray(activity_probabilities, dtype=float),
        "uncertainty": np.asarray(uncertainty, dtype=float),
        "support_similarity": np.asarray(support_similarity, dtype=float),
        "diversity": np.asarray(diversity, dtype=float),
    }
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1 or any(values.shape != valid.shape for values in arrays.values()):
        raise ValueError("Les composantes de récompense doivent avoir la même forme 1D.")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Les composantes de récompense doivent être finies.")
    if any(np.any((values < 0.0) | (values > 1.0)) for values in arrays.values()):
        raise ValueError("Les composantes normalisées doivent être dans [0, 1].")
    if set(weights) != set(_REWARD_COMPONENTS):
        raise ValueError(
            "Les poids doivent définir exactement activity, diversity, support "
            "et uncertainty."
        )
    normalized_weights = {name: float(weights[name]) for name in _REWARD_COMPONENTS}
    if any(not np.isfinite(value) or value < 0.0 for value in normalized_weights.values()):
        raise ValueError("Les poids de récompense doivent être finis et positifs ou nuls.")
    total_weight = sum(normalized_weights.values())
    if total_weight <= 0.0:
        raise ValueError("Au moins un poids de récompense doit être positif.")

    components = {
        "activity_score": 1.0 - arrays["activity_probability"],
        "diversity_score": arrays["diversity"],
        "support_score": arrays["support_similarity"],
        "uncertainty_penalty": arrays["uncertainty"],
        "valid_mask": valid,
    }
    reward = (
        normalized_weights["activity"] * components["activity_score"]
        + normalized_weights["diversity"] * components["diversity_score"]
        + normalized_weights["support"] * components["support_score"]
        - normalized_weights["uncertainty"] * components["uncertainty_penalty"]
    ) / total_weight
    reward = np.asarray(reward, dtype=np.float32)
    reward[~valid] = -1.0
    components["total_reward"] = reward
    return reward, components


def _batch_diversity(
    fingerprints: Sequence[Any],
    valid_mask: np.ndarray,
) -> np.ndarray:
    scores = np.zeros((len(valid_mask),), dtype=float)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 2:
        return scores
    for index in valid_indices:
        others = [
            fingerprints[other]
            for other in valid_indices
            if other != index
        ]
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index],
            others,
        )
        scores[index] = 1.0 - float(np.mean(similarities))
    return scores


def _sample_graphs_and_log_probabilities(
    model: TrackedMolGANModel,
    batch_size: int,
) -> tuple[list[GraphMatrix], torch.Tensor]:
    generator = model.generators[0]
    generator.train()
    noise = torch.as_tensor(
        model.get_noise_batch(batch_size),
        dtype=torch.float32,
        device=model.device,
    )
    edge_probabilities, node_probabilities = generator(noise, training=True)
    if not torch.isfinite(edge_probabilities).all() or not torch.isfinite(
        node_probabilities
    ).all():
        raise FloatingPointError("MolGAN a produit des probabilités non finies.")

    upper_i, upper_j = torch.triu_indices(
        model.vertices,
        model.vertices,
        offset=1,
        device=model.device,
    )
    edge_distribution = torch.distributions.Categorical(probs=edge_probabilities)
    node_distribution = torch.distributions.Categorical(probs=node_probabilities)
    edge_choices = edge_distribution.sample()
    node_choices = node_distribution.sample()
    edge_log_probability = torch.distributions.Categorical(
        probs=edge_probabilities[:, upper_i, upper_j, :]
    ).log_prob(edge_choices[:, upper_i, upper_j]).sum(dim=1)
    node_log_probability = node_distribution.log_prob(node_choices).sum(dim=1)
    log_probabilities = edge_log_probability + node_log_probability
    if not torch.isfinite(log_probabilities).all():
        raise FloatingPointError("La log-probabilité MolGAN n'est pas finie.")

    upper_rows = upper_i.detach().cpu().numpy()
    upper_columns = upper_j.detach().cpu().numpy()
    sampled_edges = edge_choices.detach().cpu().numpy()
    sampled_nodes = node_choices.detach().cpu().numpy()
    graphs = []
    for index in range(batch_size):
        adjacency = np.zeros((model.vertices, model.vertices), dtype=np.int32)
        upper_edge_choices = sampled_edges[index, upper_rows, upper_columns]
        adjacency[upper_rows, upper_columns] = upper_edge_choices
        adjacency[upper_columns, upper_rows] = upper_edge_choices
        graphs.append(GraphMatrix(adjacency, sampled_nodes[index].astype(np.int32)))
    return graphs, log_probabilities


def reinforce_tox21_generator(
    model: TrackedMolGANModel,
    featurizer: Any,
    predictor: Tox21EndpointPredictor,
    reference_fingerprints: Sequence[Any],
    *,
    weights: Mapping[str, float],
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> list[dict[str, float | int | str]]:
    """Update MolGAN with a score-function estimator on sampled discrete graphs.

    DeepChem's native differentiable generator distributions are sampled
    categorically here; RDKit and predictor rewards are non-differentiable, so
    REINFORCE multiplies each sampled graph's log probability by its detached
    centered reward. The WGAN discriminator is not updated in this phase.
    """
    if steps < 0:
        raise ValueError("steps doit être positif ou nul.")
    if batch_size < 1:
        raise ValueError("batch_size doit être strictement positif.")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate doit être fini et strictement positif.")
    if not reference_fingerprints:
        raise ValueError("Les empreintes de référence d'entraînement sont requises.")
    if steps == 0:
        return []

    generator = model.generators[0]
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    moving_baseline = 0.0
    logs: list[dict[str, float | int | str]] = []
    for step in range(steps):
        graphs, log_probabilities = _sample_graphs_and_log_probabilities(
            model,
            batch_size,
        )
        reconstruction = reconstruct_graphs(graphs, featurizer)
        molecules = reconstruction.molecules
        valid_mask = np.asarray(
            [molecule is not None for molecule in molecules],
            dtype=bool,
        )
        activity = np.zeros((batch_size,), dtype=float)
        uncertainty = np.zeros((batch_size,), dtype=float)
        support = np.zeros((batch_size,), dtype=float)
        fingerprints: list[Any | None] = [None] * batch_size
        valid_indices = np.flatnonzero(valid_mask)
        if len(valid_indices):
            valid_molecules = [molecules[index] for index in valid_indices]
            probabilities, uncertainty_estimates = (
                predictor.predict_activity_with_uncertainty(valid_molecules)
            )
            activity[valid_indices] = probabilities
            uncertainty[valid_indices] = uncertainty_estimates
            for index, molecule in zip(valid_indices, valid_molecules):
                fingerprint = predictor.fingerprint_generator.GetFingerprint(molecule)
                fingerprints[index] = fingerprint
                support[index] = max(
                    DataStructs.BulkTanimotoSimilarity(
                        fingerprint,
                        reference_fingerprints,
                    )
                )
        diversity = _batch_diversity(fingerprints, valid_mask)
        rewards, components = compute_tox21_guided_rewards(
            activity,
            uncertainty,
            support,
            diversity,
            valid_mask,
            weights,
        )
        reward_tensor = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=model.device,
        )
        update_applied = bool(np.ptp(rewards) > 1e-8)
        if update_applied:
            advantages = reward_tensor - moving_baseline
            loss = -(log_probabilities * advantages.detach()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("La perte policy-gradient n'est pas finie.")
            optimizer.zero_grad()
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in generator.parameters()
            ):
                raise FloatingPointError("Le gradient du générateur n'est pas fini.")
            optimizer.step()
            loss_value = float(loss.detach().cpu())
        else:
            loss_value = 0.0
        moving_baseline = 0.9 * moving_baseline + 0.1 * float(reward_tensor.mean())

        valid_count = int(valid_mask.sum())
        record: dict[str, float | int | str] = {
            "step": step + 1,
            "valid_graph_count": valid_count,
            "validity_rate": valid_count / batch_size,
            "policy_update_applied": int(update_applied),
            "generator_policy_loss": loss_value,
            "reward_mean": float(rewards.mean()),
        }
        for key, values in components.items():
            if key == "valid_mask" or key == "total_reward":
                continue
            record[f"{key}_mean_valid"] = (
                float(np.mean(values[valid_mask])) if valid_count else 0.0
            )
        record["reward_mean_valid"] = (
            float(np.mean(rewards[valid_mask])) if valid_count else 0.0
        )
        logs.append(record)
    return logs
