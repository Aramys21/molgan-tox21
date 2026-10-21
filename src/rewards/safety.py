"""Policy-gradient fine-tuning to reward valid molecules with lower predicted risk."""

from __future__ import annotations

import numpy as np
import torch
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix


def compute_safety_rewards(risks, valid_mask, toxicity_weight: float) -> np.ndarray:
    if toxicity_weight < 0:
        raise ValueError("Le poids de toxicité doit être positif ou nul.")
    risks = np.asarray(risks, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if risks.shape != valid.shape:
        raise ValueError("risks et valid_mask doivent avoir la même dimension.")
    if np.any((risks[valid] < 0) | (risks[valid] > 1)):
        raise ValueError("Les risques de toxicité doivent être dans [0, 1].")
    rewards = np.full(risks.shape, -toxicity_weight, dtype=np.float32)
    rewards[valid] = 1.0 - toxicity_weight * risks[valid]
    return rewards


def reinforce_safe_generation(
    model,
    featurizer,
    toxicity_predictor,
    toxicity_weight: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    if toxicity_weight < 0:
        raise ValueError("Le poids de toxicité doit être positif ou nul.")
    if steps <= 0:
        return
    if batch_size < 1:
        raise ValueError("La taille de batch doit être strictement positive.")

    generator = model.generators[0]
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    baseline = 0.0

    for step in range(steps):
        noise = torch.as_tensor(
            model.get_noise_batch(batch_size),
            dtype=torch.float32,
            device=model.device,
        )
        edge_probabilities, node_probabilities = generator(
            noise,
            training=True,
        )
        edge_choices = torch.distributions.Categorical(
            probs=edge_probabilities
        ).sample()
        node_choices = torch.distributions.Categorical(
            probs=node_probabilities
        ).sample()

        upper_i, upper_j = torch.triu_indices(
            model.vertices,
            model.vertices,
            offset=1,
            device=model.device,
        )
        edge_log_prob = torch.distributions.Categorical(
            probs=edge_probabilities[:, upper_i, upper_j, :]
        ).log_prob(edge_choices[:, upper_i, upper_j]).sum(dim=1)
        node_log_prob = torch.distributions.Categorical(
            probs=node_probabilities
        ).log_prob(node_choices).sum(dim=1)

        graphs = []
        for batch_index in range(batch_size):
            adjacency = np.zeros((model.vertices, model.vertices), dtype=np.int32)
            sampled_edges = edge_choices[
                batch_index, upper_i, upper_j
            ].detach().cpu().numpy()
            i = upper_i.detach().cpu().numpy()
            j = upper_j.detach().cpu().numpy()
            adjacency[i, j] = sampled_edges
            adjacency[j, i] = sampled_edges
            nodes = (
                node_choices[batch_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )
            graphs.append(GraphMatrix(adjacency, nodes))

        molecules = list(featurizer.defeaturize(graphs))
        valid_indices = [
            index for index, molecule in enumerate(molecules) if molecule is not None
        ]
        risks = np.zeros((batch_size,), dtype=float)
        valid_mask = np.zeros((batch_size,), dtype=bool)
        if valid_indices:
            valid_mask[valid_indices] = True
            risks[valid_indices] = toxicity_predictor.predict_risk(
                [molecules[index] for index in valid_indices]
            )

        rewards = compute_safety_rewards(
            risks,
            valid_mask,
            toxicity_weight,
        )
        reward_tensor = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=model.device,
        )
        advantages = reward_tensor - baseline
        loss = -(
            (edge_log_prob + node_log_prob) * advantages.detach()
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        baseline = 0.9 * baseline + 0.1 * float(reward_tensor.mean())
        if (step + 1) % 10 == 0 or step + 1 == steps:
            print(
                f"Récompense sûreté {step + 1}/{steps} : "
                f"{float(reward_tensor.mean()):.3f}"
            )
