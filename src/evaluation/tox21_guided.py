"""Metrics for comparing baseline and Tox21-guided MolGAN samples."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from rdkit import Chem, DataStructs

from src.data.molgan_graphs import GraphReconstruction

if TYPE_CHECKING:
    from src.models.tox21_endpoint import Tox21EndpointPredictor


def evaluate_tox21_guided_samples(
    reconstruction: GraphReconstruction,
    predictor: Tox21EndpointPredictor,
    training_smiles: set[str],
    training_fingerprints: Sequence[Any],
    *,
    out_of_domain_similarity_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate valid samples; support and diversity use Morgan/Tanimoto."""
    if not 0.0 <= out_of_domain_similarity_threshold <= 1.0:
        raise ValueError("Le seuil de similarité doit être dans [0, 1].")
    valid_indices = [
        index
        for index, molecule in enumerate(reconstruction.molecules)
        if molecule is not None and reconstruction.canonical_smiles[index] is not None
    ]
    unique_indices: list[int] = []
    seen_smiles: set[str] = set()
    for index in valid_indices:
        smiles = reconstruction.canonical_smiles[index]
        if smiles not in seen_smiles:
            seen_smiles.add(str(smiles))
            unique_indices.append(index)

    valid_molecules = [
        reconstruction.molecules[index] for index in valid_indices
    ]
    valid_probabilities, valid_uncertainty = (
        predictor.predict_activity_with_uncertainty(valid_molecules)
    )
    reference_rows: list[dict[str, Any]] = []
    for offset, graph_index in enumerate(valid_indices):
        molecule = valid_molecules[offset]
        fingerprint = predictor.fingerprint_generator.GetFingerprint(molecule)
        reference_rows.append(
            {
                "graph_index": graph_index,
                "smiles": reconstruction.canonical_smiles[graph_index],
                "predicted_activity_probability": float(valid_probabilities[offset]),
                "forest_tree_disagreement": float(valid_uncertainty[offset]),
                "maximum_training_similarity": None,
                "out_of_domain": False,
            }
        )
        reference_rows[-1]["_fingerprint"] = fingerprint

    reference_fingerprints = [
        row.pop("_fingerprint") for row in reference_rows
    ]
    if reference_fingerprints:
        support_scores = [
            max(DataStructs.BulkTanimotoSimilarity(fp, training_fingerprints))
            for fp in reference_fingerprints
        ]
        for row, score in zip(reference_rows, support_scores):
            row["maximum_training_similarity"] = float(score)
            row["out_of_domain"] = bool(
                score < out_of_domain_similarity_threshold
            )

    unique_fingerprints = [
        predictor.fingerprint_generator.GetFingerprint(
            reconstruction.molecules[index]
        )
        for index in unique_indices
    ]
    pairwise_distances = [
        1.0 - DataStructs.TanimotoSimilarity(left, right)
        for left, right in combinations(unique_fingerprints, 2)
    ]
    unique_probabilities = [
        row["predicted_activity_probability"]
        for row in reference_rows
        if row["graph_index"] in unique_indices
    ]
    unique_uncertainty = [
        row["forest_tree_disagreement"]
        for row in reference_rows
        if row["graph_index"] in unique_indices
    ]
    unique_support = [
        row["maximum_training_similarity"]
        for row in reference_rows
        if row["graph_index"] in unique_indices
    ]
    positive_count = sum(
        probability >= predictor.decision_threshold
        for probability in unique_probabilities
    )
    unique_count = len(unique_indices)
    valid_count = len(valid_indices)
    novel_count = sum(
        reconstruction.canonical_smiles[index] not in training_smiles
        for index in unique_indices
    )
    out_of_domain_count = sum(
        row["out_of_domain"] for row in reference_rows
    )
    report = {
        "generated_graph_count": reconstruction.input_count,
        "rdkit_valid_count": valid_count,
        "validity_rate": (
            valid_count / reconstruction.input_count
            if reconstruction.input_count
            else 0.0
        ),
        "unique_valid_count": unique_count,
        "uniqueness_rate": unique_count / valid_count if valid_count else 0.0,
        "novel_unique_count": int(novel_count),
        "novelty_rate": novel_count / unique_count if unique_count else 0.0,
        "mean_pairwise_tanimoto_distance_unique": (
            float(np.mean(pairwise_distances)) if pairwise_distances else None
        ),
        "toxicity_evaluation_excluded_valid_count": 0,
        "toxicity_evaluation_excluded_reason": None,
        "unique_sample_activity_probability_mean": (
            float(np.mean(unique_probabilities)) if unique_probabilities else None
        ),
        "unique_sample_activity_probability_std": (
            float(np.std(unique_probabilities)) if unique_probabilities else None
        ),
        "unique_sample_activity_probability_quantiles_2_5_50_97_5": (
            np.quantile(unique_probabilities, [0.025, 0.5, 0.975]).tolist()
            if unique_probabilities
            else None
        ),
        "unique_sample_predicted_active_count": int(positive_count),
        "unique_sample_predicted_active_rate": (
            positive_count / unique_count if unique_count else None
        ),
        "decision_threshold_selected_on_validation": predictor.decision_threshold,
        "mean_forest_tree_disagreement_unique": (
            float(np.mean(unique_uncertainty)) if unique_uncertainty else None
        ),
        "mean_maximum_training_similarity_unique": (
            float(np.mean(unique_support)) if unique_support else None
        ),
        "out_of_domain_similarity_threshold": out_of_domain_similarity_threshold,
        "out_of_domain_valid_count": int(out_of_domain_count),
        "out_of_domain_valid_rate": (
            out_of_domain_count / valid_count if valid_count else 0.0
        ),
        "split_reference": "official Tox21 training split only",
    }
    return report, reference_rows
