import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.data.molgan_graphs import GraphReconstruction, create_molgan_featurizer
from src.models.molgan import build_model
from src.rewards import tox21_guided
from src.rewards.tox21_guided import (
    _batch_diversity,
    _sample_graphs_and_log_probabilities,
    compute_tox21_guided_rewards,
    reinforce_tox21_generator,
)


def _weights(**overrides):
    values = {
        "activity": 0.5,
        "diversity": 0.2,
        "support": 0.2,
        "uncertainty": 0.1,
    }
    values.update(overrides)
    return values


def test_reward_uses_normalized_components_and_invalidity_gate():
    rewards, components = compute_tox21_guided_rewards(
        activity_probabilities=[0.2, 0.7],
        uncertainty=[0.1, 0.9],
        support_similarity=[0.7, 0.2],
        diversity=[0.4, 0.8],
        valid_mask=[True, False],
        weights=_weights(),
    )

    np.testing.assert_allclose(rewards, [0.61, -1.0])
    np.testing.assert_allclose(components["activity_score"], [0.8, 0.3])
    np.testing.assert_array_equal(components["valid_mask"], [True, False])


def test_changing_assay_activity_weight_changes_reward():
    common = {
        "activity_probabilities": [0.1, 0.9],
        "uncertainty": [0.0, 0.0],
        "support_similarity": [0.5, 0.5],
        "diversity": [0.5, 0.5],
        "valid_mask": [True, True],
    }
    activity_focused, _ = compute_tox21_guided_rewards(
        **common,
        weights=_weights(activity=0.8, diversity=0.1, support=0.05, uncertainty=0.05),
    )
    diversity_focused, _ = compute_tox21_guided_rewards(
        **common,
        weights=_weights(activity=0.2, diversity=0.4, support=0.2, uncertainty=0.2),
    )

    assert activity_focused[0] - activity_focused[1] > (
        diversity_focused[0] - diversity_focused[1]
    )


@pytest.mark.parametrize(
    "values,weights,error",
    [
        (
            {
                "activity_probabilities": [0.1, 0.2],
                "uncertainty": [0.0],
                "support_similarity": [0.5, 0.5],
                "diversity": [0.5, 0.5],
                "valid_mask": [True, True],
            },
            _weights(),
            "même forme",
        ),
        (
            {
                "activity_probabilities": [1.1],
                "uncertainty": [0.0],
                "support_similarity": [0.5],
                "diversity": [0.5],
                "valid_mask": [True],
            },
            _weights(),
            r"\[0, 1\]",
        ),
        (
            {
                "activity_probabilities": [0.1],
                "uncertainty": [0.0],
                "support_similarity": [0.5],
                "diversity": [0.5],
                "valid_mask": [True],
            },
            _weights(activity=0, diversity=0, support=0, uncertainty=0),
            "positif",
        ),
    ],
)
def test_reward_rejects_malformed_components_and_weights(values, weights, error):
    with pytest.raises(ValueError, match=error):
        compute_tox21_guided_rewards(**values, weights=weights)


def test_batch_diversity_handles_singletons_and_varied_molecules():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in ["CCO", "c1ccccc1"]
    ]

    singleton = _batch_diversity(fingerprints, np.asarray([True, False]))
    varied = _batch_diversity(fingerprints, np.asarray([True, True]))

    np.testing.assert_array_equal(singleton, [0.0, 0.0])
    assert varied[0] == pytest.approx(
        1.0 - DataStructs.TanimotoSimilarity(fingerprints[0], fingerprints[1])
    )
    np.testing.assert_allclose(varied[0], varied[1])


def test_generator_sampling_produces_symmetric_graphs_and_finite_log_probs():
    model = build_model(
        {
            "molgan": {
                "learning_rate": 0.001,
                "vertices": 9,
                "nodes": 5,
                "edges": 5,
                "embedding_dim": 10,
                "batch_size": 2,
            },
            "baseline": {},
        },
        seed=7,
        device="cpu",
    )

    graphs, log_probabilities = _sample_graphs_and_log_probabilities(model, 2)

    assert len(graphs) == 2
    assert np.array_equal(graphs[0].adjacency_matrix, graphs[0].adjacency_matrix.T)
    assert log_probabilities.shape == (2,)
    assert np.isfinite(log_probabilities.detach().numpy()).all()


def test_policy_gradient_updates_generator_and_logs_reward_components(monkeypatch):
    class FakePredictor:
        fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=128,
        )

        @staticmethod
        def predict_activity_with_uncertainty(molecules):
            return np.asarray([0.2, 0.8]), np.full(len(molecules), 0.1)

    def valid_reconstruction(graphs, featurizer):
        molecules = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("c1ccccc1"),
        ]
        return GraphReconstruction(
            molecules=molecules,
            canonical_smiles=["CCO", "c1ccccc1"],
            failures=[],
            tensor_valid_count=len(graphs),
            conversion_count=len(graphs),
            sanitized_count=len(graphs),
            ignored_padding_edge_count=0,
        )

    monkeypatch.setattr(tox21_guided, "reconstruct_graphs", valid_reconstruction)

    model = build_model(
        {
            "molgan": {
                "learning_rate": 0.001,
                "vertices": 9,
                "nodes": 5,
                "edges": 5,
                "embedding_dim": 10,
                "batch_size": 2,
            },
            "baseline": {},
        },
        seed=13,
        device="cpu",
    )
    before = [
        parameter.detach().clone()
        for parameter in model.generators[0].parameters()
    ]
    training_fingerprint = FakePredictor.fingerprint_generator.GetFingerprint(
        Chem.MolFromSmiles("C")
    )

    logs = reinforce_tox21_generator(
        model,
        create_molgan_featurizer(),
        FakePredictor(),
        [training_fingerprint],
        weights=_weights(),
        steps=1,
        batch_size=2,
        learning_rate=1e-4,
    )

    after = list(model.generators[0].parameters())
    assert any(
        not np.array_equal(old.detach().numpy(), new.detach().numpy())
        for old, new in zip(before, after)
    )
    assert len(logs) == 1
    assert logs[0]["policy_update_applied"] == 1
    assert "activity_score_mean_valid" in logs[0]
    assert "uncertainty_penalty_mean_valid" in logs[0]


def test_constant_invalid_rewards_do_not_apply_a_noisy_policy_update():
    class FakePredictor:
        fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=128,
        )

        @staticmethod
        def predict_activity_with_uncertainty(molecules):
            raise AssertionError("Invalid graphs must not be sent to the predictor.")

    model = build_model(
        {
            "molgan": {
                "learning_rate": 0.001,
                "vertices": 9,
                "nodes": 5,
                "edges": 5,
                "embedding_dim": 10,
                "batch_size": 2,
            },
            "baseline": {},
        },
        seed=13,
        device="cpu",
    )
    before = [
        parameter.detach().clone()
        for parameter in model.generators[0].parameters()
    ]

    logs = reinforce_tox21_generator(
        model,
        create_molgan_featurizer(),
        FakePredictor(),
        [FakePredictor.fingerprint_generator.GetFingerprint(Chem.MolFromSmiles("C"))],
        weights=_weights(),
        steps=1,
        batch_size=2,
        learning_rate=1e-4,
    )

    assert logs[0]["valid_graph_count"] == 0
    assert logs[0]["policy_update_applied"] == 0
    assert all(
        np.array_equal(old.detach().numpy(), new.detach().numpy())
        for old, new in zip(before, model.generators[0].parameters())
    )
