import numpy as np
import pytest
import torch

from src.data.molgan_graphs import featurize_smiles
from src.models.molgan import (
    _training_batches,
    build_model,
    resolve_device,
    seed_everything,
    train_adversarial,
)


@pytest.fixture
def config(tmp_path):
    return {
        "molgan": {
            "vertices": 9,
            "nodes": 5,
            "edges": 5,
            "embedding_dim": 10,
            "batch_size": 2,
            "learning_rate": 0.001,
        },
        "baseline": {
            "learning_rate_decay_rate": 0.9,
            "learning_rate_decay_steps": 5000,
        },
        "model_dir": tmp_path,
    }


def test_training_batches_have_deepchem_one_hot_shapes(config):
    graphs = featurize_smiles(["CCO", "c1ccccc1"]).graphs
    model = build_model(config, device="cpu")
    batches = list(_training_batches(model, graphs, epochs=1))

    assert len(batches) == 1
    assert batches[0][model.data_inputs[0]].shape == (2, 9, 9, 5)
    assert batches[0][model.data_inputs[1]].shape == (2, 9, 5)
    assert torch.all(batches[0][model.data_inputs[0]].sum(dim=-1) == 1)
    assert torch.all(batches[0][model.data_inputs[1]].sum(dim=-1) == 1)


def test_native_wgan_training_has_finite_losses_and_updates_both_networks(
    config,
    tmp_path,
):
    graphs = featurize_smiles(["CCO", "c1ccccc1"]).graphs
    model = build_model(config, seed=31, device="cpu", model_dir=tmp_path / "model")
    model._ensure_built()
    generator_before = [
        parameter.detach().clone() for parameter in model.generators[0].parameters()
    ]
    discriminator_before = [
        parameter.detach().clone()
        for parameter in model.discriminators[0].parameters()
    ]

    losses = train_adversarial(
        model,
        graphs,
        epochs=1,
        generator_steps=1.0,
        checkpoint_interval=0,
    )

    assert {record["component"] for record in losses} == {
        "generator",
        "discriminator",
    }
    assert all(np.isfinite(record["loss"]) for record in losses)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            generator_before,
            model.generators[0].parameters(),
        )
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.generators[0].parameters()
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            discriminator_before,
            model.discriminators[0].parameters(),
        )
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.discriminators[0].parameters()
    )
    model.save_checkpoint(model_dir=str(tmp_path / "model"))

    restored_model = build_model(
        config,
        seed=99,
        device="cpu",
        model_dir=tmp_path / "model",
    )
    restored_model.restore()
    assert restored_model.model.noise_input.shape == (2, 10)
    for trained_parameter, restored_parameter in zip(
        model.model.parameters(),
        restored_model.model.parameters(),
    ):
        torch.testing.assert_close(trained_parameter, restored_parameter)


def test_model_initialization_is_seed_reproducible(config):
    first = build_model(config, seed=123, device="cpu")
    second = build_model(config, seed=123, device="cpu")
    first._ensure_built()
    second._ensure_built()

    for first_parameter, second_parameter in zip(
        first.model.parameters(),
        second.model.parameters(),
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_graph_preprocessing_is_deterministic():
    smiles = ["CCO", "c1ccccc1", "CCN"]
    first = featurize_smiles(smiles)
    second = featurize_smiles(smiles)

    assert first.canonical_smiles == second.canonical_smiles
    for first_graph, second_graph in zip(first.graphs, second.graphs):
        np.testing.assert_array_equal(
            first_graph.adjacency_matrix,
            second_graph.adjacency_matrix,
        )
        np.testing.assert_array_equal(
            first_graph.node_features,
            second_graph.node_features,
        )


def test_seed_helper_and_device_selection(config):
    seed_everything(44)
    first = torch.rand(4)
    seed_everything(44)
    second = torch.rand(4)

    torch.testing.assert_close(first, second)
    assert resolve_device("cpu").type == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


@pytest.mark.parametrize(
    ("epochs", "generator_steps", "message"),
    [
        (0, 1.0, "époques"),
        (1, 0.0, "generator_steps"),
    ],
)
def test_invalid_training_parameters_are_rejected(
    config,
    epochs,
    generator_steps,
    message,
):
    model = build_model(config, device="cpu")
    graphs = featurize_smiles(["CCO"]).graphs

    with pytest.raises(ValueError, match=message):
        train_adversarial(model, graphs, epochs, generator_steps)
