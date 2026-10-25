import numpy as np
import pytest
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
from rdkit import Chem

from src.data.molgan_graphs import (
    featurize_smiles,
    graph_matrices_from_tensors,
    graph_to_molecule,
    molecule_to_canonical_smiles,
    reconstruct_graphs,
)
from src.visualization.molecules import save_molecule_grid


def test_featurizer_graph_dimensions_labels_and_valid_round_trip():
    preparation = featurize_smiles(["CCO", "c1ccccc1"])

    assert len(preparation.graphs) == 2
    assert preparation.featurizer.max_atom_count == 9
    assert preparation.featurizer.atom_labels == [0, 6, 7, 8, 9]
    assert len(preparation.featurizer.bond_labels) == 5
    assert preparation.graphs[0].adjacency_matrix.shape == (9, 9)
    assert preparation.graphs[0].node_features.shape == (9,)

    reconstruction = reconstruct_graphs(preparation.graphs, preparation.featurizer)

    assert reconstruction.summary()["tensor_valid_rate"] == 1.0
    assert reconstruction.summary()["rdkit_conversion_rate"] == 1.0
    assert reconstruction.summary()["sanitization_rate_after_conversion"] == 1.0
    assert reconstruction.canonical_smiles == preparation.canonical_smiles
    assert [molecule.GetNumAtoms() for molecule in reconstruction.molecules] == [3, 6]


def test_preparation_reports_invalid_and_unsupported_inputs():
    preparation = featurize_smiles(
        [
            "C1CC",
            "CCCl",
            "CCCCCCCCCC",
            "[NH4+]",
            "C[C@H](O)F",
            "F/C=C/F",
            "C[CH2]",
            "C",
        ]
    )

    assert preparation.summary()["input_count"] == 8
    assert preparation.summary()["featurized_count"] == 0
    assert {item.reason for item in preparation.rejections} == {
        "invalid_smiles",
        "unsupported_atom",
        "atom_limit_exceeded",
        "formal_charge_not_represented",
        "stereochemistry_not_represented",
        "structure_not_preserved",
        "isolated_atom",
    }


def test_preparation_limit_counts_successful_graphs_not_input_rows():
    preparation = featurize_smiles(
        ["CCCl", "CCO", "CCN"],
        max_molecules=1,
    )

    assert preparation.input_count == 2
    assert len(preparation.graphs) == 1
    assert preparation.rejections[0].reason == "unsupported_atom"
    assert preparation.summary()["max_molecules"] == 1


def test_input_sanitization_failure_is_reported():
    preparation = featurize_smiles(["[CH5]"])

    assert preparation.graphs == []
    assert preparation.rejections[0].reason == "sanitization_failed"


def test_aromatic_input_is_kekulized_and_reconstructed():
    preparation = featurize_smiles(["c1ccccc1"])
    graph = preparation.graphs[0]
    bond_indices = set(np.unique(graph.adjacency_matrix)) - {0}

    assert preparation.featurizer.kekulize is True
    assert bond_indices == {1, 2}
    molecule = graph_to_molecule(graph, preparation.featurizer)
    assert molecule_to_canonical_smiles(molecule) == "c1ccccc1"


def test_graph_tensors_support_discrete_and_one_hot_batch_shapes():
    preparation = featurize_smiles(["CCO"])
    graph = preparation.graphs[0]
    adjacency = graph.adjacency_matrix[np.newaxis, ...]
    nodes = graph.node_features[np.newaxis, ...]

    decoded_indices = graph_matrices_from_tensors(
        adjacency,
        nodes,
        preparation.featurizer,
    )
    one_hot_adjacency = np.eye(5, dtype=np.float32)[adjacency.astype(np.int64)]
    one_hot_nodes = np.eye(5, dtype=np.float32)[nodes]
    decoded_one_hot = graph_matrices_from_tensors(
        one_hot_adjacency,
        one_hot_nodes,
        preparation.featurizer,
    )

    assert one_hot_adjacency.shape == (1, 9, 9, 5)
    assert one_hot_nodes.shape == (1, 9, 5)
    np.testing.assert_array_equal(
        decoded_indices[0].adjacency_matrix,
        decoded_one_hot[0].adjacency_matrix,
    )
    np.testing.assert_array_equal(
        decoded_indices[0].node_features,
        decoded_one_hot[0].node_features,
    )
    assert reconstruct_graphs(decoded_one_hot, preparation.featurizer).sanitized_count == 1


def test_invalid_graph_dimensions_are_rejected():
    preparation = featurize_smiles(["CCO"])
    graph = preparation.graphs[0]
    invalid_graph = GraphMatrix(
        graph.adjacency_matrix[:-1, :-1],
        graph.node_features[:-1],
    )

    with pytest.raises(ValueError, match="Dimensions attendues"):
        graph_to_molecule(invalid_graph, preparation.featurizer)


def test_rdkit_sanitization_failure_is_counted_and_not_returned_as_valid():
    preparation = featurize_smiles(["CCO"])
    adjacency = np.zeros((9, 9), dtype=np.int32)
    adjacency[0, 1:6] = 1
    adjacency[1:6, 0] = 1
    nodes = np.zeros((9,), dtype=np.int32)
    nodes[:6] = 1

    reconstruction = reconstruct_graphs(
        [GraphMatrix(adjacency, nodes)],
        preparation.featurizer,
    )

    assert reconstruction.tensor_valid_count == 1
    assert reconstruction.conversion_count == 1
    assert reconstruction.sanitized_count == 0
    assert reconstruction.molecules == [None]
    assert reconstruction.failures[0].reason == "rdkit_sanitization_failed"


def test_padding_edges_are_counted_and_removed_before_conversion():
    preparation = featurize_smiles(["CCO"])
    graph = preparation.graphs[0]
    adjacency = graph.adjacency_matrix.copy()
    adjacency[0, 8] = adjacency[8, 0] = 1
    graph_with_padding_edge = GraphMatrix(adjacency, graph.node_features.copy())

    reconstruction = reconstruct_graphs(
        [graph_with_padding_edge],
        preparation.featurizer,
    )

    assert reconstruction.ignored_padding_edge_count == 1
    assert reconstruction.canonical_smiles == ["CCO"]
    assert reconstruction.sanitized_count == 1


def test_visualization_saves_a_grid_of_sanitized_molecules(tmp_path):
    output = save_molecule_grid(
        [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")],
        tmp_path / "molecules.png",
        legends=["ethanol", "benzene"],
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_visualization_rejects_missing_molecules_and_mismatched_legends(tmp_path):
    with pytest.raises(ValueError, match="absente"):
        save_molecule_grid([None], tmp_path / "missing.png")
    with pytest.raises(ValueError, match="légendes"):
        save_molecule_grid(
            [Chem.MolFromSmiles("CCO")],
            tmp_path / "legend.png",
            legends=[],
        )
