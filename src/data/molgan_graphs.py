"""MolGAN graph preparation and chemically validated RDKit reconstruction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from deepchem.feat import MolGanFeaturizer
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
from rdkit import Chem


@dataclass(frozen=True)
class MoleculeRejection:
    input_smiles: str
    reason: str
    detail: str


@dataclass
class MolGANPreparation:
    featurizer: MolGanFeaturizer
    input_count: int
    graphs: list[GraphMatrix]
    canonical_smiles: list[str]
    rejections: list[MoleculeRejection]
    max_molecules: int | None

    def summary(self) -> dict:
        rejected_by_reason = Counter(item.reason for item in self.rejections)
        return {
            "input_count": self.input_count,
            "max_molecules": self.max_molecules,
            "featurized_count": len(self.graphs),
            "rejected_count": len(self.rejections),
            "featurization_rate": (
                len(self.graphs) / self.input_count if self.input_count else 0.0
            ),
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        }


@dataclass(frozen=True)
class GraphFailure:
    graph_index: int
    stage: str
    reason: str
    detail: str


@dataclass
class GraphReconstruction:
    molecules: list[Chem.Mol | None]
    canonical_smiles: list[str | None]
    failures: list[GraphFailure]
    tensor_valid_count: int
    conversion_count: int
    sanitized_count: int
    ignored_padding_edge_count: int

    @property
    def input_count(self) -> int:
        return len(self.molecules)

    def summary(self) -> dict:
        return {
            "input_graph_count": self.input_count,
            "tensor_valid_count": self.tensor_valid_count,
            "rdkit_conversion_count": self.conversion_count,
            "sanitized_molecule_count": self.sanitized_count,
            "ignored_padding_edge_count": self.ignored_padding_edge_count,
            "tensor_valid_rate": (
                self.tensor_valid_count / self.input_count if self.input_count else 0.0
            ),
            "rdkit_conversion_rate": (
                self.conversion_count / self.input_count if self.input_count else 0.0
            ),
            "sanitization_rate_after_conversion": (
                self.sanitized_count / self.conversion_count
                if self.conversion_count
                else 0.0
            ),
            "overall_survival_rate": (
                self.sanitized_count / self.input_count if self.input_count else 0.0
            ),
            "failures": [
                {
                    "graph_index": failure.graph_index,
                    "stage": failure.stage,
                    "reason": failure.reason,
                    "detail": failure.detail,
                }
                for failure in self.failures
            ],
        }


class GraphConversionError(ValueError):
    def __init__(self, stage: str, reason: str, detail: str):
        super().__init__(detail)
        self.stage = stage
        self.reason = reason
        self.detail = detail


def create_molgan_featurizer(max_atom_count: int = 9) -> MolGanFeaturizer:
    if max_atom_count < 1:
        raise ValueError("max_atom_count doit être strictement positif.")
    return MolGanFeaturizer(max_atom_count=max_atom_count)


def featurize_smiles(
    smiles: Iterable[str],
    max_atom_count: int = 9,
    max_molecules: int | None = None,
) -> MolGANPreparation:
    if max_molecules is not None and max_molecules < 1:
        raise ValueError("max_molecules doit être strictement positif ou None.")
    featurizer = create_molgan_featurizer(max_atom_count)
    allowed_atoms = set(featurizer.atom_labels) - {0}
    allowed_bonds = set(featurizer.bond_labels) - {Chem.BondType.ZERO}
    graphs: list[GraphMatrix] = []
    canonical_smiles: list[str] = []
    rejections: list[MoleculeRejection] = []
    input_count = 0

    for input_smiles in smiles:
        input_count += 1
        if not isinstance(input_smiles, str) or not input_smiles.strip():
            rejections.append(
                MoleculeRejection(str(input_smiles), "invalid_smiles", "SMILES vide.")
            )
            continue

        molecule = Chem.MolFromSmiles(input_smiles, sanitize=False)
        if molecule is None:
            rejections.append(
                MoleculeRejection(
                    input_smiles, "invalid_smiles", "RDKit n'a pas pu parser le SMILES."
                )
            )
            continue
        try:
            Chem.SanitizeMol(molecule)
        except Exception as error:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "sanitization_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue

        atom_count = molecule.GetNumAtoms()
        if atom_count > featurizer.max_atom_count:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "atom_limit_exceeded",
                    f"{atom_count} atomes; maximum {featurizer.max_atom_count}.",
                )
            )
            continue

        unsupported_atoms = sorted(
            {
                atom.GetSymbol()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() not in allowed_atoms
            }
        )
        if unsupported_atoms:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "unsupported_atom",
                    f"Éléments non pris en charge : {', '.join(unsupported_atoms)}.",
                )
            )
            continue

        if any(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()):
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "formal_charge_not_represented",
                    "MolGanFeaturizer ne conserve pas les charges formelles.",
                )
            )
            continue
        if any(atom.GetIsotope() != 0 for atom in molecule.GetAtoms()):
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "isotope_not_represented",
                    "MolGanFeaturizer ne conserve pas les isotopes.",
                )
            )
            continue
        if any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            for atom in molecule.GetAtoms()
        ) or any(
            bond.GetStereo() != Chem.BondStereo.STEREONONE
            or bond.GetBondDir() != Chem.BondDir.NONE
            for bond in molecule.GetBonds()
        ):
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "stereochemistry_not_represented",
                    "MolGanFeaturizer ne conserve pas la stéréochimie.",
                )
            )
            continue

        unsupported_bonds = sorted(
            {
                str(bond.GetBondType())
                for bond in molecule.GetBonds()
                if bond.GetBondType() not in allowed_bonds
            }
        )
        if unsupported_bonds:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "unsupported_bond",
                    f"Types de liaison non pris en charge : {', '.join(unsupported_bonds)}.",
                )
            )
            continue

        if len(Chem.GetMolFrags(molecule)) != 1:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "disconnected_molecule",
                    "MolGAN attend un graphe représentant une seule molécule connectée.",
                )
            )
            continue
        if atom_count == 1 or any(atom.GetDegree() == 0 for atom in molecule.GetAtoms()):
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "isolated_atom",
                    "Le featurizer MolGAN requiert que chaque atome ait une liaison.",
                )
            )
            continue

        canonical = Chem.MolToSmiles(molecule, canonical=True)
        try:
            graph = featurizer.featurize([Chem.Mol(molecule)])[0]
        except Exception as error:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "featurization_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue

        if not isinstance(graph, GraphMatrix):
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "featurization_failed",
                    "DeepChem n'a pas produit de GraphMatrix.",
                )
            )
            continue
        if graph.adjacency_matrix.shape != (
            featurizer.max_atom_count,
            featurizer.max_atom_count,
        ) or graph.node_features.shape != (featurizer.max_atom_count,):
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "unexpected_graph_shape",
                    "GraphMatrix ne respecte pas les dimensions MolGAN configurées.",
                )
            )
            continue

        try:
            reconstructed_smiles = molecule_to_canonical_smiles(
                graph_to_molecule(graph, featurizer)
            )
        except Exception as error:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "round_trip_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        if reconstructed_smiles != canonical:
            rejections.append(
                MoleculeRejection(
                    input_smiles,
                    "structure_not_preserved",
                    f"SMILES canonique reconstruit : {reconstructed_smiles}.",
                )
            )
            continue

        graphs.append(graph)
        canonical_smiles.append(canonical)
        if max_molecules is not None and len(graphs) >= max_molecules:
            break

    return MolGANPreparation(
        featurizer=featurizer,
        input_count=input_count,
        graphs=graphs,
        canonical_smiles=canonical_smiles,
        rejections=rejections,
        max_molecules=max_molecules,
    )


def graph_matrices_from_tensors(
    adjacency: np.ndarray,
    nodes: np.ndarray,
    featurizer: MolGanFeaturizer,
) -> list[GraphMatrix]:
    adjacency = np.asarray(adjacency)
    nodes = np.asarray(nodes)
    vertices = featurizer.max_atom_count
    edge_categories = len(featurizer.bond_labels)
    atom_categories = len(featurizer.atom_labels)

    if adjacency.ndim == 2 and nodes.ndim == 1:
        adjacency = adjacency[np.newaxis, ...]
        nodes = nodes[np.newaxis, ...]
    elif adjacency.ndim == 3 and nodes.ndim == 2:
        pass
    elif adjacency.ndim == 4 and nodes.ndim == 3:
        adjacency = _decode_one_hot(adjacency, edge_categories, "adjacency")
        nodes = _decode_one_hot(nodes, atom_categories, "nodes")
    else:
        raise ValueError(
            "Tenseurs attendus: indices (B,V,V)/(B,V) ou one-hot "
            "(B,V,V,E)/(B,V,N)."
        )

    if adjacency.shape[1:] != (vertices, vertices):
        raise ValueError(
            f"adjacency doit avoir la forme (B,{vertices},{vertices})."
        )
    if nodes.shape[1:] != (vertices,):
        raise ValueError(f"nodes doit avoir la forme (B,{vertices}).")
    if adjacency.shape[0] != nodes.shape[0]:
        raise ValueError(
            "Les tenseurs adjacency et nodes doivent avoir le même batch."
        )
    if not _is_real_numeric_array(adjacency) or not _is_real_numeric_array(nodes):
        raise ValueError("Les tenseurs adjacency et nodes doivent être numériques.")
    if not np.all(np.isfinite(adjacency)) or not np.all(
        np.equal(adjacency, np.floor(adjacency))
    ):
        raise ValueError("adjacency doit contenir des indices de liaison entiers.")
    if not np.all(np.isfinite(nodes)) or not np.all(
        np.equal(nodes, np.floor(nodes))
    ):
        raise ValueError("nodes doit contenir des indices d'atome entiers.")
    if np.any((adjacency < 0) | (adjacency >= edge_categories)):
        raise ValueError("adjacency contient un indice de liaison hors limites.")
    if np.any((nodes < 0) | (nodes >= atom_categories)):
        raise ValueError("nodes contient un indice d'atome hors limites.")

    return [
        GraphMatrix(np.asarray(edge_matrix), np.asarray(node_vector))
        for edge_matrix, node_vector in zip(adjacency, nodes)
    ]


def _decode_one_hot(tensor: np.ndarray, categories: int, name: str) -> np.ndarray:
    if tensor.shape[-1] != categories:
        raise ValueError(f"{name} doit avoir {categories} catégories one-hot.")
    if not _is_real_numeric_array(tensor):
        raise ValueError(f"{name} doit contenir des valeurs numériques.")
    if not np.all(np.isfinite(tensor)) or not np.all(
        (tensor == 0) | (tensor == 1)
    ):
        raise ValueError(f"{name} doit être un tenseur one-hot binaire.")
    if not np.all(tensor.sum(axis=-1) == 1):
        raise ValueError(f"{name} doit contenir exactement une catégorie active.")
    return tensor.argmax(axis=-1)


def _is_real_numeric_array(values: np.ndarray) -> bool:
    return np.issubdtype(values.dtype, np.integer) or np.issubdtype(
        values.dtype, np.floating
    )


def graph_to_molecule(
    graph: GraphMatrix,
    featurizer: MolGanFeaturizer,
) -> Chem.Mol:
    adjacency, nodes = _validate_graph(graph, featurizer)
    active_indices = np.flatnonzero(nodes != 0)
    if not len(active_indices):
        raise GraphConversionError("validation", "empty_graph", "Aucun atome actif.")

    active_adjacency = adjacency[np.ix_(active_indices, active_indices)]
    active_nodes = nodes[active_indices]
    if len(active_indices) == 1 or np.any(
        np.count_nonzero(active_adjacency, axis=1) == 0
    ):
        raise GraphConversionError(
            "validation",
            "isolated_atom",
            "Chaque atome actif doit être relié à un autre atome.",
        )

    reduced_graph = GraphMatrix(active_adjacency, active_nodes)
    try:
        molecule = featurizer._defeaturize(
            reduced_graph,
            sanitize=False,
            cleanup=False,
        )
    except Exception as error:
        raise GraphConversionError(
            "conversion",
            "rdkit_conversion_failed",
            f"{type(error).__name__}: {error}",
        ) from error
    if molecule is None:
        raise GraphConversionError(
            "conversion",
            "rdkit_conversion_failed",
            "DeepChem n'a pas créé de molécule RDKit.",
        )
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise GraphConversionError(
            "validation",
            "disconnected_graph",
            "Le graphe contient plusieurs composantes moléculaires.",
        )
    return molecule


def _validate_graph(
    graph: GraphMatrix,
    featurizer: MolGanFeaturizer,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(graph, GraphMatrix):
        raise GraphConversionError(
            "validation",
            "invalid_graph_type",
            f"GraphMatrix attendu, reçu {type(graph).__name__}.",
        )
    adjacency = np.asarray(graph.adjacency_matrix)
    nodes = np.asarray(graph.node_features)
    vertices = featurizer.max_atom_count

    if adjacency.shape != (vertices, vertices) or nodes.shape != (vertices,):
        raise GraphConversionError(
            "validation",
            "invalid_tensor_shape",
            "Dimensions attendues: "
            f"adjacency=({vertices},{vertices}), nodes=({vertices},).",
        )
    if not _is_real_numeric_array(adjacency) or not _is_real_numeric_array(nodes):
        raise GraphConversionError(
            "validation",
            "invalid_tensor_dtype",
            "Les tenseurs adjacency et nodes doivent être numériques.",
        )
    if not np.all(np.isfinite(adjacency)) or not np.all(
        np.equal(adjacency, np.floor(adjacency))
    ):
        raise GraphConversionError(
            "validation", "invalid_bond_indices", "Les liaisons doivent être des entiers."
        )
    if not np.all(np.isfinite(nodes)) or not np.all(
        np.equal(nodes, np.floor(nodes))
    ):
        raise GraphConversionError(
            "validation", "invalid_atom_indices", "Les atomes doivent être des entiers."
        )
    if np.any((adjacency < 0) | (adjacency >= len(featurizer.bond_labels))):
        raise GraphConversionError(
            "validation", "invalid_bond_indices", "Indice de liaison hors limites."
        )
    if np.any((nodes < 0) | (nodes >= len(featurizer.atom_labels))):
        raise GraphConversionError(
            "validation", "invalid_atom_indices", "Indice d'atome hors limites."
        )
    if not np.array_equal(adjacency, adjacency.T):
        raise GraphConversionError(
            "validation",
            "asymmetric_adjacency",
            "La matrice d'adjacence n'est pas symétrique.",
        )
    if np.any(np.diag(adjacency) != 0):
        raise GraphConversionError(
            "validation", "self_bond", "Les boucles de liaison sont interdites."
        )
    return adjacency.astype(np.int64, copy=False), nodes.astype(np.int64, copy=False)


def molecule_to_canonical_smiles(molecule: Chem.Mol) -> str:
    if molecule is None:
        raise ValueError("Une molécule RDKit non nulle est requise.")
    sanitized = Chem.Mol(molecule)
    Chem.SanitizeMol(sanitized)
    smiles = Chem.MolToSmiles(sanitized, canonical=True)
    if not smiles:
        raise ValueError("RDKit n'a pas produit de SMILES canonique.")
    return smiles


def reconstruct_graphs(
    graphs: Iterable[GraphMatrix],
    featurizer: MolGanFeaturizer,
) -> GraphReconstruction:
    graph_list = list(graphs)
    molecules: list[Chem.Mol | None] = [None] * len(graph_list)
    canonical_smiles: list[str | None] = [None] * len(graph_list)
    failures: list[GraphFailure] = []
    tensor_valid_count = 0
    conversion_count = 0
    sanitized_count = 0
    ignored_padding_edge_count = 0

    for index, graph in enumerate(graph_list):
        try:
            adjacency, nodes = _validate_graph(graph, featurizer)
        except GraphConversionError as error:
            failures.append(
                GraphFailure(index, error.stage, error.reason, error.detail)
            )
            continue
        tensor_valid_count += 1
        padding = nodes == 0
        padding_edges = np.triu(adjacency != 0, k=1) & (
            padding[:, np.newaxis] | padding[np.newaxis, :]
        )
        ignored_padding_edge_count += int(np.count_nonzero(padding_edges))

        try:
            molecule = graph_to_molecule(graph, featurizer)
            conversion_count += 1
        except GraphConversionError as error:
            failures.append(
                GraphFailure(index, error.stage, error.reason, error.detail)
            )
            continue

        try:
            Chem.SanitizeMol(molecule)
        except Exception as error:
            failures.append(
                GraphFailure(
                    index,
                    "sanitization",
                    "rdkit_sanitization_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        sanitized_count += 1

        try:
            canonical_smiles[index] = molecule_to_canonical_smiles(molecule)
        except Exception as error:
            failures.append(
                GraphFailure(
                    index,
                    "canonicalization",
                    "canonical_smiles_failed",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        molecules[index] = molecule

    return GraphReconstruction(
        molecules=molecules,
        canonical_smiles=canonical_smiles,
        failures=failures,
        tensor_valid_count=tensor_valid_count,
        conversion_count=conversion_count,
        sanitized_count=sanitized_count,
        ignored_padding_edge_count=ignored_padding_edge_count,
    )
