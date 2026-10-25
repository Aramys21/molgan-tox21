"""Load and audit the official MoleculeNet Tox21 partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOLNET_CACHE_DIR = PROJECT_ROOT / "data" / "deepchem_cache"
MOLNET_AUDIT_FILE = PROJECT_ROOT / "data" / "audit" / "tox21_molnet_audit.json"

TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
@dataclass(frozen=True)
class Tox21Splits:
    """Official MoleculeNet Tox21 splits and their loader metadata."""

    tasks: tuple[str, ...]
    train: dc.data.Dataset
    valid: dc.data.Dataset
    test: dc.data.Dataset
    featurizer: str
    splitter: str
    transformers: tuple[str, ...]
    seed: int | None = None

    @property
    def datasets(self) -> tuple[dc.data.Dataset, dc.data.Dataset, dc.data.Dataset]:
        return self.train, self.valid, self.test

    def validate(self) -> None:
        """Raise on malformed labels, arrays, IDs, or split overlap."""
        if not self.tasks:
            raise ValueError("Le loader DeepChem n'a retourné aucune tâche.")
        seen_ids: set[str] = set()
        for split_name, dataset in zip(
            ("train", "validation", "test"),
            self.datasets,
            strict=True,
        ):
            labels = np.asarray(dataset.y)
            weights = np.asarray(dataset.w)
            ids = np.asarray(dataset.ids)
            if labels.ndim != 2 or labels.shape[1] != len(self.tasks):
                raise ValueError(
                    f"{split_name}: y attendu de forme (n, {len(self.tasks)}), "
                    f"reçu {labels.shape}."
                )
            if weights.shape != labels.shape:
                raise ValueError(
                    f"{split_name}: w de forme {weights.shape}, y de forme "
                    f"{labels.shape}."
                )
            if len(dataset.X) != len(labels) or len(ids) != len(labels):
                raise ValueError(f"{split_name}: X, y, w et ids ont des tailles différentes.")
            observed = weights > 0
            if not np.isin(weights, [0, 1]).all():
                raise ValueError(
                    f"{split_name}: les poids attendus sont 0 (masqué) ou 1."
                )
            if not np.isin(labels[observed], [0, 1]).all():
                raise ValueError(
                    f"{split_name}: labels observés attendus dans {{0, 1}}."
                )
            if pd.isna(ids).any():
                raise ValueError(f"{split_name}: IDs manquants.")
            split_ids = [str(value) for value in ids]
            if len(set(split_ids)) != len(split_ids):
                raise ValueError(f"{split_name}: IDs dupliqués.")
            overlap = seen_ids.intersection(split_ids)
            if overlap:
                raise ValueError(
                    f"IDs communs entre partitions (ex. {next(iter(overlap))!r})."
                )
            seen_ids.update(split_ids)

    def audit(self) -> dict:
        """Summarize split sizes, SMILES IDs, features, and masked task labels."""
        self.validate()
        first_graph = next(
            (
                graph
                for dataset in self.datasets
                for graph in dataset.X
                if graph is not None
            ),
            None,
        )
        sample_ids = [
            str(value)
            for dataset in self.datasets
            for value in dataset.ids[: min(20, len(dataset))]
        ]
        parsed_smiles = sum(Chem.MolFromSmiles(smile) is not None for smile in sample_ids)
        split_stats = {}
        for split_name, dataset in zip(
            ("train", "validation", "test"),
            self.datasets,
            strict=True,
        ):
            labels = np.asarray(dataset.y)
            observed = np.asarray(dataset.w) > 0
            task_stats = {}
            for task_index, task_name in enumerate(self.tasks):
                task_labels = labels[:, task_index]
                task_observed = observed[:, task_index]
                task_stats[task_name] = {
                    "positive": int(np.sum(task_observed & (task_labels == 1))),
                    "negative": int(np.sum(task_observed & (task_labels == 0))),
                    "missing": int(np.sum(~task_observed)),
                }
            split_stats[split_name] = {
                "compounds": int(len(dataset)),
                "label_cells": int(labels.size),
                "observed_label_cells": int(observed.sum()),
                "masked_label_cells": int((~observed).sum()),
                "tasks": task_stats,
            }
        total = sum(split["compounds"] for split in split_stats.values())
        return {
            "dataset": "MoleculeNet Tox21 via deepchem.molnet.load_tox21",
            "deepchem_version": dc.__version__,
            "tasks": list(self.tasks),
            "featurizer": self.featurizer,
            "feature_representation": (
                f"DeepChem {self.featurizer} molecular graphs"
            ),
            "feature_object_type": (
                f"{type(first_graph).__module__}.{type(first_graph).__name__}"
                if first_graph is not None
                else None
            ),
            "ids": {
                "field": "Dataset.ids",
                "contains_smiles": bool(sample_ids)
                and parsed_smiles == len(sample_ids),
                "sample_size_checked": len(sample_ids),
                "sample_smiles_parsed": parsed_smiles,
            },
            "splitter": self.splitter,
            "requested_split_fractions": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1,
            },
            "split_seed": self.seed,
            "split_seed_note": (
                "DeepChem 2.8.0 ScaffoldSplitter sorts scaffold groups and does "
                "not use its seed argument; the default official loader split is "
                "preserved."
                if self.splitter.lower() == "scaffold"
                else "The official loader's splitter was used."
            ),
            "transformers": list(self.transformers),
            "labels": {
                "observed_values": [0, 1],
                "missing_handling": (
                    "Dataset.y contains label values and may use 0 as a placeholder "
                    "for missing assays. Dataset.w == 0 masks those entries; "
                    "Dataset.w == 1 marks observed labels. Metrics and losses must "
                    "use w and must not treat masked y values as negatives."
                ),
            },
            "total_compounds": total,
            "splits": split_stats,
        }


def load_tox21_splits(
    cache_dir: Path = MOLNET_CACHE_DIR,
    *,
    featurizer: str = "GraphConv",
    splitter: str = "scaffold",
    reload: bool = True,
) -> Tox21Splits:
    """Load the official MoleculeNet Tox21 train/validation/test partitions.

    DeepChem 2.8.0's loader applies its default 80/10/10 scaffold split. No
    balancing or other label-transforming transformer is applied so that the
    original assay labels and missing-label masks remain directly auditable.
    """
    cache_dir = Path(cache_dir)
    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer=featurizer,
        splitter=splitter,
        transformers=[],
        reload=reload,
        data_dir=str(cache_dir / "raw"),
        save_dir=str(cache_dir / "processed"),
    )
    if len(datasets) != 3:
        raise ValueError(
            "Le loader Tox21 doit retourner train, validation et test; "
            f"il a retourné {len(datasets)} partition(s)."
        )
    result = Tox21Splits(
        tasks=tuple(tasks),
        train=datasets[0],
        valid=datasets[1],
        test=datasets[2],
        featurizer=featurizer,
        splitter=splitter,
        transformers=tuple(type(transformer).__name__ for transformer in transformers),
    )
    result.validate()
    return result


def write_tox21_audit(
    splits: Tox21Splits,
    output_file: Path = MOLNET_AUDIT_FILE,
) -> dict:
    """Write and return the auditable statistics from official DeepChem splits."""
    report = splits.audit()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
