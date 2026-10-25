from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.molgan_graphs import featurize_smiles, reconstruct_graphs
from src.data.tox21 import load_tox21_splits
from src.visualization.molecules import save_molecule_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Démontre la préparation et la reconstruction de graphes Tox21."
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "deepchem_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "molgan_graph_demo.png",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts" / "molgan_graph_demo.json",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count doit être strictement positif.")

    splits = load_tox21_splits(cache_dir=args.cache_dir)
    preparation = featurize_smiles(splits.train.ids)
    graphs = preparation.graphs[: args.count]
    original_smiles = preparation.canonical_smiles[: len(graphs)]
    reconstruction = reconstruct_graphs(graphs, preparation.featurizer)
    report = {
        "source_split": "official_scaffold_train",
        "molgan_input_preparation": preparation.summary(),
        "graph_reconstruction": reconstruction.summary(),
        "structure_comparisons": [
            {
                "original_canonical_smiles": original,
                "reconstructed_canonical_smiles": generated,
                "same_canonical_structure": original == generated,
            }
            for original, generated in zip(
                original_smiles,
                reconstruction.canonical_smiles,
            )
            if generated is not None
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    valid_pairs = [
        (original, molecule)
        for original, molecule in zip(original_smiles, reconstruction.molecules)
        if molecule is not None
    ]
    if not valid_pairs:
        raise RuntimeError(
            "Aucune molécule n'a survécu à la reconstruction; consultez les "
            "échecs affichés ci-dessus."
        )
    save_molecule_grid(
        [molecule for _, molecule in valid_pairs],
        args.output,
        legends=[smiles for smiles, _ in valid_pairs],
    )
    print(f"Image RDKit enregistrée : {args.output}")
    print(f"Rapport enregistré : {args.report}")


if __name__ == "__main__":
    main()
