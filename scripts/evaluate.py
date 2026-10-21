from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_molecules
from src.models.toxicity import Tox21ToxicityPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue les SMILES générés.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts" / "generated_molecules.csv",
    )
    parser.add_argument(
        "--predictor",
        type=Path,
        default=ROOT / "artifacts" / "tox21_toxicity.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "evaluation.json",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"CSV molécules absent : {args.input}")
    predictor = Tox21ToxicityPredictor.load(args.predictor)
    frame = pd.read_csv(args.input)
    if "smiles" not in frame.columns:
        raise ValueError("Le CSV d'entrée doit contenir une colonne 'smiles'.")
    molecules = [
        molecule
        for smile in frame["smiles"].dropna().astype(str)
        if (molecule := Chem.MolFromSmiles(smile)) is not None
    ]
    metrics = evaluate_molecules(molecules, predictor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
