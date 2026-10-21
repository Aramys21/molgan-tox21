import pandas as pd
import pytest
from rdkit import Chem

from src.data.tox21 import TASKS, analyze_and_prepare


def test_cleaning_canonicalizes_smiles_and_preserves_unlabeled_values(tmp_path):
    rows = [
        {
            "mol_id": "a",
            "smiles": "CCO",
            **{task: 0 for task in TASKS},
        },
        {
            "mol_id": "b",
            "smiles": "OCC",
            **{task: 1 for task in TASKS},
        },
        {
            "mol_id": "c",
            "smiles": "not-a-smiles",
            **{task: 0 for task in TASKS},
        },
    ]
    rows[1]["NR-AR"] = None
    input_file = tmp_path / "tox21.csv"
    pd.DataFrame(rows).to_csv(input_file, index=False)

    cleaned, report = analyze_and_prepare(input_file, tmp_path / "prepared")

    assert len(cleaned) == 1
    assert cleaned.loc[0, "smiles"] == Chem.MolToSmiles(
        Chem.MolFromSmiles("CCO"),
        canonical=True,
    )
    assert cleaned.loc[0, "NR-AR"] == 0
    assert report["invalid_smiles_rows_removed"] == 1
    assert report["duplicate_canonical_structures"] == 1
    assert report["duplicate_structure_label_conflicts"]["NR-AR"] == 0


def test_conflicting_duplicate_labels_become_missing(tmp_path):
    rows = []
    for mol_id, label in [("a", 0), ("b", 1)]:
        rows.append(
            {"mol_id": mol_id, "smiles": "CCO", **{task: 0 for task in TASKS}}
        )
        rows[-1]["NR-AR"] = label
    input_file = tmp_path / "tox21.csv"
    pd.DataFrame(rows).to_csv(input_file, index=False)

    cleaned, report = analyze_and_prepare(input_file, tmp_path / "prepared")

    assert len(cleaned) == 1
    assert pd.isna(cleaned.loc[0, "NR-AR"])
    assert report["duplicate_structure_label_conflicts"]["NR-AR"] == 1


def test_missing_required_column_raises(tmp_path):
    input_file = tmp_path / "tox21.csv"
    pd.DataFrame({"mol_id": ["a"], "smiles": ["CCO"]}).to_csv(
        input_file,
        index=False,
    )

    with pytest.raises(ValueError, match="Colonnes requises absentes"):
        analyze_and_prepare(input_file, tmp_path / "prepared")
