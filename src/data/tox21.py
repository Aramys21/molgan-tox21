"""Load and clean the local Tox21 CSV without changing the source file."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "tox21.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "tox21_clean.csv"
MOLGAN_COMPATIBLE_FILE = PROCESSED_DIR / "tox21_molgan_compatible.csv"
ANALYSIS_FILE = PROCESSED_DIR / "tox21_analysis.json"

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
MOLGAN_ATOM_NUMBERS = {6, 7, 8, 9}
MOLGAN_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}
MOLGAN_MAX_ATOMS = 9


def _merge_labels(values: pd.Series) -> float:
    observed = values.dropna().unique()
    return float(observed[0]) if len(observed) == 1 else float("nan")


def _molgan_compatible(smiles: str) -> bool:
    molecule = Chem.MolFromSmiles(smiles)
    return bool(
        molecule is not None
        and molecule.GetNumAtoms() <= MOLGAN_MAX_ATOMS
        and all(
            atom.GetAtomicNum() in MOLGAN_ATOM_NUMBERS
            for atom in molecule.GetAtoms()
        )
        and all(
            bond.GetBondType() in MOLGAN_BOND_TYPES
            for bond in molecule.GetBonds()
        )
    )


def analyze_and_prepare(
    raw_file: Path = RAW_FILE,
    output_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, dict]:
    """Canonicalize structures, preserve missing labels, and report data quality."""
    raw_file = Path(raw_file)
    output_dir = Path(output_dir)
    if not raw_file.is_file():
        raise FileNotFoundError(f"Fichier Tox21 introuvable : {raw_file}")

    raw = pd.read_csv(raw_file)
    required_columns = {"smiles", "mol_id", *TASKS}
    missing_columns = sorted(required_columns.difference(raw.columns))
    if missing_columns:
        raise ValueError(
            "Colonnes requises absentes : " + ", ".join(missing_columns)
        )

    report = {
        "source": str(raw_file),
        "raw_rows": int(len(raw)),
        "raw_columns": list(raw.columns),
        "missing_by_column": {
            column: int(count) for column, count in raw.isna().sum().items()
        },
        "duplicate_full_rows": int(raw.duplicated().sum()),
        "duplicate_mol_ids": int(raw["mol_id"].duplicated().sum()),
        "duplicate_smiles_text": int(raw["smiles"].duplicated().sum()),
    }

    frame = raw.copy()
    invalid_label_counts = {}
    for task in TASKS:
        frame[task] = pd.to_numeric(frame[task], errors="coerce")
        invalid = frame[task].notna() & ~frame[task].isin([0, 1])
        invalid_label_counts[task] = int(invalid.sum())
        frame.loc[invalid, task] = float("nan")
    report["invalid_label_values_converted_to_missing"] = invalid_label_counts

    canonical_smiles = []
    invalid_smiles_indices = []
    for index, value in frame["smiles"].items():
        smile = str(value).strip() if pd.notna(value) else ""
        molecule = Chem.MolFromSmiles(smile) if smile else None
        if molecule is None:
            invalid_smiles_indices.append(int(index))
            canonical_smiles.append(None)
        else:
            canonical_smiles.append(Chem.MolToSmiles(molecule, canonical=True))
    frame["smiles"] = canonical_smiles
    frame = frame.loc[frame["smiles"].notna()].copy()
    report["invalid_smiles_rows_removed"] = len(invalid_smiles_indices)
    report["invalid_smiles_source_indices"] = invalid_smiles_indices
    report["valid_rows_before_structure_deduplication"] = int(len(frame))
    report["duplicate_canonical_structures"] = int(frame["smiles"].duplicated().sum())

    duplicate_rows = frame.loc[frame["smiles"].duplicated(keep=False)]
    conflicts = {}
    for task in TASKS:
        conflicts[task] = sum(
            len(group[task].dropna().unique()) > 1
            for _, group in duplicate_rows.groupby("smiles")
        )
    report["duplicate_structure_label_conflicts"] = conflicts

    clean = (
        frame.groupby("smiles", as_index=False, sort=False)
        .agg(
            {
                "mol_id": "first",
                **{task: _merge_labels for task in TASKS},
            }
        )
        .loc[:, ["mol_id", "smiles", *TASKS]]
    )
    report["clean_rows"] = int(len(clean))
    report["tasks"] = {}
    for task in TASKS:
        labels = clean[task]
        labeled_count = int(labels.notna().sum())
        report["tasks"][task] = {
            "positive": int((labels == 1).sum()),
            "negative": int((labels == 0).sum()),
            "missing": int(labels.isna().sum()),
            "positive_rate_among_labeled": (
                float((labels == 1).sum() / labeled_count)
                if labeled_count
                else None
            ),
        }

    atom_counts = []
    elements = Counter()
    for smile in clean["smiles"]:
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(f"SMILES canonique invalide après nettoyage : {smile}")
        atom_counts.append(molecule.GetNumHeavyAtoms())
        elements.update(atom.GetSymbol() for atom in molecule.GetAtoms())
    report["clean_smiles"] = {
        "unique_canonical": int(clean["smiles"].nunique()),
        "heavy_atoms": {
            "minimum": int(min(atom_counts)) if atom_counts else None,
            "median": float(pd.Series(atom_counts).median()) if atom_counts else None,
            "maximum": int(max(atom_counts)) if atom_counts else None,
        },
        "element_counts": dict(elements.most_common()),
    }

    compatible_mask = clean["smiles"].map(_molgan_compatible)
    compatible = clean.loc[compatible_mask].copy()
    report["molgan_filter"] = {
        "maximum_atoms": MOLGAN_MAX_ATOMS,
        "allowed_elements": ["C", "N", "O", "F"],
        "compatible_rows": int(len(compatible)),
        "excluded_rows": int(len(clean) - len(compatible)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_dir / PROCESSED_FILE.name, index=False)
    compatible.to_csv(output_dir / MOLGAN_COMPATIBLE_FILE.name, index=False)
    (output_dir / ANALYSIS_FILE.name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return clean, report


def load_clean_tox21(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Jeu nettoyé absent : {path}. Lancez `python scripts/prepare_data.py`."
        )
    return pd.read_csv(path)
