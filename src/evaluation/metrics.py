"""Evaluate generated molecule validity, uniqueness, diversity, and toxicity."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


def evaluate_molecules(molecules, toxicity_predictor) -> dict:
    molecules = list(molecules)
    smiles = []
    valid_molecules = []
    for molecule in molecules:
        if molecule is not None:
            from rdkit import Chem

            canonical = Chem.MolToSmiles(molecule, canonical=True)
            if canonical:
                smiles.append(canonical)
                valid_molecules.append(molecule)

    unique_smiles = set(smiles)
    fingerprints = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mol) for mol in valid_molecules]
    similarities = [
        DataStructs.TanimotoSimilarity(left, right)
        for left, right in combinations(fingerprints, 2)
    ]
    risks = toxicity_predictor.predict_risk(valid_molecules)
    return {
        "generated_count": len(molecules),
        "valid_count": len(valid_molecules),
        "validity_rate": (
            len(valid_molecules) / len(molecules) if molecules else 0.0
        ),
        "unique_count": len(unique_smiles),
        "uniqueness_rate": (
            len(unique_smiles) / len(valid_molecules)
            if valid_molecules
            else 0.0
        ),
        "mean_pairwise_tanimoto": (
            float(np.mean(similarities)) if similarities else None
        ),
        "mean_predicted_toxicity_risk": (
            float(np.mean(risks)) if len(risks) else None
        ),
    }
