from rdkit import Chem

from src.evaluation.metrics import evaluate_molecules


class FakePredictor:
    def predict_risk(self, molecules):
        return [0.25] * len(molecules)


def test_evaluation_reports_validity_uniqueness_diversity_and_risk():
    molecules = [
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("c1ccccc1"),
        None,
    ]

    metrics = evaluate_molecules(molecules, FakePredictor())

    assert metrics["generated_count"] == 4
    assert metrics["valid_count"] == 3
    assert metrics["validity_rate"] == 0.75
    assert metrics["unique_count"] == 2
    assert metrics["uniqueness_rate"] == 2 / 3
    assert metrics["mean_pairwise_tanimoto"] is not None
    assert metrics["mean_predicted_toxicity_risk"] == 0.25
