import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.data.molgan_graphs import GraphReconstruction
from src.evaluation.tox21_guided import evaluate_tox21_guided_samples
from scripts.compare_tox21_rewards import _diagnose_arms


class FakePredictor:
    endpoint = "NR-AR"
    biological_interpretation = "NR-AR reporter assay activity"
    decision_threshold = 0.5
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=128,
    )

    @staticmethod
    def predict_activity_with_uncertainty(molecules):
        return (
            np.full(len(molecules), 0.6),
            np.full(len(molecules), 0.2),
        )


def _reconstruction(molecules, smiles):
    count = len(molecules)
    return GraphReconstruction(
        molecules=molecules,
        canonical_smiles=smiles,
        failures=[],
        tensor_valid_count=count,
        conversion_count=count,
        sanitized_count=sum(molecule is not None for molecule in molecules),
        ignored_padding_edge_count=0,
    )


def test_sample_metrics_use_valid_and_unique_denominators():
    first = Chem.MolFromSmiles("CCO")
    second = Chem.MolFromSmiles("CCO")
    invalid = None
    report, rows = evaluate_tox21_guided_samples(
        _reconstruction([first, second, invalid], ["CCO", "CCO", None]),
        FakePredictor(),
        {"CCO"},
        [FakePredictor.fingerprint_generator.GetFingerprint(first)],
        out_of_domain_similarity_threshold=0.25,
    )

    assert report["generated_graph_count"] == 3
    assert report["rdkit_valid_count"] == 2
    assert report["validity_rate"] == pytest.approx(2 / 3)
    assert report["unique_valid_count"] == 1
    assert report["uniqueness_rate"] == pytest.approx(1 / 2)
    assert report["novel_unique_count"] == 0
    assert report["unique_sample_predicted_active_rate"] == 1.0
    assert report["toxicity_evaluation_excluded_valid_count"] == 0
    assert len(rows) == 2


def test_empty_generated_samples_return_null_assay_metrics():
    report, rows = evaluate_tox21_guided_samples(
        _reconstruction([], []),
        FakePredictor(),
        set(),
        [FakePredictor.fingerprint_generator.GetFingerprint(Chem.MolFromSmiles("C"))],
        out_of_domain_similarity_threshold=0.25,
    )

    assert report["generated_graph_count"] == 0
    assert report["validity_rate"] == 0.0
    assert report["unique_sample_activity_probability_mean"] is None
    assert report["mean_pairwise_tanimoto_distance_unique"] is None
    assert rows == []


def test_out_of_range_support_threshold_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_tox21_guided_samples(
            _reconstruction([], []),
            FakePredictor(),
            set(),
            [],
            out_of_domain_similarity_threshold=1.1,
        )


def test_diagnostics_decline_to_infer_with_no_valid_samples():
    reports = [
        {
            "arm": "baseline",
            "unique_valid_count": 0,
        },
        {
            "arm": "activity_focused",
            "unique_valid_count": 0,
        },
    ]

    diagnostics = _diagnose_arms(
        reports,
        {
            "ood_fraction_increase": 0.25,
            "uncertainty_increase": 0.1,
            "low_uniqueness_rate": 0.5,
            "low_tanimoto_distance": 0.2,
        },
    )

    assert diagnostics[0]["possible_mode_collapse_flag"] is None
    assert diagnostics[1]["possible_predictor_exploitation_flag"] is None
    assert diagnostics[1]["assessment"] == "insufficient unique valid molecules"
