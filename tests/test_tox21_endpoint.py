import deepchem as dc
import numpy as np
import pytest
from rdkit import Chem

from src.data.tox21 import TASKS
from src.evaluation.tox21_assay import evaluate_tox21_endpoint
from src.evaluation.tox21_endpoint import binary_classification_metrics
from src.models.tox21_endpoint import (
    Tox21EndpointPredictor,
    extract_endpoint_data,
)


def _dataset(smiles, labels, weights=None):
    labels = np.asarray(labels, dtype=float)
    if labels.ndim == 1:
        matrix = np.zeros((len(labels), len(TASKS)), dtype=float)
        matrix[:, TASKS.index("NR-AR")] = labels
        labels = matrix
    if weights is None:
        weights = np.ones_like(labels)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 1:
            matrix = np.ones((len(weights), len(TASKS)), dtype=float)
            matrix[:, TASKS.index("NR-AR")] = weights
            weights = matrix
    return dc.data.NumpyDataset(
        X=np.zeros((len(smiles), 1)),
        y=labels,
        w=weights,
        ids=np.asarray(smiles),
    )


def _training_sets():
    smiles = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "c1ccccc1",
        "CC(=O)O",
        "COC",
        "CNC",
        "CCBr",
        "CCF",
        "CCS",
        "CC(C)O",
        "C1CC1",
        "CC=O",
        "CCCO",
        "CCCN",
        "CC(C)C",
        "c1ccncc1",
        "CCOC",
        "CCNC",
    ]
    train_y = np.asarray([0, 1] * 10)
    validation_smiles = [
        "CC(C)N",
        "CC(C)Cl",
        "CC(C)F",
        "CC(C)Br",
        "CC(C)C=O",
        "CC(C)CO",
        "c1ccoc1",
        "c1ccsc1",
    ]
    validation_y = np.asarray([0, 1] * 4)
    train_weights = np.ones_like(train_y)
    train_weights[0] = 0
    return (
        _dataset(smiles, train_y, train_weights),
        _dataset(validation_smiles, validation_y),
    )


def test_masked_labels_are_excluded_not_replaced_with_negative():
    train, _ = _training_sets()
    x, y, counts = extract_endpoint_data(train, "NR-AR", fingerprint_size=128)

    assert x.shape == (19, 128)
    assert counts["masked_label_count"] == 1
    assert counts["usable_label_count"] == 19
    assert counts["negative_count"] == 9
    assert counts["positive_count"] == 10
    assert y.sum() == 10


def test_fit_selects_using_validation_and_returns_bounded_probabilities(tmp_path):
    train, valid = _training_sets()
    predictor = Tox21EndpointPredictor.fit(
        train,
        valid,
        endpoint="NR-AR",
        estimator_candidates=[5, 10],
        fingerprint_size=128,
        seed=19,
    )
    molecules = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]

    probabilities = predictor.predict_active_probability(molecules)
    predictions = predictor.predict(molecules)

    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(predictor.candidate_validation_auc) == {"5", "10"}
    assert predictor.selected_estimators in {5, 10}
    assert predictor.validation_counts["masked_label_count"] == 0
    assert all(
        prediction["outcome_label"] == "predicted Tox21 assay activity"
        for prediction in predictions
    )
    model_path = tmp_path / "predictor.joblib"
    predictor.save(model_path)
    reloaded = Tox21EndpointPredictor.load(model_path)
    np.testing.assert_allclose(
        probabilities,
        reloaded.predict_active_probability(molecules),
        rtol=1e-14,
        atol=1e-15,
    )


def test_invalid_molecules_are_explicitly_rejected():
    train, valid = _training_sets()
    predictor = Tox21EndpointPredictor.fit(
        train,
        valid,
        estimator_candidates=[3],
        fingerprint_size=64,
    )
    invalid_smiles = Chem.MolFromSmiles("C1CC")

    with pytest.raises(ValueError, match="index 0"):
        predictor.predict_active_probability([invalid_smiles])

    unsanitized = Chem.MolFromSmiles("[CH5]", sanitize=False)
    with pytest.raises(ValueError, match="index 0"):
        predictor.predict_active_probability([unsanitized])


def test_endpoint_and_class_requirements_are_validated():
    train, valid = _training_sets()
    with pytest.raises(ValueError, match="Endpoint non pris en charge"):
        Tox21EndpointPredictor.fit(train, valid, endpoint="NR-AhR")
    with pytest.raises(ValueError, match="entiers positifs"):
        Tox21EndpointPredictor.fit(train, valid, estimator_candidates=[])
    one_class = _dataset(["CCO", "CCN"], [0, 0])
    with pytest.raises(ValueError, match="deux classes"):
        Tox21EndpointPredictor.fit(one_class, valid, estimator_candidates=[2])


def test_seed_reproduces_selected_model_and_predictions():
    train, valid = _training_sets()
    kwargs = {
        "estimator_candidates": [5, 8],
        "fingerprint_size": 128,
        "seed": 71,
    }
    first = Tox21EndpointPredictor.fit(train, valid, **kwargs)
    second = Tox21EndpointPredictor.fit(train, valid, **kwargs)

    assert first.selected_estimators == second.selected_estimators
    assert first.decision_threshold == second.decision_threshold
    np.testing.assert_array_equal(
        first.predict_active_probability([Chem.MolFromSmiles("CCO")]),
        second.predict_active_probability([Chem.MolFromSmiles("CCO")]),
    )


def test_forest_disagreement_and_training_support_are_bounded():
    train, valid = _training_sets()
    predictor = Tox21EndpointPredictor.fit(
        train,
        valid,
        estimator_candidates=[5],
        fingerprint_size=128,
        seed=31,
    )

    probabilities, uncertainty = predictor.predict_activity_with_uncertainty(
        [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    )
    reference_fingerprints, invalid_count = predictor.reference_fingerprints(train)

    assert probabilities.shape == uncertainty.shape == (2,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.all((uncertainty >= 0.0) & (uncertainty <= 1.0))
    assert len(reference_fingerprints) == 19
    assert invalid_count == 0


def test_test_metrics_use_masks_and_report_bootstrap_intervals():
    train, valid = _training_sets()
    predictor = Tox21EndpointPredictor.fit(
        train,
        valid,
        estimator_candidates=[3],
        fingerprint_size=64,
    )
    test = _dataset(
        ["CCO", "CCN", "CCC", "CCCl", "C1CC"],
        [0, 1, 0, 1, 1],
        [1, 1, 1, 1, 0],
    )

    report, rows = evaluate_tox21_endpoint(
        predictor,
        test,
        split_name="test",
        n_bootstrap=20,
        seed=7,
    )

    assert report["split"] == "test"
    assert report["sample_count"] == 4
    assert report["class_counts"] == {"negative": 2, "positive": 2}
    assert report["masked_label_count"] == 1
    assert report["invalid_structure_count"] == 1
    assert len(rows) == 4
    interval = report["confidence_intervals_95_percent"]["roc_auc"]
    assert interval["lower"] <= report["metrics"]["roc_auc"] <= interval["upper"]


def test_binary_metrics_validate_probabilities_and_ci_reproduce():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    first = binary_classification_metrics(
        labels,
        probabilities,
        threshold=0.5,
        n_bootstrap=30,
        seed=5,
    )
    second = binary_classification_metrics(
        labels,
        probabilities,
        threshold=0.5,
        n_bootstrap=30,
        seed=5,
    )

    assert first == second
    assert first["metrics"]["roc_auc"] == 1.0
    assert 0 <= first["metrics"]["brier_score"] <= 1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        binary_classification_metrics(labels, [0.2, 0.4, 1.1, 0.9], 0.5)
