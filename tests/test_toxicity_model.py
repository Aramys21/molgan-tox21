import deepchem as dc
import numpy as np

from src.data.tox21 import TASKS
from src.models.toxicity import Tox21ToxicityPredictor


def test_predictor_can_be_saved_and_loaded(tmp_path):
    predictor = Tox21ToxicityPredictor(models={}, validation_aucs={})
    predictor_path = tmp_path / "predictor.joblib"

    predictor.save(predictor_path)
    loaded = Tox21ToxicityPredictor.load(predictor_path)

    assert loaded.models == {}
    assert loaded.validation_aucs == {}
    assert loaded.fingerprint_generator is not None


def test_predictor_fits_train_partition_and_reports_validation_only():
    train_smiles = [
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
    ]
    valid_smiles = ["CCS", "CC(C)O", "C1CC1", "CC=O"]
    train_y = (
        np.arange(len(train_smiles))[:, None] + np.arange(len(TASKS))[None, :]
    ) % 2
    valid_y = (
        np.arange(len(valid_smiles))[:, None] + np.arange(len(TASKS))[None, :]
    ) % 2
    train_data = dc.data.NumpyDataset(
        X=np.zeros((len(train_smiles), 1)),
        y=train_y,
        w=np.ones_like(train_y),
        ids=train_smiles,
    )
    valid_data = dc.data.NumpyDataset(
        X=np.zeros((len(valid_smiles), 1)),
        y=valid_y,
        w=np.ones_like(valid_y),
        ids=valid_smiles,
    )

    predictor = Tox21ToxicityPredictor.fit(
        train_data,
        valid_data,
        estimators=3,
        seed=42,
    )

    assert predictor.training_compounds == len(train_smiles)
    assert set(predictor.validation_aucs) <= set(TASKS)
    assert all(0 <= auc <= 1 for auc in predictor.validation_aucs.values())
    assert len(predictor.models) == len(TASKS)
