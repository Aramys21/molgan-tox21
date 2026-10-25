import pytest
import deepchem as dc
import numpy as np

from src.data.tox21 import (
    TASKS,
    Tox21Splits,
    load_tox21_splits,
    write_tox21_audit,
)


def _fake_split(ids):
    features = dc.feat.ConvMolFeaturizer().featurize(ids)
    labels = np.tile(np.arange(len(TASKS)) % 2, (len(ids), 1)).astype(float)
    weights = np.ones_like(labels)
    if len(ids):
        weights[0, 0] = 0
    return dc.data.NumpyDataset(
        X=features,
        y=labels,
        w=weights,
        ids=np.asarray(ids),
    )


def _fake_loader_result():
    return (
        TASKS,
        (
            _fake_split(["CCO", "CCN", "CCC", "CCCl"]),
            _fake_split(["c1ccccc1", "CC(=O)O"]),
            _fake_split(["COC", "CNC"]),
        ),
        [],
    )


def test_official_loader_is_called_with_graphconv_scaffold_and_no_transformers(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_load_tox21(**kwargs):
        calls.append(kwargs)
        return _fake_loader_result()

    monkeypatch.setattr(dc.molnet, "load_tox21", fake_load_tox21)

    splits = load_tox21_splits(tmp_path, reload=False)

    assert calls == [
        {
            "featurizer": "GraphConv",
            "splitter": "scaffold",
            "transformers": [],
            "reload": False,
            "data_dir": str(tmp_path / "raw"),
            "save_dir": str(tmp_path / "processed"),
        }
    ]
    assert splits.tasks == tuple(TASKS)
    assert tuple(map(len, splits.datasets)) == (4, 2, 2)


def test_split_validation_checks_shapes_binary_observed_labels_and_mask_values(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dc.molnet,
        "load_tox21",
        lambda **kwargs: _fake_loader_result(),
    )
    splits = load_tox21_splits(tmp_path)
    splits.validate()

    for dataset in splits.datasets:
        assert dataset.y.shape == (len(dataset), len(TASKS))
        assert dataset.w.shape == dataset.y.shape
        observed = dataset.w > 0
        assert set(np.unique(dataset.y[observed])) <= {0.0, 1.0}
        assert set(np.unique(dataset.w)) <= {0.0, 1.0}


def test_split_ids_are_disjoint_and_repeated_loader_calls_are_reproducible(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dc.molnet,
        "load_tox21",
        lambda **kwargs: _fake_loader_result(),
    )
    first = load_tox21_splits(tmp_path)
    second = load_tox21_splits(tmp_path)

    first_ids = [tuple(dataset.ids) for dataset in first.datasets]
    second_ids = [tuple(dataset.ids) for dataset in second.datasets]
    assert first_ids == second_ids
    assert len(set().union(*(set(ids) for ids in first_ids))) == 8
    first.validate()


def test_audit_counts_masked_labels_and_smiles_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dc.molnet,
        "load_tox21",
        lambda **kwargs: _fake_loader_result(),
    )
    splits = load_tox21_splits(tmp_path)

    report = splits.audit()

    assert report["ids"]["contains_smiles"] is True
    assert report["feature_object_type"].endswith("ConvMol")
    assert report["total_compounds"] == 8
    assert report["split_seed"] is None
    assert report["splits"]["train"]["masked_label_cells"] == 1
    assert report["splits"]["train"]["tasks"]["NR-AR"]["missing"] == 1
    output_file = tmp_path / "audit" / "audit.json"
    written_report = write_tox21_audit(splits, output_file)
    assert output_file.is_file()
    assert written_report == report


def test_overlap_between_official_splits_is_rejected():
    train = _fake_split(["CCO", "CCN"])
    valid = _fake_split(["CCO"])
    test = _fake_split(["CCC"])
    splits = Tox21Splits(
        tasks=tuple(TASKS),
        train=train,
        valid=valid,
        test=test,
        featurizer="GraphConv",
        splitter="scaffold",
        transformers=(),
    )

    with pytest.raises(ValueError, match="IDs communs entre partitions"):
        splits.validate()
