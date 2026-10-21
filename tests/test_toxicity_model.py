from src.models.toxicity import Tox21ToxicityPredictor


def test_predictor_can_be_saved_and_loaded(tmp_path):
    predictor = Tox21ToxicityPredictor(models={}, validation_aucs={})
    predictor_path = tmp_path / "predictor.joblib"

    predictor.save(predictor_path)
    loaded = Tox21ToxicityPredictor.load(predictor_path)

    assert loaded.models == {}
    assert loaded.validation_aucs == {}
    assert loaded.fingerprint_generator is not None
