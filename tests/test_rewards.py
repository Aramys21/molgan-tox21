import numpy as np
import pytest

from src.rewards.safety import compute_safety_rewards


def test_safety_rewards_penalize_toxicity_and_invalid_samples():
    rewards = compute_safety_rewards(
        risks=[0.0, 0.5, 1.0],
        valid_mask=[True, True, False],
        toxicity_weight=2.0,
    )

    np.testing.assert_allclose(rewards, [1.0, 0.0, -2.0])


def test_negative_toxicity_weight_is_rejected():
    with pytest.raises(ValueError, match="positif ou nul"):
        compute_safety_rewards([0.4], [True], toxicity_weight=-0.1)


def test_risk_and_validity_shapes_must_match():
    with pytest.raises(ValueError, match="même dimension"):
        compute_safety_rewards([0.2, 0.8], [True], toxicity_weight=1.0)
