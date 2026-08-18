from __future__ import annotations

import numpy as np

from fruit_pipeline.realism import feature_statistics, frechet_distance


def test_frechet_distance_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(64, 8))
    mu, sigma = feature_statistics(features)
    distance = frechet_distance(mu, sigma, mu, sigma)
    assert distance < 1e-6


def test_frechet_distance_matches_closed_form_for_equal_covariance() -> None:
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([3.0, 4.0])
    sigma = np.eye(2)
    distance = frechet_distance(mu1, sigma, mu2, sigma)
    assert abs(distance - 25.0) < 1e-6
