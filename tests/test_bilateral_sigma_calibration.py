from __future__ import annotations

import numpy as np
import pytest

from tactile_force_envelope.bilateral_sigma_calibration import (
    _finite_sample_tail_quantile,
    bilateral_uncertainty_scale,
)


def test_finite_sample_tail_ranks_are_conservative() -> None:
    values = np.arange(1.0, 100.0)

    lower, lower_rank = _finite_sample_tail_quantile(values, 0.05, upper=False)
    upper, upper_rank = _finite_sample_tail_quantile(values, 0.95, upper=True)

    assert (lower, lower_rank) == (5.0, 5)
    assert (upper, upper_rank) == (95.0, 95)


def test_finite_sample_tail_quantile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        _finite_sample_tail_quantile(np.asarray([]), 0.95, upper=True)


def test_bilateral_uncertainty_scale_definitions() -> None:
    left = np.asarray([3.0, 4.0])
    right = np.asarray([4.0, 3.0])

    np.testing.assert_allclose(
        bilateral_uncertainty_scale(left, right, "mean"), [3.5, 3.5]
    )
    np.testing.assert_allclose(
        bilateral_uncertainty_scale(left, right, "rss"), [2.5, 2.5]
    )
    np.testing.assert_allclose(
        bilateral_uncertainty_scale(left, right, "max"), [4.0, 4.0]
    )


def test_bilateral_uncertainty_scale_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown bilateral uncertainty scale"):
        bilateral_uncertainty_scale(np.ones(2), np.ones(2), "std")
