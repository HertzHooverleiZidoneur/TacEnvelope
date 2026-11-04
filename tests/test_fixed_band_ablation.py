from __future__ import annotations

import numpy as np
import pytest

from tactile_force_envelope.fixed_band_ablation import (
    _resolve_checkpoint_path,
    interval_score,
)


def test_interval_score_penalizes_both_tails() -> None:
    target = np.asarray([-1.0, 1.5, 4.0])
    lower = np.asarray([0.0, 0.0, 0.0])
    upper = np.asarray([3.0, 3.0, 3.0])
    score = interval_score(target, lower, upper, alpha=0.1)
    assert score.tolist() == pytest.approx([23.0, 3.0, 23.0])


def test_interval_score_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="upper bound"):
        interval_score(np.ones(1), np.ones(1), np.zeros(1))


def test_checkpoint_resolution_falls_back_to_configured_source(tmp_path) -> None:
    source = tmp_path / "seed44" / "transformer_force_envelope_model.pt"
    source.parent.mkdir()
    source.touch()
    config = {
        "training": {"seed": 44},
        "stable_contact": {"source_checkpoint": str(source)},
    }
    assert _resolve_checkpoint_path(config, tmp_path, 44) == source.resolve()


def test_checkpoint_resolution_refuses_wrong_source_seed(tmp_path) -> None:
    source = tmp_path / "seed44" / "transformer_force_envelope_model.pt"
    source.parent.mkdir()
    source.touch()
    config = {
        "training": {"seed": 44},
        "stable_contact": {"source_checkpoint": str(source)},
    }
    with pytest.raises(FileNotFoundError, match="refusing to retrain"):
        _resolve_checkpoint_path(config, tmp_path, 43)
