"""Tests for imbalance-aware training sampling.

The quantities that matter are the ones a silent bug would get subtly wrong:
no clip may be discarded, the epoch length must not change, the rare classes
must actually be lifted, and the cap must bound repetition.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sampling import build_train_sampler, class_sample_weights  # noqa: E402
from src.utils import load_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Published DAiSEE engagement counts for the training split. Used as a realistic
# skew to test against; the pipeline itself always measures the real labels.
DAISEE_TRAIN = [34, 213, 2617, 2494]


def _labels(counts):
    return np.concatenate([np.full(n, c) for c, n in enumerate(counts)])


def test_natural_mode_keeps_the_real_distribution():
    _, info = class_sample_weights(_labels(DAISEE_TRAIN), 4, "natural")
    assert info["expected_share"] == info["natural_share"]


def test_sqrt_lifts_rare_classes_without_equalising_them():
    _, info = class_sample_weights(_labels(DAISEE_TRAIN), 4, "sqrt")
    share = [info["expected_share"][str(c)] for c in range(4)]
    natural = [info["natural_share"][str(c)] for c in range(4)]
    assert share[0] > natural[0] * 5          # rarest clearly lifted
    assert share[2] < natural[2]              # most common reduced
    assert share[0] < share[1] < share[2]     # but ordering preserved


def test_balanced_mode_equalises_shares():
    _, info = class_sample_weights(_labels(DAISEE_TRAIN), 4, "balanced")
    share = [info["expected_share"][str(c)] for c in range(4)]
    assert max(share) - min(share) < 1e-6


def test_max_ratio_bounds_repetition_of_the_rarest_class():
    _, capped = class_sample_weights(_labels(DAISEE_TRAIN), 4, "balanced", max_ratio=8.0)
    _, uncapped = class_sample_weights(_labels(DAISEE_TRAIN), 4, "balanced")
    assert capped["max_repetition_of_rarest"] == pytest.approx(8.0)
    assert uncapped["max_repetition_of_rarest"] > 50      # 2617/34 without a cap


def test_absent_class_gets_zero_weight_not_infinity():
    w, info = class_sample_weights(_labels([10, 0, 5, 5]), 4, "balanced")
    assert w[1] == 0.0 and np.isfinite(w).all()
    assert info["expected_share"]["1"] == 0.0


def test_sampler_preserves_epoch_length_and_discards_nothing():
    labels = _labels(DAISEE_TRAIN)
    sampler, info = build_train_sampler(labels, 4, {"sampling": "sqrt",
                                                    "sampling_max_ratio": 8.0})
    assert sampler is not None
    assert len(sampler) == len(labels)        # same number of steps per epoch
    assert "WeightedRandomSampler" in info["sampler"]


def test_natural_mode_returns_no_sampler_so_shuffle_is_used():
    sampler, info = build_train_sampler(_labels(DAISEE_TRAIN), 4, {"sampling": "natural"})
    assert sampler is None
    assert "shuffle" in info["sampler"]


def test_sampler_draws_match_the_intended_share():
    labels = _labels(DAISEE_TRAIN)
    sampler, info = build_train_sampler(labels, 4, {"sampling": "sqrt",
                                                    "sampling_max_ratio": 8.0})
    drawn = labels[np.fromiter(iter(sampler), dtype=int, count=len(sampler))]
    for c in range(4):
        observed = (drawn == c).mean()
        assert observed == pytest.approx(info["expected_share"][str(c)], abs=0.03)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        class_sample_weights(_labels(DAISEE_TRAIN), 4, "oversample_everything")


def test_full_config_requests_sqrt_sampling():
    cfg = load_config(REPO / "configs" / "config_full.yaml")
    assert cfg["training"]["sampling"] == "sqrt"
    assert cfg["training"]["sampling_max_ratio"] == 8.0
    # The development configs stay on the natural distribution: their splits are
    # already balanced, so sampling would only add noise.
    assert load_config(REPO / "configs" / "config_ft32.yaml")["training"].get(
        "sampling", "natural") == "natural"
