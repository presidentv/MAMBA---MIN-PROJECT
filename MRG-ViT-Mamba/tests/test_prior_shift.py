"""Tests for the prior correction, balanced accuracy and TTA views.

The correction's whole value is that it recovers the accuracy that imbalance-
aware sampling costs, without touching the weights and without looking at the
held-out labels. So the tests check exactly that: it is identity when there is
nothing to correct, it moves predictions the right way when there is, it never
derives its shift from the evaluation split, and it leaves the model's ranking
of clips intact where the shift cannot change the argmax.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment import CenterCropView  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.prior_shift import (  # noqa: E402
    adjust_predictions, apply_adjustment, effective_train_counts, logit_adjustment,
    prior_from_counts,
)
from src.sampling import class_sample_weights  # noqa: E402
from src.utils import load_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

NATURAL_3 = {0: 247, 1: 2617, 2: 2494}          # merged DAiSEE train split
TEST_3 = [88, 882, 814]


# ------------------------------------------------------------------- priors
def test_prior_from_counts_normalises_a_dict_or_a_sequence():
    assert prior_from_counts(NATURAL_3, 3).sum() == pytest.approx(1.0)
    assert prior_from_counts([1, 1, 2], 3).tolist() == [0.25, 0.25, 0.5]


def test_prior_from_counts_accepts_string_keys_from_json():
    assert prior_from_counts({"0": 1, "1": 3}, 2).tolist() == [0.25, 0.75]


def test_a_missing_class_counts_as_zero_not_as_an_error():
    assert prior_from_counts({0: 5, 2: 5}, 3).tolist() == [0.5, 0.0, 0.5]


def test_all_zero_counts_are_rejected():
    with pytest.raises(ValueError, match="zero"):
        prior_from_counts({0: 0, 1: 0}, 2)


# --------------------------------------------------------------- the shift
def test_no_shift_when_the_priors_already_agree():
    shift = logit_adjustment(NATURAL_3, NATURAL_3, 3)
    assert np.allclose(shift, 0.0)


def test_identical_priors_leave_every_prediction_unchanged():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=200)
    out = adjust_predictions(probs, NATURAL_3, 3, natural_counts=NATURAL_3)
    assert np.array_equal(out["predictions"], probs.argmax(axis=1))


def test_the_shift_penalises_classes_the_sampler_over_drew():
    """sqrt sampling lifts class 0 from 4.6% to 13.4% of the draws.

    Correcting back to the natural prior must therefore push class 0's logit
    DOWN and the two common classes' logits up.
    """
    labels = np.concatenate([np.full(n, c) for c, n in NATURAL_3.items()])
    _, info = class_sample_weights(labels, 3, "sqrt", max_ratio=8.0)
    effective = {c: info["expected_share"][str(c)] for c in range(3)}
    shift = logit_adjustment(effective, NATURAL_3, 3)
    assert shift[0] < 0 < shift[1]
    assert shift[1] == pytest.approx(shift[2], abs=0.05)   # both common classes


def test_the_correction_reduces_over_prediction_of_the_rare_class():
    """A model calibrated to a lifted prior over-predicts the rare class.

    Simulated: well-separated logits on a natural-prior test split, then a
    constant boost applied to class 0 as a sampler-trained model would learn.
    The correction should remove most of the resulting false positives.
    """
    rng = np.random.default_rng(1)
    y = np.concatenate([np.full(n, c) for c, n in enumerate(TEST_3)])
    logits = np.full((len(y), 3), -1.0)
    logits[np.arange(len(y)), y] = 1.0
    logits += 0.6 * rng.standard_normal(logits.shape)

    effective = {0: 721, 1: 2346, 2: 2291}                 # sqrt-sampled draws
    biased = logits + np.log(prior_from_counts(effective, 3) /
                             prior_from_counts(NATURAL_3, 3))[None, :]

    before = (biased.argmax(axis=1) == 0).sum()
    out = adjust_predictions(biased, effective, 3, natural_counts=NATURAL_3,
                             in_log_space=True)
    after = (out["predictions"] == 0).sum()
    true_zeros = int((y == 0).sum())

    assert before > after >= true_zeros * 0.5
    assert (out["predictions"] == y).mean() > (biased.argmax(axis=1) == y).mean()


def test_tau_scales_the_correction_and_zero_disables_it():
    probs = np.array([[0.4, 0.35, 0.25]])
    full = logit_adjustment({0: 100, 1: 10, 2: 10}, {0: 10, 1: 10, 2: 10}, 3, tau=1.0)
    half = logit_adjustment({0: 100, 1: 10, 2: 10}, {0: 10, 1: 10, 2: 10}, 3, tau=0.5)
    assert np.allclose(half, full / 2)
    none = adjust_predictions(probs, {0: 100, 1: 10, 2: 10}, 3,
                              natural_counts={0: 10, 1: 10, 2: 10}, tau=0.0)
    assert np.array_equal(none["predictions"], probs.argmax(axis=1))


def test_an_unobserved_class_does_not_produce_an_infinite_shift():
    shift = logit_adjustment({0: 100, 1: 0, 2: 50}, {0: 1, 1: 1, 2: 1}, 3)
    assert np.isfinite(shift).all()


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="do not match"):
        apply_adjustment(np.zeros((4, 3)), np.zeros(4))


# --------------------------------------------------- what the record claims
def test_the_record_states_the_priors_and_does_not_claim_to_use_test_labels():
    probs = np.full((10, 3), 1 / 3)
    rec = adjust_predictions(probs, {0: 721, 1: 2346, 2: 2291}, 3,
                             natural_counts=NATURAL_3)["record"]
    assert rec["applied"] is True
    assert rec["applied_to"] == "natural class distribution of the training split"
    assert sum(rec["target_prior"]) == pytest.approx(1.0, abs=1e-5)
    assert sum(rec["effective_train_prior"]) == pytest.approx(1.0, abs=1e-5)


def test_a_configured_reference_prior_is_used_and_named():
    cfg = {"evaluation": {"reference_prior": [0.049, 0.494, 0.456]}}
    rec = adjust_predictions(np.full((5, 3), 1 / 3), NATURAL_3, 3, cfg=cfg,
                             natural_counts=NATURAL_3)["record"]
    assert rec["target_prior"][0] == pytest.approx(0.049, abs=1e-3)
    assert "reference_prior" in rec["applied_to"]


# ------------------------------------------------- effective training counts
def test_effective_counts_follow_the_sampler():
    labels = np.concatenate([np.full(n, c) for c, n in NATURAL_3.items()])
    _, info = class_sample_weights(labels, 3, "sqrt", max_ratio=8.0)
    eff = effective_train_counts(NATURAL_3, info, 3)
    assert eff[0] > NATURAL_3[0]            # the rare class is drawn more often
    assert eff[1] < NATURAL_3[1]
    assert sum(eff.values()) == pytest.approx(sum(NATURAL_3.values()), rel=0.01)


def test_effective_counts_are_the_natural_counts_when_sampling_is_off():
    assert effective_train_counts(NATURAL_3, None, 3) == NATURAL_3
    assert effective_train_counts(NATURAL_3, {"sampler": "shuffle"}, 3) == NATURAL_3


# ------------------------------------------------------- balanced accuracy
def test_balanced_accuracy_is_not_fooled_by_the_majority_class():
    y = np.concatenate([np.full(n, c) for c, n in enumerate(TEST_3)])
    m = compute_metrics(y, np.full_like(y, 1), 3)          # always "High"
    assert m["accuracy"] == pytest.approx(0.494, abs=0.002)
    assert m["balanced_accuracy"] == pytest.approx(1 / 3, abs=1e-6)


def test_balanced_accuracy_equals_accuracy_on_a_balanced_split():
    rng = np.random.default_rng(3)
    y = np.concatenate([np.full(100, c) for c in range(3)])
    pred = np.where(rng.random(len(y)) < 0.7, y, rng.integers(0, 3, len(y)))
    m = compute_metrics(y, pred, 3)
    assert m["balanced_accuracy"] == pytest.approx(m["accuracy"], abs=1e-6)


def test_metrics_report_the_split_prior_and_majority_accuracy():
    y = np.concatenate([np.full(n, c) for c, n in enumerate(TEST_3)])
    m = compute_metrics(y, y, 3)
    assert m["majority_class_accuracy"] == pytest.approx(0.494, abs=0.002)
    assert sum(m["class_prior"]) == pytest.approx(1.0, abs=1e-5)


# -------------------------------------------------------------- TTA views
def test_scale_one_is_the_untouched_clip():
    clip = np.random.default_rng(4).integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)
    assert np.array_equal(CenterCropView(1.0)(clip), clip)


def test_a_smaller_scale_crops_centrally_and_keeps_the_shape():
    clip = np.random.default_rng(5).integers(0, 255, (4, 48, 48, 3), dtype=np.uint8)
    out = CenterCropView(0.5)(clip)
    assert out.shape == clip.shape and out.dtype == np.uint8
    assert not np.array_equal(out, clip)


def test_the_views_are_deterministic():
    # A reported score that depends on an RNG is not reproducible.
    clip = np.random.default_rng(6).integers(0, 255, (2, 40, 40, 3), dtype=np.uint8)
    view = CenterCropView(0.8)
    assert np.array_equal(view(clip), view(clip))


def test_every_frame_of_a_clip_gets_the_same_crop():
    clip = np.tile(np.arange(40, dtype=np.uint8)[None, :, None, None], (5, 1, 40, 3))
    out = CenterCropView(0.64)(clip)
    assert all(np.array_equal(out[0], out[i]) for i in range(1, 5))


def test_an_invalid_scale_is_rejected():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="scale"):
            CenterCropView(bad)


def test_views_come_from_the_config_and_default_to_none():
    assert CenterCropView.set_from_config(None) == []
    assert CenterCropView.set_from_config({}) == []
    views = CenterCropView.set_from_config({"tta_scales": [1.0, 0.9]})
    assert [v.scale for v in views] == [1.0, 0.9]


def test_three_class_config_declares_tta_including_the_plain_pass():
    cfg = load_config(REPO / "configs" / "config_full_3class.yaml")
    scales = cfg["evaluation"]["tta_scales"]
    assert 1.0 in scales, "the ordinary single-pass view must be part of the average"
    assert all(0.0 < float(s) <= 1.0 for s in scales)
