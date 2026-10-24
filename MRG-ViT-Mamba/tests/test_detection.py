"""Tests for the binary engaged / not-engaged detector metrics.

The property that matters most here is the one that makes accuracy useless on
this split: a model can score 0.95 accuracy while detecting nothing, and a
genuinely better detector can score lower. These tests pin that, so no future
change can quietly make accuracy look like the headline again.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import (  # noqa: E402
    choose_threshold, detection_scores, evaluate_detector, operating_point,
    ranking_metrics,
)

# The real test split under the binary merge: 88 not-engaged, 1,696 engaged.
N_POS, N_NEG = 88, 1696


def split(n_pos=N_POS, n_neg=N_NEG):
    """y where 0 = not engaged (the positive class for detection)."""
    return np.concatenate([np.zeros(n_pos, int), np.ones(n_neg, int)])


def probs_from_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return np.stack([scores, 1.0 - scores], axis=1)


# ------------------------------------------------- the point of the module
def test_a_constant_engaged_predictor_scores_95_percent_and_detects_nothing():
    y = split()
    # Every clip scored as definitely engaged: P(not engaged) = 0.
    out = evaluate_detector(y, probs_from_scores(np.zeros(len(y))))
    assert out["argmax"]["accuracy"] == pytest.approx(0.9507, abs=5e-4)
    assert out["argmax"]["recall_not_engaged"] == 0.0
    assert out["argmax"]["predicted_not_engaged"] == 0
    assert out["baselines"]["majority_class_accuracy"] == pytest.approx(0.9507, abs=5e-4)


def test_a_perfect_detector_and_a_useless_one_are_distinguished_by_auc():
    y = split()
    perfect = np.where(y == 0, 0.9, 0.1)
    useless = np.full(len(y), 0.5)
    assert ranking_metrics(y, perfect)["roc_auc"] == pytest.approx(1.0)
    assert ranking_metrics(y, useless)["roc_auc"] == pytest.approx(0.5)


def test_a_better_detector_can_report_lower_accuracy():
    """The reason accuracy is not the headline, pinned as a test.

    A detector with real signal that predicts the minority class sometimes
    loses accuracy against the constant predictor, while being far more useful.
    """
    y = split()
    rng = np.random.default_rng(0)
    # Real but imperfect signal.
    scores = np.where(y == 0, rng.normal(0.65, 0.15, len(y)),
                      rng.normal(0.35, 0.15, len(y))).clip(0, 1)
    useful = evaluate_detector(y, probs_from_scores(scores))
    constant = evaluate_detector(y, probs_from_scores(np.zeros(len(y))))

    assert useful["ranking"]["roc_auc"] > 0.85          # clearly a good detector
    assert useful["argmax"]["recall_not_engaged"] > 0.7
    assert useful["argmax"]["accuracy"] < constant["argmax"]["accuracy"]


def test_average_precision_baseline_is_the_positive_rate_not_half():
    y = split()
    out = evaluate_detector(y, probs_from_scores(np.full(len(y), 0.5)))
    assert out["baselines"]["average_precision"] == pytest.approx(N_POS / (N_POS + N_NEG))
    assert out["baselines"]["roc_auc"] == 0.5


# ------------------------------------------------------------- mechanics
def test_detection_score_is_the_not_engaged_probability():
    p = np.array([[0.7, 0.3], [0.1, 0.9]])
    assert detection_scores(p).tolist() == [0.7, 0.1]


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match=r"\[N, 2\]"):
        detection_scores(np.zeros((4, 3)))


def test_operating_point_counts_add_up():
    y = split(10, 20)
    scores = np.where(y == 0, 0.8, 0.2)
    pt = operating_point(y, scores, 0.5)
    c = pt["confusion"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(y)
    assert c["tp"] == 10 and c["tn"] == 20
    assert pt["recall_not_engaged"] == 1.0 and pt["specificity_engaged"] == 1.0


def test_a_higher_threshold_predicts_the_minority_class_less_often():
    y = split(50, 50)
    rng = np.random.default_rng(1)
    scores = rng.random(len(y))
    counts = [operating_point(y, scores, t)["predicted_not_engaged"]
              for t in (0.2, 0.5, 0.8)]
    assert counts[0] > counts[1] > counts[2]


# ------------------------------------------------------- threshold choice
def test_the_tuned_threshold_beats_argmax_on_its_own_objective():
    y = split()
    rng = np.random.default_rng(2)
    scores = np.where(y == 0, rng.normal(0.55, 0.2, len(y)),
                      rng.normal(0.45, 0.2, len(y))).clip(0, 1)
    chosen = choose_threshold(y, scores, "balanced_accuracy")
    assert (chosen["point"]["balanced_accuracy"]
            >= operating_point(y, scores, 0.5)["balanced_accuracy"])


def test_an_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="objective"):
        choose_threshold(split(5, 5), np.linspace(0, 1, 10), "accuracy_please")


def test_the_threshold_is_tuned_on_validation_and_the_record_says_so():
    """The integrity property: test labels must not choose the threshold."""
    rng = np.random.default_rng(3)
    y_val, y_test = split(100, 400), split(88, 1696)
    s_val = np.where(y_val == 0, rng.normal(0.6, 0.2, len(y_val)),
                     rng.normal(0.4, 0.2, len(y_val))).clip(0, 1)
    s_test = np.where(y_test == 0, rng.normal(0.6, 0.2, len(y_test)),
                      rng.normal(0.4, 0.2, len(y_test))).clip(0, 1)

    out = evaluate_detector(y_test, probs_from_scores(s_test),
                            tune_on=(y_val, probs_from_scores(s_val)))
    assert out["tuned"]["threshold_chosen_on"] == "validation split"
    # The same threshold must come out of validation alone.
    expected = choose_threshold(y_val, s_val, "balanced_accuracy")["threshold"]
    assert out["tuned"]["threshold"] == pytest.approx(expected)


def test_a_single_class_split_does_not_crash():
    y = np.ones(20, int)
    out = evaluate_detector(y, probs_from_scores(np.full(20, 0.3)))
    assert out["ranking"]["roc_auc"] is None
    assert "undefined" in out["ranking"]["note"]


def test_degenerate_tuning_split_falls_back_to_half():
    chosen = choose_threshold(np.ones(10, int), np.linspace(0, 1, 10))
    assert chosen["threshold"] == 0.5 and "one class" in chosen["note"]
