"""Metrics for the binary engaged / not-engaged task.

Why this module exists. Accuracy is the wrong headline for this task and
cannot be rescued by tuning. The test split is 1,696 engaged clips against 88
not-engaged, so answering "engaged" to everything scores 0.9507 while detecting
nothing at all. Any model that actually finds disengaged students must predict
the minority class sometimes, and every such prediction that is wrong costs
accuracy -- so a *better* detector usually reports a *lower* accuracy than the
constant predictor. Reporting accuracy alone on a 95/5 split tells the reader
nothing about whether the model works.

What this module reports instead:

*   **ROC-AUC** -- the probability that a randomly chosen not-engaged clip is
    scored as less engaged than a randomly chosen engaged one. Threshold-free,
    unaffected by the class ratio, and 0.5 means no signal. This is the headline.
*   **Average precision** -- area under the precision-recall curve, which is
    the right summary when the positive class is rare, because it does not
    reward the vast true-negative mass the way ROC does.
*   **An operating point chosen on the validation split** and then applied
    unchanged to test. Picking a threshold is a real modelling decision and
    hiding it inside argmax (which is just a 0.5 threshold, arbitrary on
    imbalanced data) does not make it go away. The threshold is tuned on
    validation only; ``test`` labels never inform it, and the record says so.

Accuracy is still reported, always next to the majority-class baseline, so the
number is legible rather than flattering.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score,
    roc_curve,
)

# The positive class for detection purposes is "not engaged": that is the
# minority, and the one worth finding. Class index 0 under the merge
# [[0, 1], [2, 3]], so the detector score is P(class 0).
NOT_ENGAGED = 0

OBJECTIVES = ("balanced_accuracy", "f1", "youden")


def detection_scores(probs) -> np.ndarray:
    """P(not engaged) from a [N, 2] probability matrix."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(f"expected [N, 2] probabilities, got {probs.shape}")
    return probs[:, NOT_ENGAGED]


def _positives(y_true) -> np.ndarray:
    """1 where the clip is not engaged."""
    return (np.asarray(y_true).astype(int) == NOT_ENGAGED).astype(int)


def ranking_metrics(y_true, scores) -> dict:
    """Threshold-free summaries. Returns NaN-free placeholders if degenerate."""
    y = _positives(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "roc_auc": None, "average_precision": None,
            "n_positive": n_pos, "n_negative": n_neg,
            "note": "only one class present; ranking metrics are undefined",
        }
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "positive_rate": float(y.mean()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "roc_auc_note": ("probability that a not-engaged clip is scored above an "
                         "engaged one; 0.5 is no signal and is unaffected by the "
                         "class ratio"),
        "average_precision_note": ("area under precision-recall; the baseline is the "
                                   "positive rate, not 0.5"),
    }


def operating_point(y_true, scores, threshold: float) -> dict:
    """Everything that follows from one threshold on the detector score."""
    y = _positives(y_true)
    pred = (np.asarray(scores, dtype=np.float64) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred))
        if y.min() != y.max() else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall_not_engaged": float(recall),
        "precision_not_engaged": float(precision),
        "specificity_engaged": float(specificity),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "predicted_not_engaged": int(pred.sum()),
        "actually_not_engaged": int(y.sum()),
    }


def choose_threshold(y_true, scores, objective: str = "balanced_accuracy") -> dict:
    """Pick a threshold on THIS split, to be applied to another one.

    Sweeps every threshold the ROC curve actually distinguishes rather than a
    fixed grid, so the result does not depend on grid resolution. The caller is
    responsible for only ever calling this on validation data; what it returns
    carries ``chosen_on`` so the artifact states where the number came from.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    y = _positives(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    if y.min() == y.max():
        return {"threshold": 0.5, "objective": objective,
                "note": "only one class present; falling back to 0.5"}

    fpr, tpr, thresholds = roc_curve(y, scores)
    # roc_curve prepends an infinite threshold that predicts nothing positive.
    thresholds = thresholds[np.isfinite(thresholds)]
    candidates = np.unique(np.concatenate([thresholds, [0.5]]))

    best = None
    for t in candidates:
        point = operating_point(y_true, scores, t)
        value = (point["balanced_accuracy"] if objective == "balanced_accuracy"
                 else point["f1"] if objective == "f1"
                 else point["recall_not_engaged"] + point["specificity_engaged"] - 1.0)
        if best is None or value > best[0]:
            best = (value, t, point)
    value, t, point = best
    return {"threshold": float(t), "objective": objective,
            "objective_value": float(value), "point": point}


def evaluate_detector(y_true, probs, threshold: float | None = None,
                      tune_on: tuple | None = None, objective: str = "balanced_accuracy"
                      ) -> dict:
    """The full report for one split.

    ``threshold`` applies a fixed operating point (the one tuned elsewhere).
    ``tune_on`` is an ``(y_true, probs)`` pair -- normally the validation split --
    from which to choose one. Passing neither reports argmax only.
    """
    scores = detection_scores(probs)
    out = {"ranking": ranking_metrics(y_true, scores),
           "argmax": operating_point(y_true, scores, 0.5)}
    out["argmax"]["note"] = "the default 0.5 threshold, i.e. plain argmax"

    chosen = None
    if tune_on is not None:
        v_true, v_probs = tune_on
        chosen = choose_threshold(v_true, detection_scores(v_probs), objective)
        chosen["chosen_on"] = "validation split"
    elif threshold is not None:
        chosen = {"threshold": float(threshold), "objective": objective,
                  "chosen_on": "supplied by the caller"}

    if chosen is not None:
        out["tuned"] = operating_point(y_true, scores, chosen["threshold"])
        out["tuned"]["threshold_chosen_on"] = chosen["chosen_on"]
        out["tuned"]["threshold_objective"] = chosen["objective"]
        out["tuned"]["note"] = (
            "threshold selected to maximise " + chosen["objective"] + " on the "
            + chosen["chosen_on"] + "; the evaluated split's labels did not "
            "inform it")
    majority = max(float((_positives(y_true) == 1).mean()),
                   float((_positives(y_true) == 0).mean()))
    out["baselines"] = {
        "majority_class_accuracy": majority,
        "balanced_accuracy": 0.5,
        "roc_auc": 0.5,
        "average_precision": float(_positives(y_true).mean()),
        "note": ("a constant 'engaged' predictor achieves majority_class_accuracy "
                 "while detecting nothing; accuracy above it is the only accuracy "
                 "claim that means anything here"),
    }
    return out
