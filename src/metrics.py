"""Evaluation metrics.

Macro-F1 and the full confusion matrix are the headline numbers everywhere.
Accuracy is computed but never reported on its own: on DAiSEE's usual label
distribution a majority-class predictor already reaches roughly 50%, so a bare
accuracy figure carries almost no information.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 4,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Return the full metric bundle for one model on one split."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(num_classes))

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=num_classes)

    metrics: Dict[str, object] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {str(i): float(v) for i, v in enumerate(per_class_f1)},
        "support": {str(i): int(v) for i, v in enumerate(support)},
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
        "n": int(len(y_true)),
        # Ordinal-aware: quadratic-weighted kappa penalises a 0-vs-3 error far
        # more than a 2-vs-3 error, which plain F1 treats identically.
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        )
        if len(np.unique(y_true)) > 1
        else float("nan"),
        # Mean absolute error over the ordinal scale: the natural "how far off"
        # measure when the classes are ordered.
        "mean_absolute_error": float(np.mean(np.abs(y_true - y_pred))),
        "classes_absent_from_truth": [i for i in labels if support[i] == 0],
        "classes_never_predicted": [
            i for i in labels if not np.any(y_pred == i)
        ],
    }
    if y_prob is not None:
        metrics["mean_max_probability"] = float(np.mean(np.max(np.asarray(y_prob), axis=1)))
    return metrics


def format_confusion_matrix(matrix: Sequence[Sequence[int]], num_classes: int = 4) -> str:
    """Render a confusion matrix as fixed-width text for the console and log."""
    matrix = np.asarray(matrix, dtype=int)
    header = "        " + "".join(f"pred{i:>4}" for i in range(num_classes))
    lines: List[str] = [header, "        " + "-" * (8 * num_classes)]
    for i in range(num_classes):
        row = "".join(f"{int(v):>8}" for v in matrix[i])
        lines.append(f"true{i:>3} |{row}")
    return "\n".join(lines)


def text_report(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 4) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0, digits=3
    )
