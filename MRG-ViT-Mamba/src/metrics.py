"""Evaluation metrics.

Accuracy alone is not reported anywhere (spec section 7). Macro-F1 is the
headline number because on a distribution this skewed, accuracy and weighted-F1
are both dominated by the majority class and can look healthy while a minority
class is never predicted at all.

    Acc-4        = correct / total
    weighted F1  = sum_c (n_c / N) * F1_c
    macro F1     = (1/K) * sum_c F1_c
    balanced acc = (1/K') * sum_c recall_c      (K' = classes with support)

Balanced accuracy is the number a hand-balanced test split would have produced,
computed on the full split instead. On DAiSEE's test distribution plain accuracy
starts at 0.494 for a model that always answers "High", so it mostly measures
how well the class prior was guessed; balanced accuracy weights every class
equally and is not inflated that way. Reporting both, from all 1,784 clips, says
more than either alone and keeps the comparison with published work intact --
subsampling the test set to balance it would discard 85% of the evidence and
make the result incomparable.

A class with no support in the evaluated split has an undefined F1. sklearn
returns 0.0 there; this module records *which* classes were undefined so a
macro-F1 depressed by an absent class is never mistaken for poor performance on
a class that was actually present.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)

DEFAULT_CLASS_NAMES = ("Very Low", "Low", "High", "Very High")


def resolve_names(class_names, num_classes: int) -> tuple[str, ...]:
    """Names for ``num_classes`` classes, whatever the caller supplied.

    Callers that merge classes pass their own names; callers that do not pass
    None and get the DAiSEE level names. A length mismatch raises here rather
    than becoming an IndexError inside the per-class loop or, worse, a report
    whose rows are labelled with the wrong class.
    """
    if class_names is None:
        if num_classes <= len(DEFAULT_CLASS_NAMES):
            return DEFAULT_CLASS_NAMES[:num_classes]
        return tuple(f"class {i}" for i in range(num_classes))
    names = tuple(str(n) for n in class_names)
    if len(names) != num_classes:
        raise ValueError(
            f"got {len(names)} class names for {num_classes} classes: {names}")
    return names


def compute_metrics(y_true, y_pred, num_classes: int = 4,
                    class_names: tuple[str, ...] | None = None) -> dict:
    class_names = resolve_names(class_names, num_classes)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)

    absent = [int(c) for c in labels if support[c] == 0]
    never_predicted = [int(c) for c in labels if (y_pred == c).sum() == 0 and support[c] > 0]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # sklearn averages recall over the classes present in y_true, so an absent
    # class does not drag this to zero the way it does macro-F1.
    balanced = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                      zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "per_class": {
            class_names[c]: {
                "index": int(c),
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
            for c in labels
        },
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(y_true)),
        "class_prior": [round(float(support[c] / max(len(y_true), 1)), 6) for c in labels],
        "majority_class_accuracy": (float(support.max() / len(y_true))
                                    if len(y_true) else 0.0),
        "classes_absent_from_split": absent,
        "classes_present_but_never_predicted": never_predicted,
        "balanced_accuracy_note": (
            "mean per-class recall over the classes present in this split; equals "
            "the accuracy a class-balanced version of this split would give, "
            "without discarding any clips"),
        "macro_f1_note": (
            f"macro-F1 averages over all {num_classes} classes including "
            f"{len(absent)} with zero support in this split"
            if absent else
            f"all {num_classes} classes have support in this split"
        ),
    }


def format_report(y_true, y_pred, num_classes: int = 4,
                  class_names: tuple[str, ...] | None = None) -> str:
    class_names = resolve_names(class_names, num_classes)
    return classification_report(
        np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
        labels=list(range(num_classes)), target_names=list(class_names),
        digits=4, zero_division=0)


def plot_confusion_matrix(cm, class_names, out_path, title: str = "Confusion matrix",
                          normalize: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    display = cm.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=150)
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max() if display.max() > 0 else 1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = display.max() / 2.0 if display.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if display[i, j] > thresh else "black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
