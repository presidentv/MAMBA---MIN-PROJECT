"""Class-imbalance sampling for the training loader.

DAiSEE engagement is severely skewed -- in the full release the rarest class
holds tens of clips and the most common holds thousands. There are three ways
to respond, and only one of them is right here:

*   **Undersample the common classes to balance.** Tempting, and wrong: the
    rarest class would cap the training set at 4 x its own size. On DAiSEE that
    is roughly 136 clips, fewer than the 120 that already overfit badly. It
    trades a large problem (too little data) for a smaller one.

*   **Oversample the rare classes.** Keeps every clip. Repetition creates no new
    information -- 34 clips drawn 17 times are still 34 clips -- but combined
    with augmentation it stops the gradient being dominated by the majority.

*   **Leave the distribution alone** and let the weighted loss carry it.

``sqrt`` is the middle setting and the sensible default for a skew this large:
sampling probability proportional to 1/sqrt(n_c) pulls the rare classes up
without repeating a handful of clips dozens of times per epoch.

Weights are computed from the *measured* training labels, never from a
hardcoded distribution, so this adapts to whatever subset is actually present.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import WeightedRandomSampler

MODES = ("natural", "sqrt", "balanced")


def class_sample_weights(labels, num_classes: int, mode: str = "sqrt",
                         max_ratio: float | None = None) -> tuple[np.ndarray, dict]:
    """Per-class sampling weights and a record of what they imply.

    ``natural``   every clip equally likely (the ordinary shuffle).
    ``sqrt``      p(class) proportional to 1/sqrt(n_c).
    ``balanced``  p(class) proportional to 1/n_c; every class equally likely.

    ``max_ratio`` caps how much more often the rarest class is drawn than the
    most common one, which keeps a 77:1 corpus from becoming 77:1 repetition.
    """
    if mode not in MODES:
        raise ValueError(f"sampling mode must be one of {MODES}, got {mode!r}")
    counts = np.array([int((np.asarray(labels) == c).sum()) for c in range(num_classes)],
                      dtype=np.float64)
    present = counts > 0
    w = np.zeros(num_classes, dtype=np.float64)
    if mode == "natural":
        w[present] = 1.0
    elif mode == "sqrt":
        w[present] = 1.0 / np.sqrt(counts[present])
    else:
        w[present] = 1.0 / counts[present]

    if present.any():
        w[present] /= w[present].min()        # rarest-relative, min weight 1
    if max_ratio is not None and present.any():
        w[present] = np.minimum(w[present], float(max_ratio))

    expected = counts * w
    info = {
        "mode": mode,
        "max_ratio": max_ratio,
        "counts": {str(c): int(counts[c]) for c in range(num_classes)},
        "class_weight": {str(c): round(float(w[c]), 4) for c in range(num_classes)},
        # Share of a drawn epoch each class should occupy.
        "expected_share": {
            str(c): round(float(expected[c] / expected.sum()), 4) if expected.sum() else 0.0
            for c in range(num_classes)},
        "natural_share": {
            str(c): round(float(counts[c] / counts.sum()), 4) if counts.sum() else 0.0
            for c in range(num_classes)},
        "max_repetition_of_rarest": (
            round(float(w[present].max()), 2) if present.any() else None),
    }
    return w, info


def build_train_sampler(labels, num_classes: int, cfg_training: dict):
    """A WeightedRandomSampler for the training split, or None for plain shuffling.

    Returns (sampler, info). ``None`` means the caller should keep
    ``shuffle=True``; the two are mutually exclusive in a DataLoader.
    """
    mode = str(cfg_training.get("sampling", "natural")).lower()
    max_ratio = cfg_training.get("sampling_max_ratio")
    w, info = class_sample_weights(labels, num_classes, mode,
                                   float(max_ratio) if max_ratio else None)
    if mode == "natural":
        info["sampler"] = "none (plain shuffle)"
        return None, info

    labels = np.asarray(labels)
    per_item = w[labels]
    # One draw per clip per epoch, with replacement: epoch length is unchanged,
    # only the class mix within it moves.
    sampler = WeightedRandomSampler(weights=per_item.tolist(),
                                    num_samples=len(labels), replacement=True)
    info["sampler"] = f"WeightedRandomSampler({mode}), {len(labels)} draws/epoch"
    return sampler, info
