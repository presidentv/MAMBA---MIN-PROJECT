"""Loss functions and class weighting.

DAiSEE engagement is strongly imbalanced, so weighted cross-entropy is supported.
Class weights are computed from the **training split only** (spec section 20,
Rule 7/8) -- a weight derived from validation or test counts would leak the
evaluation distribution into training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def class_weights_from_counts(counts: dict[int, int], num_classes: int,
                              scheme: str = "inverse_frequency",
                              beta: float = 0.999) -> torch.Tensor:
    """Return a [num_classes] weight tensor.

    ``inverse_frequency``  w_c = N / (K * n_c), normalised to mean 1.
    ``effective_number``   w_c = (1 - beta) / (1 - beta^{n_c}), from Cui et al.,
                           "Class-Balanced Loss Based on Effective Number of
                           Samples" (CVPR 2019). Less aggressive than pure
                           inverse frequency when a class is very rare.
    ``none``               all ones.

    A class absent from the training split gets weight 0 and a stated warning
    rather than an infinite weight: the model cannot learn a class it never sees,
    and an infinity here would produce NaN on the first backward pass.
    """
    n = np.array([counts.get(c, 0) for c in range(num_classes)], dtype=np.float64)
    if scheme == "none":
        return torch.ones(num_classes, dtype=torch.float32)

    total = n.sum()
    if total <= 0:
        raise ValueError("cannot compute class weights from empty counts")

    present = n > 0
    w = np.zeros(num_classes, dtype=np.float64)
    if scheme == "inverse_frequency":
        w[present] = total / (present.sum() * n[present])
    elif scheme == "effective_number":
        eff = (1.0 - np.power(beta, n[present])) / (1.0 - beta)
        w[present] = 1.0 / eff
    else:
        raise ValueError(f"unknown class weighting scheme: {scheme!r}")

    if w[present].mean() > 0:
        w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


class WeightedCrossEntropy(nn.Module):
    """Cross-entropy with optional class weights and label smoothing."""

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        if weight is not None:
            self.register_buffer("class_weight", weight)
        else:
            self.class_weight = None
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weight
        if weight is not None:
            weight = weight.to(logits.device, logits.dtype)
        return nn.functional.cross_entropy(
            logits, targets, weight=weight, label_smoothing=self.label_smoothing)


def build_loss(cfg_training: dict, train_counts: dict[int, int],
               num_classes: int) -> tuple[WeightedCrossEntropy, dict]:
    scheme = cfg_training.get("class_weighting", "none")
    weights = class_weights_from_counts(train_counts, num_classes, scheme)
    missing = [c for c in range(num_classes) if train_counts.get(c, 0) == 0]
    info = {
        "scheme": scheme,
        "weights": weights.tolist(),
        "train_counts": {str(k): int(v) for k, v in sorted(train_counts.items())},
        "classes_absent_from_train": missing,
        "label_smoothing": float(cfg_training.get("label_smoothing", 0.0)),
    }
    loss = WeightedCrossEntropy(
        weight=weights if scheme != "none" else None,
        label_smoothing=float(cfg_training.get("label_smoothing", 0.0)),
    )
    return loss, info
