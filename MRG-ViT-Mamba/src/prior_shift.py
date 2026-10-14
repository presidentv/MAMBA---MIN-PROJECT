"""Correcting a classifier trained under one class prior for use under another.

The problem. Imbalance-aware sampling and class-weighted loss both change the
class distribution the model is optimised against. ``sampling: sqrt`` on the
full DAiSEE training split turns a natural prior of

    0.006  0.040  0.489  0.466          (34 / 213 / 2617 / 2494)

into an effective training prior of

    0.044  0.121  0.423  0.413

which is the point -- the rare classes get enough gradient to be learned at all.
But the test split still arrives at the natural prior, so the decision boundary
is now calibrated for a distribution that does not occur at evaluation time. The
model over-predicts the rare classes, and accuracy falls even where macro-F1
rises. The sampler buys recall on the tail and pays for it in accuracy.

The correction. For a model whose softmax estimates p_train(y | x), Bayes gives
the estimate under a different prior by reweighting:

    p_target(y | x)  ∝  p_train(y | x) * p_target(y) / p_train(y)

which in logit space is a per-class additive shift computed once:

    adjusted_logit_c = logit_c + tau * (log p_target(c) - log p_train(c))

This is the "logit adjustment" of Menon et al. (2021), applied post hoc. It
costs one vector addition at inference, changes no weights, and is exact under
the assumption that only the prior differs between training and target -- i.e.
that p(x | y) is unchanged, which is what sampling and reweighting do.

Choosing the target prior, and why it is not cheating. The default target is the
*natural prior of the training split*: the real class distribution of the data
the model was fitted on, before the sampler distorted it. That uses no
information from the validation or test split at all, and it undoes exactly the
distortion the sampler introduced.

``evaluation.reference_prior`` can name a different target, which is the right
option when training on a subset: a subset's own natural prior is not the
dataset's prior, so the published full-DAiSEE distribution should be used
instead. A published dataset-level class distribution is a documented property
of the corpus, not a measurement of the held-out split -- but a prior read off
the test split's own labels *would* be using the test set, so that is never the
default and any run that does it must say so. ``applied_to`` in the returned
record states which was used.
"""

from __future__ import annotations

import numpy as np

# Below this share a class is treated as unobserved rather than rare: log of a
# near-zero prior produces a shift large enough to dominate every logit, which
# turns a correction into a constant prediction.
MIN_PRIOR = 1e-6


def prior_from_counts(counts, num_classes: int) -> np.ndarray:
    """Normalise a {class: count} mapping (or sequence) into a prior vector."""
    if isinstance(counts, dict):
        vec = np.array([float(counts.get(c, counts.get(str(c), 0)))
                        for c in range(num_classes)], dtype=np.float64)
    else:
        vec = np.asarray(counts, dtype=np.float64)
        if vec.size != num_classes:
            raise ValueError(f"expected {num_classes} counts, got {vec.size}")
    total = vec.sum()
    if total <= 0:
        raise ValueError("class counts sum to zero; cannot form a prior")
    return vec / total


def logit_adjustment(train_prior, target_prior, num_classes: int,
                     tau: float = 1.0) -> np.ndarray:
    """The additive per-class shift taking train_prior to target_prior."""
    p_tr = np.clip(prior_from_counts(train_prior, num_classes), MIN_PRIOR, None)
    p_ta = np.clip(prior_from_counts(target_prior, num_classes), MIN_PRIOR, None)
    return float(tau) * (np.log(p_ta) - np.log(p_tr))


def apply_adjustment(logits, adjustment) -> np.ndarray:
    """Shift logits (or log-probabilities) by a per-class vector."""
    logits = np.asarray(logits, dtype=np.float64)
    adjustment = np.asarray(adjustment, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != adjustment.size:
        raise ValueError(
            f"logits {logits.shape} do not match a {adjustment.size}-class adjustment")
    return logits + adjustment[None, :]


def resolve_target_prior(cfg, num_classes: int, natural_counts) -> tuple[np.ndarray, str]:
    """(prior, description) for the distribution the model should be calibrated to."""
    ev = cfg["evaluation"] if "evaluation" in cfg else {}
    ref = ev.get("reference_prior")
    if ref is None:
        return (prior_from_counts(natural_counts, num_classes),
                "natural class distribution of the training split")
    prior = prior_from_counts(ref, num_classes)
    return prior, "evaluation.reference_prior from the config"


def adjust_predictions(probs_or_logits, effective_counts, num_classes: int,
                       cfg=None, natural_counts=None, tau: float = 1.0,
                       in_log_space: bool = False) -> dict:
    """Re-decide every clip under the target prior.

    ``effective_counts`` is the distribution the model was actually optimised
    against: the sampler's expected per-class draws when sampling is on, the
    natural counts when it is off. Returns the new predictions together with a
    record of exactly what was applied, which belongs in the artifact.
    """
    scores = np.asarray(probs_or_logits, dtype=np.float64)
    if not in_log_space:
        scores = np.log(np.clip(scores, 1e-12, None))
    natural = natural_counts if natural_counts is not None else effective_counts
    target, target_desc = resolve_target_prior(cfg or {}, num_classes, natural)
    train_prior = prior_from_counts(effective_counts, num_classes)
    shift = logit_adjustment(train_prior, target, num_classes, tau)
    adjusted = apply_adjustment(scores, shift)
    return {
        "predictions": adjusted.argmax(axis=1).astype(np.int64),
        "adjusted_logits": adjusted,
        "record": {
            "applied": True,
            "tau": float(tau),
            "effective_train_prior": [round(float(p), 6) for p in train_prior],
            "target_prior": [round(float(p), 6) for p in target],
            "applied_to": target_desc,
            "logit_shift": [round(float(s), 6) for s in shift],
            "note": ("post-hoc logit adjustment (Menon et al. 2021); corrects the "
                     "prior distortion introduced by weighted sampling and class-"
                     "weighted loss. No weights were changed and the test labels "
                     "were not used to derive the shift."),
        },
    }


def effective_train_counts(natural_counts, sampling_info, num_classes: int) -> dict[int, int]:
    """What the sampler actually fed the model, as counts on the same scale.

    ``sampling_info`` is what ``sampling.build_train_sampler`` recorded. With
    sampling off, or an info block from an older run that lacks the shares, the
    natural counts are returned unchanged -- which is correct, not a fallback.
    """
    natural = {int(k): int(v) for k, v in dict(natural_counts).items()}
    total = sum(natural.values())
    shares = (sampling_info or {}).get("expected_share")
    if not shares or total <= 0:
        return natural
    return {c: int(round(float(shares.get(str(c), shares.get(c, 0.0))) * total))
            for c in range(num_classes)}
