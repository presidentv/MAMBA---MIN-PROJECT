"""Step 5 -- the required baselines.

Every headline number must be read against these. The two that matter most:

  * ``majority`` sets the floor. On DAiSEE's usual distribution it reaches ~50%
    accuracy while its macro-F1 is ~0.17, which is exactly why accuracy alone is
    never reported.
  * ``logreg_meanpool`` is the "is the transformer earning its keep?" control. A
    2-layer transformer that cannot beat logistic regression on mean-pooled
    features has not learned anything the pooling did not already give away.

The three neural baselines (gaze-only, affect-only, late fusion) are trained by
``train.py`` using the same loop as the fusion model, so any difference between
them is architectural rather than an artefact of different training regimes.

Run standalone (majority + logistic regression only, no torch training needed):
    python src/baselines.py --output-root outputs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets import EngagementClipDataset
from metrics import evaluate

logger = logging.getLogger("baselines")


# --------------------------------------------------------------------------- #
# 1. Majority class
# --------------------------------------------------------------------------- #
def majority_baseline(
    train_dataset: EngagementClipDataset,
    eval_datasets: Dict[str, EngagementClipDataset],
    num_classes: int = 4,
) -> Dict[str, object]:
    """Always predict the most frequent TRAIN class."""
    train_labels = train_dataset.labels()
    majority_class = int(np.bincount(train_labels, minlength=num_classes).argmax())
    logger.info("Majority baseline predicts class %d for everything", majority_class)

    results: Dict[str, object] = {"majority_class": majority_class, "splits": {}}
    for split, dataset in eval_datasets.items():
        y_true = dataset.labels()
        y_pred = np.full_like(y_true, majority_class)
        results["splits"][split] = evaluate(y_true, y_pred, num_classes)
    return results


# --------------------------------------------------------------------------- #
# 2. Logistic regression on mean-pooled features
# --------------------------------------------------------------------------- #
def mean_pool_features(dataset: EngagementClipDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenated clip-level means of the gaze and affect sequences.

    Pooling respects the real length of each clip rather than averaging over the
    padded region, so a short clip is not diluted toward zero.
    """
    features: List[np.ndarray] = []
    for sample in dataset.samples:
        length = min(sample.gaze.shape[0], dataset.max_len)
        gaze_mean = sample.gaze[:length].mean(axis=0)
        affect_mean = sample.affect[:length].mean(axis=0)
        features.append(np.concatenate([gaze_mean, affect_mean]))
    return np.vstack(features), dataset.labels()


def logistic_regression_baseline(
    train_dataset: EngagementClipDataset,
    eval_datasets: Dict[str, EngagementClipDataset],
    num_classes: int = 4,
    seed: int = 42,
) -> Dict[str, object]:
    """Multinomial logistic regression over concatenated mean-pooled features."""
    x_train, y_train = mean_pool_features(train_dataset)
    logger.info("Logistic regression on mean-pooled features: X=%s", x_train.shape)

    # Strong L2 (small C) because there are more features than clips; balanced
    # class weights because the label distribution is skewed.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000, C=0.1, class_weight="balanced", random_state=seed,
        ),
    )
    model.fit(x_train, y_train)

    results: Dict[str, object] = {"n_features": int(x_train.shape[1]), "splits": {}}
    for split, dataset in eval_datasets.items():
        x_eval, y_true = mean_pool_features(dataset)
        y_pred = model.predict(x_eval)
        probabilities = model.predict_proba(x_eval)
        # predict_proba only spans the classes seen in training; re-expand to the
        # full K columns so downstream shapes are always (N, num_classes).
        full = np.zeros((len(y_true), num_classes))
        for column, class_index in enumerate(model.classes_):
            full[:, int(class_index)] = probabilities[:, column]
        results["splits"][split] = evaluate(y_true, y_pred, num_classes, full)
        results.setdefault("probabilities", {})[split] = full.tolist()
    return results


# --------------------------------------------------------------------------- #
# 5. Naive late fusion
# --------------------------------------------------------------------------- #
def late_fusion(
    gaze_probabilities: Dict[str, np.ndarray],
    affect_probabilities: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    num_classes: int = 4,
    weight: float = 0.5,
) -> Dict[str, object]:
    """Average the two single-modality probability outputs.

    This is the control the cross-attention model has to beat: if simply
    averaging two independently trained branches matches the fusion model, the
    cross-attention block is not contributing anything.
    """
    results: Dict[str, object] = {"weight_gaze": weight, "splits": {}}
    for split in labels:
        if split not in gaze_probabilities or split not in affect_probabilities:
            continue
        gaze = np.asarray(gaze_probabilities[split])
        affect = np.asarray(affect_probabilities[split])
        if gaze.shape != affect.shape:
            logger.error(
                "Late fusion shape mismatch on %s: %s vs %s", split, gaze.shape, affect.shape
            )
            continue
        averaged = weight * gaze + (1.0 - weight) * affect
        y_pred = averaged.argmax(axis=1)
        results["splits"][split] = evaluate(labels[split], y_pred, num_classes, averaged)
    return results


# --------------------------------------------------------------------------- #
# standalone entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import load_experiment_config
    from datasets import build_dataloaders, summarise_class_distribution

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_experiment_config(args.config)
    _, datasets, _ = build_dataloaders(args.output_root, cfg.data, cfg.splits)

    table, warnings = summarise_class_distribution(datasets, cfg.data.num_classes)
    print("\nClass distribution:\n", table.to_string(), "\n")
    for warning in warnings:
        logger.warning(warning)

    output = {
        "majority": majority_baseline(datasets["Train"], datasets, cfg.data.num_classes),
        "logreg_meanpool": logistic_regression_baseline(
            datasets["Train"], datasets, cfg.data.num_classes, cfg.train.seed
        ),
    }
    for name, result in output.items():
        for split, metrics in result["splits"].items():
            logger.info(
                "%-18s %-11s macro-F1 %.3f  acc %.3f", name, split,
                metrics["macro_f1"], metrics["accuracy"],
            )

    target = Path(args.output_root) / "results" / "baselines_standalone.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
