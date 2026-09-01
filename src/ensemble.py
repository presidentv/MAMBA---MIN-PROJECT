"""Clip-level model ensemble -- the right tool class for this sample size.

WHY THIS EXISTS
---------------
Every experiment in this project points the same way. The 1.26 M-parameter deep
model reaches 0.376 test macro-F1; plain logistic regression on mean-pooled
features reaches 0.349. Removing the learned appearance branch entirely once
*improved* the score. SHAP attributes 4.5% of the decision to it. With 36
training clips, a deep network is simply the wrong capacity.

What a data-starved problem wants is **external knowledge it does not have to
learn**, plus **shallow models that cannot overfit 36 examples**. That is exactly
what this module assembles:

  frozen pretrained features   HSEmotion (EfficientNet-B0, ImageNet -> AffectNet)
                               contributes a 1280-d embedding and an 8-class
                               emotion distribution that were learned from
                               hundreds of thousands of faces, not from 36 clips
  hand-engineered features     the gaze / blink / head-pose descriptors, which
                               the ablation showed carry nearly all the signal
  shallow learners             regularised linear, tree ensembles, and an
                               ordinal regressor, soft-voted

The ordinal regressor deserves a note: engagement is ordered 0-3, so predicting
a *continuous* value and thresholding it respects that ordering while fitting
only a handful of parameters. On small ordinal problems this routinely beats
multi-class classification, which throws the ordering away.

Usage:
    python src/ensemble.py --output-root outputs_ungated
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import evaluate, format_confusion_matrix  # noqa: E402

logger = logging.getLogger("ensemble")

LABEL = "Engagement"
META_COLUMNS = {"ClipID", "SubjectID", "split", "gaze_file", "affect_file",
                "dominant_emotion", "Boredom", "Engagement", "Confusion", "Frustration"}


def load_clip_features(
    output_root: Path, use_affect: bool = True, use_embeddings: bool = False
) -> pd.DataFrame:
    """Join the clip-level feature tables into one frame, one row per clip."""
    output_root = Path(output_root)
    gaze_path = output_root / "features" / "gaze_clip_features.csv"
    if not gaze_path.is_file():
        raise SystemExit(f"Missing {gaze_path}. Run src/features.py first.")
    frame = pd.read_csv(gaze_path)

    affect_path = output_root / "features" / "affect_clip_features.csv"
    if use_affect and affect_path.is_file():
        affect = pd.read_csv(affect_path)
        drop = [c for c in ("SubjectID", "split", "affect_file", "n_frames") if c in affect.columns]
        frame = frame.merge(affect.drop(columns=drop), on="ClipID", how="left", validate="1:1")
        logger.info("Joined affect features (+%d columns)", affect.shape[1] - len(drop) - 1)
    elif use_affect:
        logger.warning("No affect features at %s; continuing without them", affect_path)

    if use_embeddings:
        # The 1280-d per-frame embedding, averaged per clip. Off by default: 1280
        # dimensions over 36 training clips is memorisation, not learning.
        rows = []
        for path in sorted((output_root / "features" / "affect").rglob("affect.parquet")):
            table = pd.read_parquet(path)
            columns = [c for c in table.columns if c.startswith("emb_")]
            if not columns:
                continue
            means = table[columns].mean()
            means["ClipID"] = table["ClipID"].iloc[0]
            rows.append(means)
        if rows:
            frame = frame.merge(pd.DataFrame(rows), on="ClipID", how="left", validate="1:1")
            logger.info("Joined %d embedding dimensions", len(rows[0]) - 1)

    return frame


def split_matrices(
    frame: pd.DataFrame, target: str = LABEL
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Return per-split (X, y) plus the feature names, with no leakage."""
    feature_names = [
        c for c in frame.columns
        if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])
    ]
    frame = frame.dropna(subset=[target])
    X, y = {}, {}
    for split, group in frame.groupby("split"):
        matrix = group[feature_names].to_numpy(dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        X[split] = matrix
        y[split] = group[target].to_numpy(dtype=int)
    return X, y, feature_names


class OrdinalRegressor:
    """Regress the ordinal label as a continuous value, then threshold.

    Fits |classes| - 1 cut points on the training predictions rather than using
    the naive 0.5/1.5/2.5 boundaries, so the thresholds adapt to a skewed label
    distribution instead of assuming a balanced one.
    """

    def __init__(self, base=None, num_classes: int = 4) -> None:
        from sklearn.linear_model import RidgeCV

        self.base = base if base is not None else RidgeCV(alphas=np.logspace(-2, 3, 20))
        self.num_classes = num_classes
        self.thresholds: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalRegressor":
        self.base.fit(X, y)
        predicted = self.base.predict(X)
        # Cut points at the empirical quantiles of the training labels, so the
        # predicted distribution is matched to the observed one.
        quantiles = [(y <= k).mean() for k in range(self.num_classes - 1)]
        self.thresholds = np.quantile(predicted, np.clip(quantiles, 0.0, 1.0))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict(X)
        return np.searchsorted(self.thresholds, raw).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft scores from distance to each class centre, for the soft vote."""
        raw = np.clip(self.base.predict(X), -1, self.num_classes)
        centres = np.arange(self.num_classes, dtype=float)
        logits = -np.abs(raw[:, None] - centres[None, :])
        exponentiated = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def select_features(
    X_train: np.ndarray, y_train: np.ndarray, k: int, seed: int = 42
) -> np.ndarray:
    """Indices of the k most informative features, chosen on TRAIN only.

    Mutual information rather than a model's own importances, so the selection
    does not privilege whichever learner is fitted afterwards. Returns all
    indices when k >= the feature count.
    """
    from sklearn.feature_selection import mutual_info_classif

    if k <= 0 or k >= X_train.shape[1]:
        return np.arange(X_train.shape[1])
    scores = mutual_info_classif(X_train, y_train, random_state=seed)
    return np.argsort(scores)[::-1][:k]


def build_models(seed: int = 42) -> Dict[str, object]:
    global _OrdinalWrapper
    if _OrdinalWrapper is None:
        _OrdinalWrapper = _ordinal_wrapper_class()
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        # Strong L2: more features than clips, so the prior must do the work.
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced",
                               random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "ordinal_ridge": make_pipeline(StandardScaler(), _OrdinalWrapper()),
    }


def _ordinal_wrapper_class():
    """Build the Pipeline adapter lazily.

    sklearn >= 1.6 resolves estimator tags through ``BaseEstimator``, so a plain
    duck-typed object raises inside a Pipeline. Defining the class here keeps the
    sklearn import local to where it is needed.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _OrdinalWrapperImpl(ClassifierMixin, BaseEstimator):
        """Adapter so OrdinalRegressor can sit inside a sklearn Pipeline."""

        def __init__(self, num_classes: int = 4) -> None:
            self.num_classes = num_classes

        def fit(self, X, y):
            self.classes_ = np.arange(self.num_classes)
            self.model_ = OrdinalRegressor(num_classes=self.num_classes).fit(X, y)
            return self

        def predict(self, X):
            return self.model_.predict(X)

        def predict_proba(self, X):
            return self.model_.predict_proba(X)

    return _OrdinalWrapperImpl


_OrdinalWrapper = None  # populated on first use by build_models


def run(
    output_root: Path, use_affect: bool = True, use_embeddings: bool = False,
    seed: int = 42, num_classes: int = 4, top_k: int = 0, quiet: bool = False,
    frame: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    if frame is None:
        frame = load_clip_features(output_root, use_affect, use_embeddings)
    X, y, feature_names = split_matrices(frame)

    if "Train" not in X:
        raise SystemExit("No Train split found.")

    if top_k:
        keep = select_features(X["Train"], y["Train"], top_k, seed)
        X = {split: matrix[:, keep] for split, matrix in X.items()}
        feature_names = [feature_names[i] for i in keep]
    if not quiet:
        logger.info("Features: %d | clips per split: %s",
                    len(feature_names), {k: len(v) for k, v in y.items()})

    models = build_models(seed)
    results: Dict[str, object] = {"n_features": len(feature_names),
                                  "feature_names": feature_names, "models": {}}
    probabilities: Dict[str, Dict[str, np.ndarray]] = {}

    for name, model in models.items():
        model.fit(X["Train"], y["Train"])
        results["models"][name] = {"splits": {}}
        probabilities[name] = {}
        for split in X:
            predicted = model.predict(X[split])
            raw = model.predict_proba(X[split])
            # Re-expand to the full K columns; a model may not have seen a class.
            full = np.zeros((len(y[split]), num_classes))
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "steps"):
                classes = getattr(model.steps[-1][1], "classes_", None)
            if classes is not None and raw.shape[1] == len(classes):
                for column, index in enumerate(classes):
                    full[:, int(index)] = raw[:, column]
            else:
                full[:, : raw.shape[1]] = raw
            probabilities[name][split] = full
            results["models"][name]["splits"][split] = evaluate(
                y[split], predicted, num_classes, full
            )

    # ---- soft vote across every member ---------------------------------- #
    results["models"]["soft_vote"] = {"splits": {}}
    for split in X:
        averaged = np.mean([probabilities[n][split] for n in models], axis=0)
        results["models"]["soft_vote"]["splits"][split] = evaluate(
            y[split], averaged.argmax(axis=1), num_classes, averaged
        )

    results["labels"] = {k: v.tolist() for k, v in y.items()}
    return results


def run_multi_seed(
    output_root: Path, seeds: Sequence[int], use_affect: bool, use_embeddings: bool,
    top_k_values: Sequence[int], num_classes: int = 4,
) -> pd.DataFrame:
    """Every (model, feature-count) combination across several seeds.

    Reports mean and standard deviation, because a single-seed number at n=36 is
    not a measurement. The std column is the one that decides whether any gap in
    the mean is real.
    """
    frame = load_clip_features(output_root, use_affect, use_embeddings)
    rows = []
    for top_k in top_k_values:
        for seed in seeds:
            results = run(
                output_root, use_affect, use_embeddings, seed, num_classes,
                top_k=top_k, quiet=True, frame=frame,
            )
            for name, bundle in results["models"].items():
                for split in ("Validation", "Test"):
                    metrics = bundle["splits"].get(split)
                    if metrics:
                        rows.append({
                            "model": name, "top_k": top_k or results["n_features"],
                            "seed": seed, "split": split,
                            "macro_f1": metrics["macro_f1"],
                            "accuracy": metrics["accuracy"],
                        })
    return pd.DataFrame(rows)


def leave_one_subject_out(
    output_root: Path,
    use_affect: bool = False,
    use_embeddings: bool = False,
    top_k_values: Sequence[int] = (0,),
    seeds: Sequence[int] = (42,),
    num_classes: int = 4,
) -> pd.DataFrame:
    """Leave-one-subject-out CV over every clip, for each model and feature count.

    Feature selection and standardisation are refitted inside each fold on the
    training subjects only, so no information crosses the fold boundary.
    """
    frame = load_clip_features(output_root, use_affect, use_embeddings)
    frame = frame.dropna(subset=[LABEL])
    feature_names_all = [
        c for c in frame.columns
        if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])
    ]
    matrix_all = np.nan_to_num(
        frame[feature_names_all].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    labels = frame[LABEL].to_numpy(dtype=int)
    subjects = frame["SubjectID"].astype(str).to_numpy()
    unique_subjects = sorted(set(subjects))
    logger.info(
        "LOSO: %d clips, %d subjects, %d features",
        len(labels), len(unique_subjects), len(feature_names_all),
    )

    rows = []
    for top_k in top_k_values:
        for seed in seeds:
            predictions = np.zeros_like(labels)
            for subject in unique_subjects:
                test_mask = subjects == subject
                train_mask = ~test_mask
                if labels[train_mask].size == 0 or len(set(labels[train_mask])) < 2:
                    predictions[test_mask] = labels[train_mask][0] if train_mask.any() else 0
                    continue

                X_train, y_train = matrix_all[train_mask], labels[train_mask]
                X_test = matrix_all[test_mask]
                if top_k:
                    keep = select_features(X_train, y_train, top_k, seed)
                    X_train, X_test = X_train[:, keep], X_test[:, keep]

                for name, model in build_models(seed).items():
                    if name not in {r["model"] for r in rows if r.get("_pending")}:
                        pass
                    model.fit(X_train, y_train)
                    rows.append({
                        "_pending": True, "model": name, "top_k": top_k or len(feature_names_all),
                        "seed": seed, "subject": subject,
                        "y_true": labels[test_mask].tolist(),
                        "y_pred": model.predict(X_test).tolist(),
                    })

    # Pool the held-out predictions per (model, top_k, seed) and score once, which
    # is the correct way to aggregate LOSO folds -- per-fold macro-F1 over one or
    # two clips would be meaningless.
    pooled = {}
    for row in rows:
        key = (row["model"], row["top_k"], row["seed"])
        bucket = pooled.setdefault(key, {"y_true": [], "y_pred": []})
        bucket["y_true"] += row["y_true"]
        bucket["y_pred"] += row["y_pred"]

    out = []
    for (model, top_k, seed), bucket in pooled.items():
        metrics = evaluate(bucket["y_true"], bucket["y_pred"], num_classes)
        out.append({
            "model": model, "top_k": top_k, "seed": seed,
            "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
            "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
            "n": metrics["n"],
        })
    return pd.DataFrame(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-affect", dest="use_affect", action="store_false")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Add the 1280-d HSEmotion embedding (likely to overfit at n=36).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Run a multi-seed sweep and report mean +/- std.")
    parser.add_argument("--top-k", nargs="+", type=int, default=[0, 10, 20, 40],
                        help="Feature counts to sweep (0 = all).")
    parser.add_argument("--loso", action="store_true",
                        help="Leave-one-subject-out CV over all clips instead of the "
                             "fixed split. Far more stable at this sample size.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.loso:
        table = leave_one_subject_out(
            args.output_root, args.use_affect, args.use_embeddings,
            args.top_k, args.seeds or [args.seed],
        )
        target = Path(args.output_root) / "results" / "ensemble_loso.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
        summary = (
            table.groupby(["model", "top_k"])[["macro_f1", "accuracy",
                                               "quadratic_weighted_kappa"]]
            .agg(["mean", "std"]).reset_index()
        )
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]
        summary = summary.sort_values("macro_f1_mean", ascending=False)
        logger.info("=" * 78)
        logger.info("LEAVE-ONE-SUBJECT-OUT (all 108 clips, 16 folds)")
        logger.info("  %-16s %6s %9s %8s %9s %8s",
                    "model", "top_k", "macro-F1", "std", "accuracy", "QWK")
        for row in summary.head(14).to_dict(orient="records"):
            std = row.get("macro_f1_std") or 0.0
            logger.info("  %-16s %6d %9.3f %8.3f %9.3f %8.3f",
                        row["model"], int(row["top_k"]), row["macro_f1_mean"],
                        std if std == std else 0.0,
                        row["accuracy_mean"], row["quadratic_weighted_kappa_mean"])
        logger.info("Wrote %s", target)
        return 0

    if args.seeds:
        table = run_multi_seed(
            args.output_root, args.seeds, args.use_affect, args.use_embeddings, args.top_k
        )
        target = Path(args.output_root) / "results" / "ensemble_multiseed.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)

        summary = (
            table.groupby(["split", "model", "top_k"])["macro_f1"]
            .agg(["mean", "std", "min", "max"]).reset_index()
        )
        for split in ("Test", "Validation"):
            subset = summary[summary.split == split].sort_values("mean", ascending=False)
            logger.info("=" * 74)
            logger.info("%s -- macro-F1 over %d seeds", split.upper(), len(args.seeds))
            logger.info("  %-16s %6s %8s %8s %8s %8s",
                        "model", "top_k", "mean", "std", "min", "max")
            for row in subset.head(12).to_dict(orient="records"):
                logger.info("  %-16s %6d %8.3f %8.3f %8.3f %8.3f",
                            row["model"], row["top_k"], row["mean"],
                            row["std"] if row["std"] == row["std"] else 0.0,
                            row["min"], row["max"])
        logger.info("Wrote %s", target)
        return 0

    results = run(args.output_root, args.use_affect, args.use_embeddings, args.seed)

    logger.info("=" * 74)
    logger.info("CLIP-LEVEL MODELS (macro-F1; test then validation)")
    logger.info("  %-16s %9s %9s %9s", "model", "test", "val", "test-acc")
    rows = []
    for name, bundle in results["models"].items():
        test = bundle["splits"].get("Test", {})
        val = bundle["splits"].get("Validation", {})
        rows.append((name, test.get("macro_f1", 0.0), val.get("macro_f1", 0.0),
                     test.get("accuracy", 0.0)))
    for name, test_f1, val_f1, accuracy in sorted(rows, key=lambda r: -r[1]):
        logger.info("  %-16s %9.3f %9.3f %9.3f", name, test_f1, val_f1, accuracy)

    best = max(rows, key=lambda r: r[1])[0]
    matrix = results["models"][best]["splits"]["Test"]["confusion_matrix"]
    logger.info("Best on test: %s\n%s", best, format_confusion_matrix(matrix))

    target = Path(args.output_root) / "results" / "ensemble_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
