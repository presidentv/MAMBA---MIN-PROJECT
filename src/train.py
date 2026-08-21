"""Step 6 -- training loop, evaluation and the consolidated results file.

Runs every model named in ``ExperimentConfig.models_to_run`` and writes one
``results.json`` plus a flat ``results.csv`` containing macro-F1, per-class F1,
the full confusion matrix and ordinal metrics for each model on each split.

Guard rails that matter on a sample this small:
  * the per-split class distribution is printed BEFORE any training, with an
    explicit warning for any class holding fewer than five clips;
  * accuracy is never reported without macro-F1 beside it;
  * the majority-class baseline is always run, so an impressive-looking accuracy
    can be compared against the trivial floor immediately.

Usage:
    python src/train.py --output-root outputs --config src/configs/default.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as baseline_module  # noqa: E402
from config import ExperimentConfig, dump_config, load_experiment_config  # noqa: E402
from datasets import build_dataloaders, summarise_class_distribution  # noqa: E402
from metrics import evaluate, format_confusion_matrix  # noqa: E402
from models import (  # noqa: E402
    build_model,
    compute_loss,
    count_parameters,
    decode_predictions,
    decode_probabilities,
)

logger = logging.getLogger("train")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def class_weights_from(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale is stable.

    Empty classes get weight 1 rather than infinity.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    non_empty = counts > 0
    weights[non_empty] = counts[non_empty].sum() / (non_empty.sum() * counts[non_empty])
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Linear warmup then cosine decay, stepped per epoch."""
    total = max(cfg.epochs, 1)
    warmup = max(min(cfg.warmup_epochs, total - 1), 0)

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        if cfg.scheduler != "cosine":
            return 1.0
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(
    model: nn.Module,
    loader,
    cfg: ExperimentConfig,
    device: torch.device,
    modality: Optional[str] = None,
    optimizer=None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """One pass. Training when ``optimizer`` is given, evaluation otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss, total_items = 0.0, 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []
    all_clips: List[str] = []

    for batch in loader:
        gaze = batch["gaze"].to(device)
        affect = batch["affect"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            outputs = (
                model(gaze, affect, mask, modality=modality)
                if modality is not None
                else model(gaze, affect, mask)
            )
            logits = outputs["logits"]
            loss = compute_loss(
                logits, labels, cfg.model, cfg.data.num_classes, class_weights
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(decode_predictions(logits, cfg.model).detach().cpu().numpy())
        all_prob.append(decode_probabilities(logits, cfg.model).detach().cpu().numpy())
        all_clips.extend(batch["clip_id"])

    return (
        total_loss / max(total_items, 1),
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
        all_clips,
    )


def train_neural_model(
    name: str,
    loaders: Dict[str, object],
    datasets: Dict[str, object],
    cfg: ExperimentConfig,
    device: torch.device,
    checkpoint_dir: Path,
) -> Dict[str, object]:
    """Train one neural model with early stopping on validation macro-F1."""
    gaze_dim = int(datasets["Train"].gaze_dim)
    affect_dim = int(datasets["Train"].affect_dim)
    modality = {"gaze_only": "gaze", "affect_only": "affect"}.get(name)

    set_seed(cfg.train.seed, cfg.train.deterministic)
    model = build_model(name, gaze_dim, affect_dim, cfg.model, cfg.data.num_classes).to(device)
    logger.info("[%s] %d trainable parameters", name, count_parameters(model))

    train_labels = datasets["Train"].labels()
    weights = (
        class_weights_from(train_labels, cfg.data.num_classes, device)
        if cfg.model.class_weighting
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg.train, len(loaders["Train"]))

    best_score = -np.inf
    best_state = None
    best_epoch = -1
    patience = 0
    history: List[Dict[str, float]] = []
    validation_split = "Validation" if "Validation" in loaders else "Train"

    for epoch in range(cfg.train.epochs):
        train_loss, y_true, y_pred, _, _ = run_epoch(
            model, loaders["Train"], cfg, device, modality, optimizer, weights
        )
        train_metrics = evaluate(y_true, y_pred, cfg.data.num_classes)

        val_loss, v_true, v_pred, _, _ = run_epoch(
            model, loaders[validation_split], cfg, device, modality, None, weights
        )
        val_metrics = evaluate(v_true, v_pred, cfg.data.num_classes)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_loss,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if epoch % 5 == 0 or epoch == cfg.train.epochs - 1:
            logger.info(
                "[%s] epoch %2d  train loss %.4f f1 %.3f | val loss %.4f f1 %.3f acc %.3f",
                name, epoch, train_loss, train_metrics["macro_f1"],
                val_loss, val_metrics["macro_f1"], val_metrics["accuracy"],
            )

        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                logger.info("[%s] early stopping at epoch %d (best %d)", name, epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": best_state, "epoch": best_epoch, "val_macro_f1": best_score,
             "config": {"model": cfg.model.__dict__, "gaze_dim": gaze_dim, "affect_dim": affect_dim}},
            checkpoint_dir / f"{name}.pt",
        )

    result: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_score),
        "n_parameters": count_parameters(model),
        "history": history,
        "splits": {},
        "probabilities": {},
        "labels": {},
    }
    for split, loader in loaders.items():
        _, y_true, y_pred, y_prob, clip_ids = run_epoch(
            model, loader, cfg, device, modality, None, weights
        )
        result["splits"][split] = evaluate(y_true, y_pred, cfg.data.num_classes, y_prob)
        result["probabilities"][split] = y_prob.tolist()
        result["labels"][split] = y_true.tolist()
        result.setdefault("clip_ids", {})[split] = clip_ids
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss", choices=["coral", "ce"], default=None)
    parser.add_argument("--affect-feature-set", choices=["probs", "embedding", "both"], default=None)
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models_to_run.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(results_dir / "train.log", mode="w", encoding="utf-8"),
        ],
    )

    cfg: ExperimentConfig = load_experiment_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.loss is not None:
        cfg.model.loss = args.loss
    if args.affect_feature_set is not None:
        cfg.data.affect_feature_set = args.affect_feature_set
    if args.models is not None:
        cfg.models_to_run = tuple(args.models)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s | loss: %s | affect features: %s",
                device, cfg.model.loss, cfg.data.affect_feature_set)
    set_seed(cfg.train.seed, cfg.train.deterministic)
    dump_config(cfg, results_dir / "experiment_config.json")

    loaders, datasets, info = build_dataloaders(output_root, cfg.data, cfg.splits)
    logger.info("Feature dims -- gaze: %d, affect: %d", info["gaze_dim"], info["affect_dim"])
    logger.info("Clips per split: %s", info["n_clips_per_split"])

    # ---- class distribution up front ---------------------------------- #
    table, distribution_warnings = summarise_class_distribution(datasets, cfg.data.num_classes)
    logger.info("Class distribution (%s):\n%s", cfg.data.target, table.to_string())
    for warning in distribution_warnings:
        logger.warning("SMALL-SAMPLE: %s", warning)
    if distribution_warnings:
        logger.warning(
            "%d small-class warning(s). Treat every per-class number below as "
            "indicative only -- this is a smoke test on a sample, not a result.",
            len(distribution_warnings),
        )

    results: Dict[str, object] = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "n_clips_per_split": info["n_clips_per_split"],
            "gaze_dim": info["gaze_dim"],
            "affect_dim": info["affect_dim"],
            "subjects_per_split": info["integrity"]["subjects_per_split"],
            "class_distribution": table.to_dict(orient="index"),
            "small_sample_warnings": distribution_warnings,
            "loss": cfg.model.loss,
            "affect_feature_set": cfg.data.affect_feature_set,
        },
        "models": {},
    }
    if cfg.data.standardize:
        info["scaler"].save(results_dir / "feature_scaler.npz")

    checkpoint_dir = output_root / "checkpoints"
    neural_probabilities: Dict[str, Dict[str, np.ndarray]] = {}
    split_labels = {split: dataset.labels() for split, dataset in datasets.items()}

    for name in cfg.models_to_run:
        logger.info("=" * 78)
        logger.info("MODEL: %s", name)
        started = time.perf_counter()

        if name == "majority":
            result = baseline_module.majority_baseline(
                datasets["Train"], datasets, cfg.data.num_classes
            )
        elif name == "logreg_meanpool":
            result = baseline_module.logistic_regression_baseline(
                datasets["Train"], datasets, cfg.data.num_classes, cfg.train.seed
            )
        elif name == "late_fusion":
            if "gaze_only" not in neural_probabilities or "affect_only" not in neural_probabilities:
                logger.error(
                    "late_fusion needs gaze_only and affect_only to have run first; skipping."
                )
                continue
            result = baseline_module.late_fusion(
                neural_probabilities["gaze_only"],
                neural_probabilities["affect_only"],
                split_labels,
                cfg.data.num_classes,
            )
        else:
            result = train_neural_model(name, loaders, datasets, cfg, device, checkpoint_dir)
            neural_probabilities[name] = {
                split: np.asarray(p) for split, p in result["probabilities"].items()
            }
            # Drop the bulky per-clip arrays from the saved JSON; the checkpoint
            # reproduces them and results.json stays readable.
            result.pop("probabilities", None)
            result.pop("labels", None)

        result["seconds"] = round(time.perf_counter() - started, 2)
        results["models"][name] = result

        for split, metrics in result["splits"].items():
            logger.info(
                "  %-11s macro-F1 %.3f | acc %.3f | QWK %.3f | MAE %.2f",
                split, metrics["macro_f1"], metrics["accuracy"],
                metrics["quadratic_weighted_kappa"], metrics["mean_absolute_error"],
            )
        if "Test" in result["splits"]:
            logger.info(
                "  Test confusion matrix:\n%s",
                format_confusion_matrix(
                    result["splits"]["Test"]["confusion_matrix"], cfg.data.num_classes
                ),
            )

    # ------------------------------------------------------------------ #
    # Persist
    # ------------------------------------------------------------------ #
    (results_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat: List[Dict[str, object]] = []
    for name, result in results["models"].items():
        for split, metrics in result["splits"].items():
            row: Dict[str, object] = {
                "model": name,
                "split": split,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
                "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
                "mean_absolute_error": metrics["mean_absolute_error"],
                "n": metrics["n"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
            for class_index, value in metrics["per_class_f1"].items():
                row[f"f1_class_{class_index}"] = value
            for class_index, value in metrics["support"].items():
                row[f"support_class_{class_index}"] = value
            flat.append(row)
    pd.DataFrame(flat).to_csv(results_dir / "results.csv", index=False)

    logger.info("=" * 78)
    logger.info("FINAL TEST-SPLIT COMPARISON (macro-F1 is the headline, not accuracy)")
    test_rows = [r for r in flat if r["split"] == "Test"]
    test_rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    logger.info("  %-26s %9s %9s %9s", "model", "macro-F1", "accuracy", "QWK")
    for row in test_rows:
        logger.info(
            "  %-26s %9.3f %9.3f %9.3f",
            row["model"], row["macro_f1"], row["accuracy"], row["quadratic_weighted_kappa"],
        )
    logger.info("Results written to %s and %s",
                results_dir / "results.json", results_dir / "results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
