"""Phase 5 -- train and validate the MRG-VM engagement classifier.

Trains the whole stack end to end (Vision Mamba -> adaptive fusion -> MLP),
early-stopping on validation macro-F1, and reports macro-F1 plus the full
confusion matrix on every split.

The Phase 3 and Phase 4 deliverables (behavioural embeddings and the fused
feature representation) are exported from the trained backbone by
``extract_embeddings.py`` -- a randomly initialised Mamba would emit embeddings
that mean nothing, so the deliverable only makes sense after this step.

Usage:
    python -m mrgvm.train_mrgvm --output-root outputs --config mrgvm/configs/default.json
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import evaluate, format_confusion_matrix  # noqa: E402

from .config import MRGVMConfig, dump_config, load_mrgvm_config  # noqa: E402
from .data import build_dataloaders  # noqa: E402
from .model import MRGVMModel, count_parameters  # noqa: E402

logger = logging.getLogger("mrgvm.train")


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
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    non_empty = counts > 0
    weights[non_empty] = counts[non_empty].sum() / (non_empty.sum() * counts[non_empty])
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def summarise_class_distribution(datasets, num_classes: int) -> Tuple[pd.DataFrame, List[str]]:
    rows, warnings = [], []
    for split, dataset in datasets.items():
        labels = dataset.labels()
        counts = np.bincount(labels, minlength=num_classes)
        rows.append({"split": split, **{f"class_{i}": int(counts[i]) for i in range(num_classes)},
                     "total": int(len(labels))})
        for index, count in enumerate(counts):
            if count == 0:
                warnings.append(f"{split}: class {index} has NO clips -- its metrics are undefined.")
            elif count < 5:
                warnings.append(
                    f"{split}: class {index} has only {count} clip(s) -- "
                    "its per-class F1 is not statistically meaningful."
                )
    return pd.DataFrame(rows).set_index("split"), warnings


def run_epoch(
    model: nn.Module, loader, device: torch.device, cfg: MRGVMConfig,
    optimizer=None, class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    trues, preds, probabilities, clip_ids = [], [], [], []

    for batch in loader:
        frames = batch["frames"].to(device)
        geometric = batch["geometric"].to(device)
        mrs = batch["mrs"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            output = model(frames, geometric, mrs, mask)
            loss = F.cross_entropy(
                output["logits"], labels, weight=class_weights,
                label_smoothing=cfg.train.label_smoothing,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        size = labels.size(0)
        total_loss += float(loss.item()) * size
        total_items += size
        trues.append(labels.detach().cpu().numpy())
        preds.append(output["logits"].argmax(dim=1).detach().cpu().numpy())
        probabilities.append(F.softmax(output["logits"], dim=1).detach().cpu().numpy())
        clip_ids.extend(batch["clip_id"])

    return (
        total_loss / max(total_items, 1),
        np.concatenate(trues), np.concatenate(preds), np.concatenate(probabilities), clip_ids,
    )


def train(
    output_root: Path, cfg: MRGVMConfig, device: torch.device,
    run_name: str = "mrgvm", save_checkpoint: bool = True, quiet: bool = False,
    prebuilt: Optional[Tuple[Dict, Dict, Dict]] = None,
) -> Dict[str, object]:
    """Train one MRG-VM configuration and return its result bundle.

    ``prebuilt`` lets the Phase 6 ablation runner pass already-loaded
    ``(loaders, datasets, info)`` so the 5k+ frame cache is decoded once for the
    whole sweep rather than once per variant.
    """
    set_seed(cfg.train.seed, cfg.train.deterministic)
    if prebuilt is not None:
        loaders, datasets, info = prebuilt
    else:
        loaders, datasets, info = build_dataloaders(
            output_root, cfg.data, cfg.splits, cfg.vision_mamba.image_size
        )

    model = MRGVMModel(cfg, info["geometric_dim"]).to(device)
    n_parameters = count_parameters(model)
    if not quiet:
        logger.info("[%s] %d trainable parameters | geometric dim %d",
                    run_name, n_parameters, info["geometric_dim"])

    train_labels = datasets["Train"].labels()
    weights = (
        class_weights_from(train_labels, cfg.classifier.num_classes, device)
        if cfg.train.class_weighting else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )

    total_epochs = max(cfg.train.epochs, 1)
    warmup = max(min(cfg.train.warmup_epochs, total_epochs - 1), 0)

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    validation_split = "Validation" if "Validation" in loaders else "Train"
    best_score, best_state, best_epoch, patience = -np.inf, None, -1, 0
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.train.epochs):
        started = time.perf_counter()
        train_loss, y_true, y_pred, _, _ = run_epoch(
            model, loaders["Train"], device, cfg, optimizer, weights
        )
        train_metrics = evaluate(y_true, y_pred, cfg.classifier.num_classes)
        val_loss, v_true, v_pred, _, _ = run_epoch(
            model, loaders[validation_split], device, cfg, None, weights
        )
        val_metrics = evaluate(v_true, v_pred, cfg.classifier.num_classes)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_macro_f1": train_metrics["macro_f1"], "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"], "val_accuracy": val_metrics["accuracy"],
            "seconds": round(time.perf_counter() - started, 2),
        })
        if not quiet and (epoch % 5 == 0 or epoch == cfg.train.epochs - 1):
            logger.info(
                "[%s] epoch %2d  train loss %.4f f1 %.3f | val loss %.4f f1 %.3f acc %.3f  (%.1fs)",
                run_name, epoch, train_loss, train_metrics["macro_f1"],
                val_loss, val_metrics["macro_f1"], val_metrics["accuracy"],
                history[-1]["seconds"],
            )

        if val_metrics["macro_f1"] > best_score:
            best_score, best_epoch, patience = val_metrics["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                if not quiet:
                    logger.info("[%s] early stopping at epoch %d (best %d)",
                                run_name, epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_checkpoint:
            checkpoint_dir = output_root / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": best_state, "epoch": best_epoch,
                    "val_macro_f1": best_score, "geometric_dim": info["geometric_dim"],
                    "geometric_columns": info["geometric_columns"],
                    "config": json.loads(json.dumps(_config_dict(cfg))),
                },
                checkpoint_dir / f"{run_name}.pt",
            )

    result: Dict[str, object] = {
        "run_name": run_name,
        "n_parameters": n_parameters,
        "geometric_dim": info["geometric_dim"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_score),
        "history": history,
        "splits": {},
    }
    for split, loader in loaders.items():
        _, y_true, y_pred, y_prob, clip_ids = run_epoch(model, loader, device, cfg, None, weights)
        result["splits"][split] = evaluate(y_true, y_pred, cfg.classifier.num_classes, y_prob)
    result["_model"] = model
    result["_datasets"] = datasets
    result["_info"] = info
    return result


def _config_dict(cfg: MRGVMConfig) -> Dict[str, object]:
    from .config import config_to_dict
    return config_to_dict(cfg)


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
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--run-name", default="mrgvm")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    results_dir = output_root / "results_mrgvm"
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(results_dir / "train_mrgvm.log", mode="w", encoding="utf-8"),
        ],
    )

    overrides: Dict[str, object] = {}
    if args.epochs is not None or args.learning_rate is not None or args.seed is not None:
        overrides["train"] = {
            k: v for k, v in
            {"epochs": args.epochs, "learning_rate": args.learning_rate, "seed": args.seed}.items()
            if v is not None
        }
    if args.batch_size is not None:
        overrides["data"] = {"batch_size": args.batch_size}
    if args.image_size is not None:
        overrides["vision_mamba"] = {"image_size": args.image_size}

    cfg = load_mrgvm_config(args.config, **overrides)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)
    dump_config(cfg, results_dir / "mrgvm_config.json")

    result = train(output_root, cfg, device, args.run_name)
    datasets = result.pop("_datasets")
    result.pop("_model")
    result.pop("_info")

    table, warnings = summarise_class_distribution(datasets, cfg.classifier.num_classes)
    logger.info("Class distribution (%s):\n%s", cfg.data.target, table.to_string())
    for warning in warnings:
        logger.warning("SMALL-SAMPLE: %s", warning)

    for split, metrics in result["splits"].items():
        logger.info("  %-11s macro-F1 %.3f | acc %.3f | QWK %.3f | MAE %.2f",
                    split, metrics["macro_f1"], metrics["accuracy"],
                    metrics["quadratic_weighted_kappa"], metrics["mean_absolute_error"])
    if "Test" in result["splits"]:
        logger.info("Test confusion matrix:\n%s", format_confusion_matrix(
            result["splits"]["Test"]["confusion_matrix"], cfg.classifier.num_classes))

    result["class_distribution"] = table.to_dict(orient="index")
    result["small_sample_warnings"] = warnings
    (results_dir / f"{args.run_name}_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", results_dir / f"{args.run_name}_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
