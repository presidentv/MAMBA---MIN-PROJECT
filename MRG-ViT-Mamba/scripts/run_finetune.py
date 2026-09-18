"""Fine-tune the ViT end to end, then report train / validation / test metrics.

Unlike scripts/run_training.py this does not touch the Stage 2 feature cache:
once the backbone is trained those cached vectors describe a model that no
longer exists. Crops are re-encoded through the ViT on every step.

Run:
    python scripts/run_finetune.py --config configs/config_ft32.yaml --run-name ft32
    python scripts/run_finetune.py --config configs/config_full.yaml --run-name full_ft32 \
        --workers 8 --resume

Selection is on validation macro-F1. The test split is read exactly once, after
the best checkpoint has been chosen, and never influences that choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import collate_finetune, load_calibration  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.finetune import (  # noqa: E402
    FineTuneResult, _run_epoch, build_finetune_datasets, finetune_model,
)
from src.losses import build_loss  # noqa: E402
from src.metrics import compute_metrics, format_report, plot_confusion_matrix  # noqa: E402
from src.model import MRGViTMamba  # noqa: E402
from src.mrs import COMPONENTS  # noqa: E402
from src.preprocess import stage1_key  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir, get_logger, load_config, resolve_path, save_json,
)
from src.vit_encoder import ViTFrameEncoder  # noqa: E402

CLASS_NAMES = ("Very Low", "Low", "High", "Very High")


def evaluate_split(model, dataset, loss_fn, device, batch_size, num_classes, workers=0):
    """Full metrics plus per-clip probabilities for one split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_finetune, num_workers=workers,
                        pin_memory=device.type == "cuda")
    loss, y_true, y_pred, probs, clip_ids, subject_ids = _run_epoch(
        model, loader, loss_fn, None, device, collect_probs=True)
    metrics = compute_metrics(y_true, y_pred, num_classes, CLASS_NAMES)
    metrics["loss"] = float(loss)
    per_clip = [
        {"clip_id": c, "subject_id": s, "true": int(t), "pred": int(p),
         "probs": [round(float(x), 6) for x in pr]}
        for c, s, t, p, pr in zip(clip_ids, subject_ids, y_true, y_pred, probs)
    ]
    return metrics, per_clip, y_true, y_pred


def baseline_metrics(train_counts, y_true, num_classes, seed=42):
    """Majority-class and stratified-random references for the same split."""
    rng = np.random.default_rng(seed)
    total = sum(train_counts.values())
    prior = np.array([train_counts.get(c, 0) / total for c in range(num_classes)])
    majority = int(np.argmax(prior))

    maj = compute_metrics(y_true, np.full_like(y_true, majority), num_classes, CLASS_NAMES)
    rand = [compute_metrics(y_true, rng.choice(num_classes, size=len(y_true), p=prior),
                            num_classes, CLASS_NAMES) for _ in range(200)]
    return {
        "train_prior": [round(float(p), 6) for p in prior],
        "majority_class": {"index": majority, "accuracy": maj["accuracy"],
                           "macro_f1": maj["macro_f1"], "weighted_f1": maj["weighted_f1"]},
        "stratified_random_mean": {
            k: float(np.mean([r[k] for r in rand]))
            for k in ("accuracy", "macro_f1", "weighted_f1")},
        "stratified_random_std": {
            k: float(np.std([r[k] for r in rand]))
            for k in ("accuracy", "macro_f1", "weighted_f1")},
        "uniform_chance_accuracy": 1.0 / num_classes,
    }


def binomial_p(correct: int, n: int, p0: float) -> float:
    """One-sided P(X >= correct) under Binomial(n, p0).

    Uses the survival function rather than summing comb(n, k) * p0**k terms:
    that sum raises OverflowError converting comb(n, k) to float once n passes
    roughly a thousand, i.e. on the full DAiSEE test split - after training has
    already finished.
    """
    from scipy.stats import binom
    if n <= 0:
        return 1.0
    return float(binom.sf(correct - 1, n, p0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_ft32.yaml")
    ap.add_argument("--run-name", default="ft32")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="score the existing best.pt instead of training again")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from checkpoints/<run>/last.pt")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader worker processes (overrides training.num_workers)")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="train even though the MRS calibration file has not been fitted")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    overrides = {"epochs": args.epochs} if args.epochs else None

    logger = get_logger("finetune", Path(cfg["paths"]["logs_dir"]) / f"{args.run_name}.log")
    logger.info("config: %s", args.config)
    logger.info("T = %d frames/clip | ViT freeze = %s",
                cfg["video"]["num_frames"], cfg["vit"]["freeze"])

    logger.info("dataset_root: %s%s", cfg["paths"]["dataset_root"],
                f"  (from {cfg['_path_overrides']['dataset_root']})"
                if cfg.get("_path_overrides", {}).get("dataset_root") else "")
    index = build_index(cfg)
    if not index.is_usable():
        logger.error("dataset index has fatal problems: %s",
                     [p.detail for p in index.problems][:5])
        return 1

    # Without a fitted calibration the blur and face-size terms fall back to
    # generic bounds, which silently changes what MRS means. On a long run that
    # is worth stopping for rather than discovering afterwards.
    calibration = load_calibration(cfg)
    if not calibration.calibrated and not args.allow_uncalibrated:
        logger.error("MRS calibration %s has not been fitted. Run "
                     "scripts/fit_mrs_stats.py --config %s after Stage 1, or pass "
                     "--allow-uncalibrated.", cfg["mrs"]["calibration_file"], args.config)
        return 1

    # ------------------------------------------------------------------ train
    if args.eval_only:
        prev = json.loads(resolve_path(
            Path(cfg["paths"]["artifacts_dir"]) / f"training_{args.run_name}.json"
        ).read_text(encoding="utf-8"))
        result = FineTuneResult(
            best_epoch=int(prev["best"]["epoch"]),
            best_val_macro_f1=float(prev["best"]["macro_f1"]),
            best_checkpoint=prev["checkpoint"], history=[], info=prev)
        logger.info("eval-only: reusing %s", result.best_checkpoint)
    else:
        result = finetune_model(cfg, index, run_name=args.run_name, logger=logger,
                                overrides=overrides, resume=args.resume,
                                num_workers=args.workers)
    logger.info("best epoch %d  val macro-F1 %.4f",
                result.best_epoch, result.best_val_macro_f1)

    # ------------------------------------------------- reload best, then score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(result.best_checkpoint, map_location=device, weights_only=False)
    ccfg = ckpt["config"]

    encoder = ViTFrameEncoder(model_name=ccfg["vit"]["model_name"], pretrained=False,
                              freeze=True, chunk_size=int(ccfg["vit"]["batch_size"]))
    model = MRGViTMamba(vit_dim=int(ckpt["vit_dim"]), landmark_dim=int(ckpt["landmark_dim"]),
                        cfg=ccfg, vit_encoder=encoder).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    s1_key = stage1_key(ccfg)
    datasets = build_finetune_datasets(ccfg, index, s1_key, encoder.spec,
                                       ("train", "val", "test"))
    num_classes = int(ccfg["dataset"]["num_classes"])
    loss_fn, _ = build_loss(dict(ccfg["training"]), datasets["train"].label_counts(),
                            num_classes)
    loss_fn = loss_fn.to(device)
    batch_size = int(ccfg["training"]["batch_size"])
    eval_workers = int(args.workers if args.workers is not None
                       else ccfg["training"].get("num_workers", 0))

    art_dir = ensure_dir(ccfg["paths"]["artifacts_dir"])
    train_counts = datasets["train"].label_counts()
    report = {
        "run_name": args.run_name,
        "regime": "end-to-end fine-tuning of ViT-B/16 + Mamba head",
        "num_frames": int(ccfg["video"]["num_frames"]),
        "best_epoch": int(ckpt["epoch"]),
        "selection_metric": "val_macro_f1 (test never used for selection)",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        ds = datasets[split]
        metrics, per_clip, y_true, y_pred = evaluate_split(
            model, ds, loss_fn, device, batch_size, num_classes, eval_workers)
        base = baseline_metrics(train_counts, y_true, num_classes)
        correct = int((np.asarray(y_true) == np.asarray(y_pred)).sum())
        metrics["baselines"] = base
        majority_acc = float(base["majority_class"]["accuracy"])
        metrics["significance"] = {
            "correct": correct,
            "n": len(y_true),
            # On a balanced split the two tests coincide. On an imbalanced one
            # (the full DAiSEE release is ~95% High/Very High) always predicting
            # the most common class already beats 25%, so only the test against
            # the majority-class accuracy says anything.
            "p_vs_majority_baseline": binomial_p(correct, len(y_true), majority_acc),
            "majority_baseline_accuracy": majority_acc,
            "p_vs_uniform_chance": binomial_p(correct, len(y_true), 1.0 / num_classes),
            "note": ("one-sided binomial; the primary test is against the accuracy of "
                     "always predicting the training-majority class. On an imbalanced "
                     "split, compare macro-F1 against the baselines as well - accuracy "
                     "alone rewards predicting the majority class."),
        }
        metrics["num_unique_subjects"] = len(set(ds.subjects()))
        report["splits"][split] = metrics

        (Path(art_dir) / f"classification_report_{args.run_name}_{split}.txt").write_text(
            format_report(y_true, y_pred, num_classes, CLASS_NAMES), encoding="utf-8")
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), list(CLASS_NAMES),
            Path(art_dir) / f"confusion_matrix_{args.run_name}_{split}.png",
            title=f"{split} confusion matrix (fine-tuned, T=32)")
        # Per-clip predictions carry DAiSEE label values, so they are written to
        # their own git-ignored file rather than into the committed metrics.
        save_json({"split": split, "run": args.run_name, "predictions": per_clip},
                  Path(art_dir) / f"per_clip_predictions_{args.run_name}_{split}.json")

        logger.info("%-5s | n=%3d acc %.4f  macroF1 %.4f  weightedF1 %.4f  loss %.4f",
                    split, metrics["num_samples"], metrics["accuracy"],
                    metrics["macro_f1"], metrics["weighted_f1"], metrics["loss"])

    # The weights that matter are the ones inside the checkpoint being scored.
    # Reporting the last epoch's values instead is wrong whenever selection
    # picks an earlier epoch, which is exactly what happens when the run
    # overfits: the evaluated model and the quoted weights would be different
    # models.
    if model.learnable_mrs is not None:
        w = model.learnable_mrs.weights().detach().cpu().numpy()
        report["learned_mrs_weights"] = {
            c: round(float(v), 6) for c, v in zip(COMPONENTS, w)}
    else:
        report["learned_mrs_weights"] = None
    report["learned_mrs_weights_last_epoch"] = result.info["learnable_mrs_weights_final"]
    report["mrs_weight_note"] = (
        f"learned_mrs_weights are those stored in the selected checkpoint "
        f"(epoch {int(ckpt['epoch'])}). learned_mrs_weights_last_epoch are where "
        f"they had drifted to by the final epoch, in a model that is not the one "
        f"evaluated here."
    )
    report["mrs_weight_start"] = {c: 0.2 for c in COMPONENTS}
    report["training"] = {
        k: result.info[k] for k in
        ("epochs_run", "batch_size", "grad_accum_steps", "effective_batch",
         "head_lr", "backbone_lr", "backbone", "clips", "label_counts", "subjects")
    }
    report["history_csv"] = result.info["history_csv"]
    report["checkpoint_dir"] = result.info["checkpoint_dir"]
    report["environment"] = result.info["environment"]

    out = Path(art_dir) / f"finetune_report_{args.run_name}.json"
    save_json(report, out)
    logger.info("wrote %s", out)

    print("\n" + "=" * 74)
    print(f"FINE-TUNED  T={report['num_frames']}  best epoch {report['best_epoch']}")
    print("=" * 74)
    for split in ("train", "val", "test"):
        m = report["splits"][split]
        sig = m["significance"]
        print(f"{split:<6} n={m['num_samples']:<5} acc={m['accuracy']:.4f}  "
              f"macroF1={m['macro_f1']:.4f}  weightedF1={m['weighted_f1']:.4f}  "
              f"| majority acc={sig['majority_baseline_accuracy']:.4f} "
              f"p={sig['p_vs_majority_baseline']:.4g}")
    print(f"\nMRS weights in the SELECTED checkpoint (epoch {report['best_epoch']}), "
          f"from 0.2 each  <- these are the weights that produced the scores above:")
    for c, v in (report["learned_mrs_weights"] or {}).items():
        print(f"  {c:<20} {v:.4f}  ({v - 0.2:+.4f})")
    last = report.get("learned_mrs_weights_last_epoch") or {}
    if last:
        print("\nwhere they had drifted by the final epoch (a different, overfit model):")
        for c, v in last.items():
            print(f"  {c:<20} {v:.4f}  ({v - 0.2:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
