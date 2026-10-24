"""Fine-tune the ViT end to end, then report train / validation / test metrics.

Unlike scripts/run_training.py this does not touch the Stage 2 feature cache:
once the backbone is trained those cached vectors describe a model that no
longer exists. Crops are re-encoded through the ViT on every step.

Run:
    python scripts/run_finetune.py --config configs/config_ft32.yaml --run-name ft32
    python scripts/run_finetune.py --config configs/config_full.yaml --run-name full_ft32 \
        --workers 8 --resume

Selection is on validation macro-F1 by default, or on whatever
``training.selection_metric`` names. The test split is read exactly once, after
the best checkpoint has been chosen, and never influences that choice.

Two optional corrections, both applied after selection and neither touching the
weights:

``--logit-adjust``  undoes the prior distortion that weighted sampling and
                    class-weighted loss introduce, so accuracy is not paid as
                    the price of macro-F1 (src/prior_shift.py).
``--tta``           averages probabilities over the deterministic centre-crop
                    views in ``evaluation.tta_scales``: one extra forward pass
                    per view, nothing else changed.

Both are reported *alongside* the unadjusted numbers, never instead of them.
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

from src.augment import CenterCropView  # noqa: E402
from src.class_merge import class_names as merged_class_names  # noqa: E402
from src.dataset import collate_finetune, load_calibration  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.detection import evaluate_detector  # noqa: E402
from src.finetune import (  # noqa: E402
    FineTuneResult, _run_epoch, build_finetune_datasets, finetune_model,
)
from src.losses import build_loss  # noqa: E402
from src.metrics import compute_metrics, format_report, plot_confusion_matrix  # noqa: E402
from src.model import MRGViTMamba  # noqa: E402
from src.mrs import COMPONENTS  # noqa: E402
from src.preprocess import stage1_key  # noqa: E402
from src.prior_shift import adjust_predictions, effective_train_counts  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir, get_logger, load_config, resolve_path, save_json,
)
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


def _one_pass(model, dataset, loss_fn, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_finetune, num_workers=workers,
                        pin_memory=device.type == "cuda")
    return _run_epoch(model, loader, loss_fn, None, device, collect_probs=True)


def evaluate_split(model, dataset, loss_fn, device, batch_size, num_classes, workers=0,
                   class_names=None, tta_views=None):
    """Full metrics plus per-clip probabilities for one split.

    With ``tta_views`` the split is scored once per view and the probabilities
    are averaged. The views are deterministic centre crops, so the number does
    not depend on an RNG, and shuffle is off, so row i is the same clip in every
    pass. That alignment is asserted rather than assumed: averaging misaligned
    rows would still produce a plausible-looking score.
    """
    names = tuple(class_names) if class_names else None
    loss, y_true, y_pred, probs, clip_ids, subject_ids = _one_pass(
        model, dataset, loss_fn, device, batch_size, workers)

    tta_record = None
    extra = [v for v in (tta_views or []) if float(v.scale) < 1.0]
    if extra:
        original_augment = dataset.augment
        stacked, losses = [probs], [float(loss)]
        try:
            for view in extra:
                dataset.augment = view
                v_loss, v_true, _, v_probs, v_ids, _ = _one_pass(
                    model, dataset, loss_fn, device, batch_size, workers)
                if v_ids != clip_ids or not np.array_equal(v_true, y_true):
                    raise RuntimeError(
                        "a TTA pass returned a different clip order than the first "
                        "pass; refusing to average across misaligned rows")
                stacked.append(v_probs)
                losses.append(float(v_loss))
        finally:
            dataset.augment = original_augment
        probs = np.mean(np.stack(stacked), axis=0)
        y_pred = probs.argmax(axis=1)
        loss = float(np.mean(losses))
        tta_record = {
            "views": [{"view": "center_crop", "scale": 1.0}] + [v.describe() for v in extra],
            "num_passes": len(stacked),
            "note": ("probabilities averaged over deterministic centre-crop "
                     "views; the reported loss is the mean of the per-view "
                     "losses, not the loss of the averaged distribution"),
        }

    metrics = compute_metrics(y_true, y_pred, num_classes, names)
    metrics["loss"] = float(loss)
    metrics["tta"] = tta_record
    per_clip = [
        {"clip_id": c, "subject_id": s, "true": int(t), "pred": int(p),
         "probs": [round(float(x), 6) for x in pr]}
        for c, s, t, p, pr in zip(clip_ids, subject_ids, y_true, y_pred, probs)
    ]
    return metrics, per_clip, y_true, y_pred, probs


BASELINE_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def baseline_metrics(train_counts, y_true, num_classes, seed=42, class_names=None):
    """Majority-class and stratified-random references for the same split.

    Balanced accuracy is included for each because its baseline is a different
    number from the accuracy baseline: always answering "High" scores 0.494 by
    accuracy on DAiSEE's test split but only 1/K by balanced accuracy. A result
    has to be read against the baseline for the metric being quoted.
    """
    rng = np.random.default_rng(seed)
    total = sum(train_counts.values())
    prior = np.array([train_counts.get(c, 0) / total for c in range(num_classes)])
    majority = int(np.argmax(prior))

    maj = compute_metrics(y_true, np.full_like(y_true, majority), num_classes, class_names)
    rand = [compute_metrics(y_true, rng.choice(num_classes, size=len(y_true), p=prior),
                            num_classes, class_names) for _ in range(200)]
    return {
        "train_prior": [round(float(p), 6) for p in prior],
        "majority_class": {"index": majority, **{k: maj[k] for k in BASELINE_KEYS}},
        "stratified_random_mean": {
            k: float(np.mean([r[k] for r in rand])) for k in BASELINE_KEYS},
        "stratified_random_std": {
            k: float(np.std([r[k] for r in rand])) for k in BASELINE_KEYS},
        "uniform_chance_accuracy": 1.0 / num_classes,
        "balanced_accuracy_chance": 1.0 / num_classes,
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
    ap.add_argument("--resume-from", default=None, metavar="CKPT",
                    help="continue from a chosen snapshot instead of last.pt, e.g. "
                         "checkpoints/<run>/after_epoch_020.pt")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader worker processes (overrides training.num_workers)")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="train even though the MRS calibration file has not been fitted")
    ap.add_argument("--logit-adjust", action="store_true",
                    help="also report metrics after correcting the sampler's prior "
                         "shift (post-hoc logit adjustment; weights unchanged)")
    ap.add_argument("--logit-adjust-tau", type=float, default=1.0,
                    help="strength of that correction; 1.0 is the Bayes-exact value")
    ap.add_argument("--tta", action="store_true",
                    help="average probabilities over evaluation.tta_scales views")
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
                                num_workers=args.workers, resume_from=args.resume_from)
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
    # Names come from the checkpoint's own config, so a 3-class model is never
    # labelled with the four original DAiSEE level names.
    class_names = tuple(ckpt.get("class_names")
                        or merged_class_names(ccfg, num_classes))
    tta_views = CenterCropView.set_from_config(dict(ccfg.get("evaluation", {}))) \
        if args.tta else []
    if args.tta and not tta_views:
        logger.error("--tta was given but evaluation.tta_scales is empty in %s",
                     args.config)
        return 1
    loss_fn, _ = build_loss(dict(ccfg["training"]), datasets["train"].label_counts(),
                            num_classes)
    loss_fn = loss_fn.to(device)
    batch_size = int(ccfg["training"]["batch_size"])
    eval_workers = int(args.workers if args.workers is not None
                       else ccfg["training"].get("num_workers", 0))

    art_dir = ensure_dir(ccfg["paths"]["artifacts_dir"])
    train_counts = datasets["train"].label_counts()
    # What the model was actually optimised against: the sampler's expected
    # draws when sampling is on, the real counts when it is off.
    eff_counts = effective_train_counts(train_counts, ckpt.get("sampling"), num_classes)
    selected_on = ckpt.get("selection_metric") or result.selection_metric
    report = {
        "run_name": args.run_name,
        "regime": "end-to-end fine-tuning of ViT-B/16 + Mamba head",
        "num_frames": int(ccfg["video"]["num_frames"]),
        "best_epoch": int(ckpt["epoch"]),
        "selection_metric": f"val_{selected_on} (test never used for selection)",
        "num_classes": num_classes,
        "class_names": list(class_names),
        "class_merge": ckpt.get("class_merge") or getattr(index, "class_merge", None),
        "effective_train_counts": {str(k): int(v) for k, v in sorted(eff_counts.items())},
        "tta": ([v.describe() for v in tta_views] if tta_views else None),
        "splits": {},
    }

    collected = {}
    for split in ("train", "val", "test"):
        ds = datasets[split]
        metrics, per_clip, y_true, y_pred, probs = evaluate_split(
            model, ds, loss_fn, device, batch_size, num_classes, eval_workers,
            class_names=class_names, tta_views=tta_views)
        base = baseline_metrics(train_counts, y_true, num_classes,
                                class_names=class_names)
        correct = int((np.asarray(y_true) == np.asarray(y_pred)).sum())
        metrics["baselines"] = base

        # The prior correction is reported as an extra block. The unadjusted
        # numbers above stay exactly as they were, so nothing is overwritten by
        # a post-hoc transform and both can be quoted side by side.
        if args.logit_adjust:
            # The target prior comes from the config passed on the COMMAND
            # LINE, not from the checkpoint's. Everything structural -- the
            # architecture, T, the labelling -- must come from the checkpoint,
            # because evaluating a model with a mechanism it was not trained
            # with produces numbers that look fine and are wrong. But which
            # distribution to calibrate the decision boundary to is an
            # evaluation-time choice: a model trained on a subset may be
            # reported against the corpus prior, and that decision cannot
            # require retraining. A reference_prior of the wrong length for
            # this checkpoint's class count raises rather than being ignored.
            adj = adjust_predictions(probs, eff_counts, num_classes, cfg=cfg,
                                     natural_counts=train_counts,
                                     tau=args.logit_adjust_tau)
            adj_metrics = compute_metrics(y_true, adj["predictions"], num_classes,
                                          class_names)
            adj_correct = int((np.asarray(y_true) == adj["predictions"]).sum())
            metrics["logit_adjusted"] = {
                **{k: adj_metrics[k] for k in BASELINE_KEYS},
                "per_class": adj_metrics["per_class"],
                "confusion_matrix": adj_metrics["confusion_matrix"],
                "correct": adj_correct,
                "p_vs_majority_baseline": binomial_p(
                    adj_correct, len(y_true),
                    float(base["majority_class"]["accuracy"])),
                **adj["record"],
            }
            # A wrong reference_prior makes the correction actively harmful --
            # it moves the decision boundary towards a distribution that is not
            # there. Measured on the 216-clip subset, calibrating a 0.50/0.25/
            # 0.25 model to 0.20/0.40/0.40 dropped test accuracy from 0.750 to
            # 0.375. So the shift is logged, once, rather than being applied
            # silently: if these numbers look nothing like the split, the
            # target prior is wrong.
            if split == "train":
                logger.info("logit adjustment: train prior %s -> target %s "
                            "(%s), shift %s",
                            adj["record"]["effective_train_prior"],
                            adj["record"]["target_prior"],
                            adj["record"]["applied_to"],
                            adj["record"]["logit_shift"])
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
        collected[split] = (np.asarray(y_true), np.asarray(probs))

        (Path(art_dir) / f"classification_report_{args.run_name}_{split}.txt").write_text(
            format_report(y_true, y_pred, num_classes, class_names), encoding="utf-8")
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), list(class_names),
            Path(art_dir) / f"confusion_matrix_{args.run_name}_{split}.png",
            title=f"{split} confusion matrix (fine-tuned, "
                  f"T={int(ccfg['video']['num_frames'])})")
        # Per-clip predictions carry DAiSEE label values, so they are written to
        # their own git-ignored file rather than into the committed metrics.
        save_json({"split": split, "run": args.run_name, "predictions": per_clip},
                  Path(art_dir) / f"per_clip_predictions_{args.run_name}_{split}.json")

        logger.info("%-5s | n=%4d acc %.4f  balAcc %.4f  macroF1 %.4f  "
                    "weightedF1 %.4f  loss %.4f",
                    split, metrics["num_samples"], metrics["accuracy"],
                    metrics["balanced_accuracy"], metrics["macro_f1"],
                    metrics["weighted_f1"], metrics["loss"])

    # On the binary task accuracy is uninformative: 1,696 engaged clips against
    # 88 not-engaged means a constant "engaged" answer scores 0.95 while
    # detecting nothing, and a real detector usually scores LOWER. So ROC-AUC is
    # the headline and the operating point is chosen on validation, never on
    # test. See src/detection.py.
    if num_classes == 2:
        tune_on = collected.get("val")
        report["detection"] = {}
        for split, (y_s, p_s) in collected.items():
            report["detection"][split] = evaluate_detector(
                y_s, p_s, tune_on=tune_on if split != "val" else None,
                objective=str(ccfg.get("evaluation", {}).get(
                    "threshold_objective", "balanced_accuracy")))
        # Validation needs its own threshold reported too, but tuned and applied
        # on the same split is optimistic by construction, so it is labelled.
        v_true, v_probs = collected["val"]
        report["detection"]["val"]["tuned_in_sample"] = evaluate_detector(
            v_true, v_probs, tune_on=(v_true, v_probs))["tuned"]
        report["detection"]["val"]["tuned_in_sample"]["note"] = (
            "threshold tuned and applied on validation; optimistic by "
            "construction and shown only to bound what tuning can buy")

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
    # .get: training JSONs written before the decision point existed lack it.
    report["extend_decision"] = result.info.get("extend_decision")
    report["history_csv"] = result.info["history_csv"]
    report["checkpoint_dir"] = result.info["checkpoint_dir"]
    report["environment"] = result.info["environment"]

    out = Path(art_dir) / f"finetune_report_{args.run_name}.json"
    save_json(report, out)
    logger.info("wrote %s", out)

    print("\n" + "=" * 94)
    print(f"FINE-TUNED  T={report['num_frames']}  {num_classes} classes  "
          f"best epoch {report['best_epoch']}  (selected on val {selected_on})")
    print(f"classes: {', '.join(class_names)}")
    if report["class_merge"]:
        print(f"class_merge: {report['class_merge']['merge']['groups']}")
    print("=" * 94)
    for split in ("train", "val", "test"):
        m = report["splits"][split]
        sig = m["significance"]
        print(f"{split:<6} n={m['num_samples']:<5} acc={m['accuracy']:.4f}  "
              f"balAcc={m['balanced_accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"weightedF1={m['weighted_f1']:.4f}")
        print(f"       baselines: majority acc={sig['majority_baseline_accuracy']:.4f} "
              f"(p={sig['p_vs_majority_baseline']:.4g})   "
              f"majority balAcc="
              f"{m['baselines']['majority_class']['balanced_accuracy']:.4f}   "
              f"majority macroF1={m['baselines']['majority_class']['macro_f1']:.4f}")
        adj = m.get("logit_adjusted")
        if adj:
            print(f"       prior-corrected: acc={adj['accuracy']:.4f}  "
                  f"balAcc={adj['balanced_accuracy']:.4f}  "
                  f"macroF1={adj['macro_f1']:.4f}  "
                  f"(p={adj['p_vs_majority_baseline']:.4g})")
        if m.get("tta"):
            print(f"       TTA over {m['tta']['num_passes']} deterministic views")

    det = report.get("detection")
    if det:
        print()
        print("=" * 94)
        print("BINARY DETECTION: finding the not-engaged clips (the minority)")
        print("=" * 94)
        for split in ("train", "val", "test"):
            d = det.get(split)
            if not d:
                continue
            r, b = d["ranking"], d["baselines"]
            print(f"{split:<6} n={r['n_positive'] + r['n_negative']:<5} "
                  f"not-engaged {r['n_positive']:<5} "
                  f"ROC-AUC {r['roc_auc']:.4f} (0.5 = no signal)   "
                  f"avg precision {r['average_precision']:.4f} "
                  f"(baseline {b['average_precision']:.4f})")
            for key in ("argmax", "tuned"):
                p = d.get(key)
                if not p:
                    continue
                label = "  at 0.5     " if key == "argmax" else f"  at {p['threshold']:.3f}  "
                print(f"{label} acc {p['accuracy']:.4f} "
                      f"(majority {b['majority_class_accuracy']:.4f})   "
                      f"balAcc {p['balanced_accuracy']:.4f}   "
                      f"recall {p['recall_not_engaged']:.3f}   "
                      f"precision {p['precision_not_engaged']:.3f}   "
                      f"flags {p['predicted_not_engaged']} of "
                      f"{p['actually_not_engaged']} real")
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
