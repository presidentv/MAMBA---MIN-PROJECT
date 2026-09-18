"""Report figures for the fine-tuned T=32 run.

Reuses the plotting functions from scripts/make_report_figures.py so the two
runs are drawn in the same visual language and can be put side by side in the
report, and adds two comparisons that only exist once there are two runs:

  * frozen T=16 vs fine-tuned T=32 across the three splits;
  * the learned MRS weights reached by each regime.

Written to artifacts/report_<run>/:
    ft1_training_curves.png            loss / accuracy / macro-F1 per epoch
    ft2_mrs_weight_evolution.png       learned MRS weights per epoch
    ft3_per_class_{val,test}.png       precision / recall / F1 per class
    ft6_confusion_normalised_test.png  per-class recall breakdown
    ft7_roc_pr_test.png                one-vs-rest ROC and precision-recall
    ft8_confidence_test.png            confidence when right vs wrong, calibration
    ft9_learning_rate_schedule.png     LR per epoch, selected epoch, epoch-30 decision
    ft4/ft5 only with --baseline       comparison against a frozen-backbone run

Run:  python scripts/make_finetune_figures.py --run ft32 --baseline mini64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.make_report_figures import (  # noqa: E402
    COMP_COLOURS, DEEP, GREY, LAND, LIGHT, MRSC,
    fig_per_class, fig_training_curves, fig_weight_evolution, load_history,
)
from src.metrics import DEFAULT_CLASS_NAMES  # noqa: E402
from src.mrs import COMPONENTS  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

CLASS_COLOURS = ("#C4553B", "#B4761F", "#1F6F78", "#6B4E9E")


def _is_balanced(metrics) -> bool:
    support = [c["support"] for c in metrics["per_class"].values()]
    return bool(support) and max(support) - min(support) <= 1


def fig_split_comparison(ft_report, base_metrics, out):
    """Grouped bars: fine-tuned T=32 against the frozen T=16 run."""
    splits = ["train", "val", "test"]
    metrics = [("accuracy", "accuracy"), ("macro_f1", "macro-F1"),
               ("weighted_f1", "weighted F1")]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=True)
    for ax, (key, label) in zip(axes, metrics):
        ft = [ft_report["splits"][s][key] for s in splits]
        bl = [base_metrics.get(s, {}).get(key, np.nan) for s in splits]
        x = np.arange(len(splits))
        w = 0.36
        b1 = ax.bar(x - w / 2, bl, w, label="frozen ViT, T=16", color=LIGHT)
        b2 = ax.bar(x + w / 2, ft, w, label="fine-tuned ViT, T=32", color=DEEP)
        for b in (b1, b2):
            ax.bar_label(b, fmt="%.3f", fontsize=7, padding=1.5)
        ax.axhline(0.25, color=MRSC, ls=":", lw=1.4)
        ax.set_xticks(x, [f"{s}\n(n={ft_report['splits'][s]['num_samples']})"
                          for s in splits])
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel("score")
    axes[0].annotate("chance (0.25)", (2.4, 0.25), xytext=(0, 5),
                     textcoords="offset points", ha="right", fontsize=7.5, color=MRSC)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Frozen backbone (T=16) vs end-to-end fine-tuning (T=32)",
                 fontsize=11, y=1.05)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_weight_comparison(ft_weights, base_weights, out):
    """The weights each regime converged to, from the same 0.20 start."""
    order = sorted(COMPONENTS, key=lambda c: ft_weights.get(c, 0), reverse=True)
    y = np.arange(len(order))
    h = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.barh(y + h / 2, [base_weights.get(c, np.nan) for c in order], h,
            color=LIGHT, label="frozen ViT, T=16")
    ax.barh(y - h / 2, [ft_weights.get(c, np.nan) for c in order], h,
            color=[COMP_COLOURS[c] for c in order], label="fine-tuned ViT, T=32")
    ax.axvline(0.2, color=GREY, ls="--", lw=1.2)
    ax.annotate("uniform start (0.20)", (0.2, len(order) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=7.8, color=GREY)
    for i, c in enumerate(order):
        ax.text(ft_weights.get(c, 0) + .006, i - h / 2, f"{ft_weights.get(c, 0):.3f}",
                va="center", fontsize=7.6)
        if c in base_weights:
            ax.text(base_weights[c] + .006, i + h / 2, f"{base_weights[c]:.3f}",
                    va="center", fontsize=7.6, color=GREY)
    ax.set_yticks(y, [c.replace("_", " ") for c in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(list(ft_weights.values()) + list(base_weights.values())) * 1.35)
    ax.set_xlabel("learned weight $w_c$")
    ax.set_title("Learned MRS weights under each training regime",
                 fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_confusion_normalised(cm, split, out):
    """Row-normalised confusion matrix: each row is the recall breakdown of a class.

    On DAiSEE the raw-count matrix is dominated by High / Very High; dividing
    each row by its class size shows what happens to the rare classes too.
    """
    cm = np.asarray(cm, dtype=np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = list(DEFAULT_CLASS_NAMES)
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_yticks(range(len(names)), [f"{n}\n(n={int(r)})" for n, r in zip(names, rows[:, 0])])
    ax.grid(False)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{split} confusion matrix, row-normalised\n"
                 "(diagonal = per-class recall; counts in brackets)",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.82, label="fraction of the true class")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_roc_pr(predictions, split, out):
    """One-vs-rest ROC and precision-recall curves from per-clip probabilities.

    PR is the more informative of the two here: with ~1% Very Low clips, a
    classifier can have a respectable ROC-AUC while finding almost none of them.
    """
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

    y = np.array([p["true"] for p in predictions])
    probs = np.array([p["probs"] for p in predictions], dtype=np.float64)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for c, name in enumerate(DEFAULT_CLASS_NAMES):
        pos = (y == c)
        if pos.sum() == 0 or pos.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(pos, probs[:, c])
        a1.plot(fpr, tpr, color=CLASS_COLOURS[c], lw=1.7,
                label=f"{name}  AUC {auc(fpr, tpr):.3f}  (n={int(pos.sum())})")
        prec, rec, _ = precision_recall_curve(pos, probs[:, c])
        ap = average_precision_score(pos, probs[:, c])
        a2.plot(rec, prec, color=CLASS_COLOURS[c], lw=1.7,
                label=f"{name}  AP {ap:.3f}  (prevalence {pos.mean():.3f})")
        a2.axhline(pos.mean(), color=CLASS_COLOURS[c], lw=0.9, ls=":")
    a1.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
    a1.set_title("ROC, one class vs rest", fontsize=10, fontweight="bold")
    a2.set_xlabel("recall"); a2.set_ylabel("precision"); a2.set_ylim(0, 1.02)
    a2.set_title("Precision-recall, one class vs rest\n(dotted = prevalence = random classifier)",
                 fontsize=10, fontweight="bold")
    a1.legend(frameon=False, fontsize=7.6, loc="lower right")
    a2.legend(fontsize=7.6, loc="lower left", framealpha=.9)
    fig.suptitle(f"{split} split, {len(y)} clips", fontsize=11, y=1.03)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_confidence(predictions, split, out):
    """How confident the model is when it is right versus when it is wrong."""
    probs = np.array([p["probs"] for p in predictions], dtype=np.float64)
    correct = np.array([p["true"] == p["pred"] for p in predictions])
    conf = probs.max(axis=1)
    bins = np.linspace(0.25, 1.0, 16)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    a1.hist(conf[correct], bins=bins, color=DEEP, alpha=.8,
            label=f"correct (n={int(correct.sum())})")
    a1.hist(conf[~correct], bins=bins, color="#C4553B", alpha=.7,
            label=f"wrong (n={int((~correct).sum())})")
    a1.set_xlabel("probability of the predicted class"); a1.set_ylabel("clips")
    a1.set_title("Prediction confidence", fontsize=10, fontweight="bold")
    a1.legend(frameon=False, fontsize=8)

    # Reliability diagram: within each confidence bin, how often it was right.
    idx = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
    centres, accs, counts = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum():
            centres.append(conf[m].mean()); accs.append(correct[m].mean()); counts.append(m.sum())
    a2.plot([0.25, 1], [0.25, 1], color=GREY, ls="--", lw=1, label="perfectly calibrated")
    a2.plot(centres, accs, "-o", color=MRSC, lw=1.7, ms=4, label="model")
    ece = sum(n * abs(a - c) for c, a, n in zip(centres, accs, counts)) / max(1, len(conf))
    a2.set_xlabel("mean confidence in bin"); a2.set_ylabel("accuracy in bin")
    a2.set_xlim(0.25, 1); a2.set_ylim(0, 1.02)
    a2.set_title(f"Calibration  (expected calibration error {ece:.3f})",
                 fontsize=10, fontweight="bold")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{split} split", fontsize=11, y=1.03)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_lr_schedule(rows, best_epoch, decision, out):
    """Learning rates per epoch, with the selected epoch and the epoch-30 decision."""
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(ep, [float(r["head_lr"]) for r in rows], color=DEEP, lw=1.8,
            label="Mamba / fusion / classifier")
    if rows and rows[0].get("backbone_lr") not in (None, ""):
        ax.plot(ep, [float(r["backbone_lr"]) for r in rows], color=LAND, lw=1.8,
                label="ViT backbone")
    ax.set_yscale("log")
    ax.axvline(best_epoch, color=GREY, ls="--", lw=1.1)
    ax.annotate(f"selected epoch {best_epoch}", (best_epoch, ax.get_ylim()[1]),
                xytext=(4, -4), textcoords="offset points", fontsize=7.5, va="top", color=GREY)
    if decision:
        d = int(decision["after_epochs"]) - 1
        verdict = ("continued" if decision["extend"] else "stopped")
        ax.axvline(d, color=MRSC, lw=1.4)
        ax.annotate(f"decision after {d + 1} epochs: {verdict}\n"
                    f"best last {decision['window']} = {decision['best_last_window']:.3f} vs "
                    f"before = {decision['best_before_window']:.3f}",
                    (d, ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points",
                    fontsize=7.5, color=MRSC)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate (log scale)")
    ax.set_title("Learning-rate schedule", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", framealpha=.9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ft32")
    ap.add_argument("--baseline", default="mini64",
                    help="frozen-backbone run to compare against; '' to skip")
    ap.add_argument("--out-dir", default=None,
                    help="default: artifacts/report_<run>")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir or f"artifacts/report_{args.run}")
    report = json.loads(
        Path(f"artifacts/finetune_report_{args.run}.json").read_text(encoding="utf-8"))
    rows = load_history(args.run)
    best_epoch = int(report["best_epoch"])

    # Titles and reference lines come from the run itself, not from the
    # 216-clip development subset the defaults were written for.
    tr = report["training"]["clips"]
    val = report["splits"]["val"]
    balanced = _is_balanced(val)
    if balanced:
        baselines = None
    else:
        maj = val["baselines"]["majority_class"]
        baselines = {
            "accuracy": (maj["accuracy"], f"majority class ({maj['accuracy']:.2f})"),
            "macro-F1": (maj["macro_f1"], f"majority class ({maj['macro_f1']:.2f})"),
        }
    title = (f"Training and validation over {len(rows)} epochs "
             f"(T={report['num_frames']}, {tr['train']} train / {tr['val']} val clips)")

    written = []
    written.append(fig_training_curves(rows, best_epoch,
                                       Path(out_dir) / "ft1_training_curves.png",
                                       title=title, baselines=baselines))
    written.append(fig_weight_evolution(rows, best_epoch,
                                        Path(out_dir) / "ft2_mrs_weight_evolution.png"))
    for split in ("val", "test"):
        m = report["splits"][split]
        written.append(fig_per_class(
            m, Path(out_dir) / f"ft3_per_class_{split}.png",
            f"Per-class {split} metrics, fine-tuned T={report['num_frames']} "
            f"({m['num_samples']} clips)",
            chance=0.25 if _is_balanced(m) else None))

    written.append(fig_confusion_normalised(
        report["splits"]["test"]["confusion_matrix"], "test",
        Path(out_dir) / "ft6_confusion_normalised_test.png"))
    written.append(fig_lr_schedule(rows, best_epoch, report.get("extend_decision"),
                                   Path(out_dir) / "ft9_learning_rate_schedule.png"))
    # Per-clip probabilities are written by run_finetune.py but git-ignored
    # (they pair clip ids with DAiSEE labels), so they exist only where the run
    # happened. The curves drawn from them are aggregate and safe to share.
    preds_path = Path(f"artifacts/per_clip_predictions_{args.run}_test.json")
    if preds_path.is_file():
        preds = json.loads(preds_path.read_text(encoding="utf-8"))["predictions"]
        written.append(fig_roc_pr(preds, "test", Path(out_dir) / "ft7_roc_pr_test.png"))
        written.append(fig_confidence(preds, "test",
                                      Path(out_dir) / "ft8_confidence_test.png"))
    else:
        print(f"skipping ROC/PR and confidence figures: {preds_path} not found")

    if args.baseline:
        base_metrics, base_weights = {}, {}
        for split, path in (("val", f"artifacts/val_metrics_{args.baseline}_val.json"),
                            ("test", f"artifacts/test_metrics_{args.baseline}_test.json")):
            p = Path(path)
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                # evaluate.py nests the scores under "metrics"; the fine-tune
                # report puts them at the top level. Accept either shape rather
                # than silently plotting empty bars.
                base_metrics[split] = d.get("metrics", d)
                if "accuracy" not in base_metrics[split]:
                    raise KeyError(f"{p} has no accuracy field at top level or under 'metrics'")
        bt = Path(f"artifacts/training_{args.baseline}.json")
        if bt.is_file():
            info = json.loads(bt.read_text(encoding="utf-8"))
            base_weights = info.get("learnable_mrs_weights_final") or {}

        if base_metrics:
            written.append(fig_split_comparison(
                report, base_metrics, Path(out_dir) / "ft4_regime_comparison.png"))
        if base_weights:
            written.append(fig_weight_comparison(
                report["learned_mrs_weights"], base_weights,
                Path(out_dir) / "ft5_weight_comparison.png"))

    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
