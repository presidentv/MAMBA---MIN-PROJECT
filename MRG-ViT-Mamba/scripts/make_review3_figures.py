"""Figures for the final review, drawn in the same visual language as Review 2.

Review 2's figures came from scripts/make_report_figures.py and described the
216-clip development subset. The model has moved on since: the full DAiSEE
release, a partially fine-tuned backbone, a binary detection formulation and a
controlled ablation. These are the figures for that work, using the same
palette and rcParams so the two decks sit together.

Written to artifacts/review3/ (git-ignored, because the worked examples show
DAiSEE participants):

    r3_roc_pr.png          ROC and precision-recall for disengagement detection
    r3_confusion.png       confusion matrix at the reported operating point
    r3_ablation.png        reliability weighting on vs removed, with the gap
    r3_data_scale.png      1,500-clip subset vs the full training partition
    r3_mrs_weights.png     learned reliability weights per epoch
    r3_shap_per_dim.png    attribution per dimension by feature family
    r3_examples.png        worked examples from the DAiSEE test split

Run:  python scripts/make_review3_figures.py
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
    COMP_COLOURS, DEEP, GREY, LAND, LIGHT, MRSC)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "review3"
CACHE = Path(r"C:\Users\ashok\mrg_run\cache\faces_T16_7b2356cf6f\test")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight",
})

POS = 0          # "Not engaged" is the detection class


# ------------------------------------------------------------------- inputs
def predictions(run: str, split: str = "test") -> tuple[np.ndarray, np.ndarray,
                                                        list[str]]:
    path = ART / f"per_clip_predictions_{run}_{split}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    y = np.array([int(r["true"]) for r in rows])
    s = np.array([float(r["probs"][POS]) for r in rows])
    return y, s, [r["clip_id"] for r in rows]


def report(run: str) -> dict:
    return json.loads((ART / f"finetune_report_{run}.json").read_text(
        encoding="utf-8"))


def roc_curve(y, score):
    """ROC without sklearn, so the figure script has no extra dependency."""
    order = np.argsort(-score)
    y = (y[order] == POS).astype(int)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def pr_curve(y, score):
    order = np.argsort(-score)
    y = (y[order] == POS).astype(int)
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    recall = tp / max(tp[-1], 1)
    ap = float(np.sum((recall - np.concatenate([[0], recall[:-1]])) * precision))
    return recall, precision, ap


# ------------------------------------------------------------------ figures
def fig_roc_pr(out: Path):
    y, s, _ = predictions("fullbin")
    fpr, tpr, auc = roc_curve(y, s)
    rec, prec, ap = pr_curve(y, s)
    rate = float((y == POS).mean())

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax[0].plot(fpr, tpr, color=DEEP, lw=2, label=f"MRS-ViT-Mamba  AUC {auc:.4f}")
    ax[0].plot([0, 1], [0, 1], ls="--", color=GREY, lw=1,
               label="no signal  AUC 0.5000")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("Disengagement detection \u2014 ROC", fontsize=10)
    ax[0].legend(loc="lower right", frameon=False, fontsize=8)

    ax[1].plot(rec, prec, color=LAND, lw=2, label=f"MRS-ViT-Mamba  AP {ap:.4f}")
    ax[1].axhline(rate, ls="--", color=GREY, lw=1,
                  label=f"positive rate  {rate:.4f}")
    ax[1].set_xlabel("Recall (not engaged)")
    ax[1].set_ylabel("Precision (not engaged)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Disengagement detection \u2014 precision/recall", fontsize=10)
    ax[1].legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(f"Full DAiSEE test partition, {len(y):,} clips "
                 f"({int((y == POS).sum())} not engaged)", fontsize=9,
                 color=GREY, y=1.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}  AUC {auc:.4f}  AP {ap:.4f}")


def fig_confusion(out: Path):
    det = report("fullbin")["detection"]["test"]["argmax"]["confusion"]
    names = ["Not engaged", "Engaged"]
    m = np.array([[det["tp"], det["fn"]], [det["fp"], det["tn"]]], float)
    rown = m / m.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j]):,}\n{rown[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=10,
                    color="white" if rown[i, j] > 0.55 else "#1A1A1A")
    ax.set_title("Binary detection at the default threshold", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}  {int(m[0,0])} of {int(m[0].sum())} found, "
          f"{int(m[1,0])} false alarms")


def fig_ablation(out: Path):
    full = report("fullbin")["detection"]["test"]
    ctrl = report("fullbin_nomrs")["detection"]["test"]
    metrics = [("ROC-AUC", "ranking", "roc_auc"),
               ("Average precision", "ranking", "average_precision"),
               ("Balanced accuracy", "argmax", "balanced_accuracy")]
    a = [full[b][k] for _, b, k in metrics]
    c = [ctrl[b][k] for _, b, k in metrics]
    labels = [m[0] for m in metrics]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - 0.19, a, 0.36, color=DEEP, label="Reliability weighting on")
    ax.bar(x + 0.19, c, 0.36, color=LIGHT, edgecolor=GREY, lw=.6,
           label="Reliability weighting removed")
    for i, (va, vc) in enumerate(zip(a, c)):
        ax.text(i - 0.19, va + .012, f"{va:.4f}", ha="center", fontsize=8)
        ax.text(i + 0.19, vc + .012, f"{vc:.4f}", ha="center", fontsize=8)
        ax.text(i, max(va, vc) + .075, f"\u0394 {va - vc:+.4f}", ha="center",
                fontsize=8.5, color=MRSC, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score on the full test partition")
    ax.set_title("Controlled ablation \u2014 two of 117 settings changed, "
                 "same clips, same seed", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}  delta ROC-AUC {a[0] - c[0]:+.4f}")


def fig_data_scale(out: Path):
    """What actually moved the number: minority-class data, not architecture."""
    runs = [("1,500-clip subset\n(69 not-engaged)", 0.6497, 0.7324),
            ("Full training partition\n(247 not-engaged)", 0.7385, 0.7943)]
    x = np.arange(len(runs))
    val = [r[1] for r in runs]
    tst = [r[2] for r in runs]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - 0.19, val, 0.36, color=LIGHT, edgecolor=GREY, lw=.6,
           label="Validation ROC-AUC")
    ax.bar(x + 0.19, tst, 0.36, color=DEEP, label="Test ROC-AUC")
    for i in range(len(runs)):
        ax.text(i - 0.19, val[i] + .012, f"{val[i]:.4f}", ha="center", fontsize=8)
        ax.text(i + 0.19, tst[i] + .012, f"{tst[i]:.4f}", ha="center", fontsize=8)
    ax.annotate(f"+{tst[1] - tst[0]:.3f}", xy=(1.19, tst[1]), xytext=(0.62, 0.90),
                fontsize=10, color=MRSC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MRSC, lw=1.3))
    ax.axhline(0.5, ls="--", color=GREY, lw=1)
    ax.text(1.42, 0.513, "no signal", fontsize=8, color=GREY, ha="right")
    ax.set_xticks(x, [r[0] for r in runs])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Effect of training-set size \u2014 identical architecture "
                 "and evaluation data", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


def fig_mrs_weights(out: Path):
    import csv

    rows = list(csv.DictReader(
        open(ROOT / "logs" / "training_history_fullbin.csv", encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for comp, colour in COMP_COLOURS.items():
        ax.plot(ep, [float(r["w_" + comp]) for r in rows], marker="o", ms=3.5,
                lw=1.7, color=colour, label=comp.replace("_", " "))
    ax.axhline(0.2, ls="--", color=GREY, lw=1)
    ax.text(ep[-1], 0.2015, "uniform 0.200", ha="right", va="bottom",
            fontsize=8, color=GREY)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learned MRS weight")
    ax.set_xticks(ep)
    ax.set_ylim(0.15, 0.25)
    ax.set_title("Learned reliability weights stay bounded under weight decay",
                 fontsize=10)
    ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.30))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    lo = min(float(r["w_" + c]) for r in rows for c in COMP_COLOURS)
    hi = max(float(r["w_" + c]) for r in rows for c in COMP_COLOURS)
    print(f"  {out.name}  range {lo:.3f}-{hi:.3f}")


def fig_shap_per_dim(out: Path):
    """Group totals mislead because the groups differ in size; per dimension
    is the comparable quantity, and it reverses the ranking."""
    data = json.loads((ART / "shap_analysis_fullbin_test.json").read_text(
        encoding="utf-8"))
    sizes = data["group_sizes"]
    totals = data["group_total_mean_abs_shap"]
    shares = data["group_share_of_total"]
    per_dim = data["group_share_per_dimension"]
    rows = [(name, float(totals[name]) , int(sizes[name]), float(per_dim[name]),
             float(shares[name]) * 100.0)
            for name in sizes if sizes[name]]
    rows.sort(key=lambda r: -r[3])
    names = [r[0].replace("_", " ") for r in rows]
    per = [r[3] for r in rows]
    share = [r[4] for r in rows]
    y = np.arange(len(rows))
    colours = [DEEP if "deep" in r[0] else LAND for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].barh(y, per, color=colours)
    ax[0].set_yticks(y, names)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Mean |SHAP| per dimension")
    ax[0].set_title("Per dimension \u2014 the comparable measure", fontsize=10)
    for i, v in enumerate(per):
        ax[0].text(v, i, f" {v:.5f}", va="center", fontsize=7.5)

    order = np.argsort(-np.array(share))
    ax[1].barh(np.arange(len(rows)), [share[i] for i in order],
               color=[colours[i] for i in order])
    ax[1].set_yticks(np.arange(len(rows)), [names[i] for i in order])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Share of total attribution (%)")
    ax[1].set_title("Group total \u2014 dominated by group size", fontsize=10)
    for i, v in enumerate([share[i] for i in order]):
        ax[1].text(v, i, f" {v:.1f}%", va="center", fontsize=7.5)

    fig.suptitle("SHAP attribution, 512 test clips against 512 background "
                 "clips \u2014 deep block in teal, landmark families in purple",
                 fontsize=9, color=GREY, y=1.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}  top per-dim: {names[0]} ({per[0]:.5f})")


# -------------------------------------------------------- worked examples
def pick_examples(n_each: int = 2):
    """Choose confident-correct and confident-wrong clips from the test split."""
    rows = json.loads((ART / "per_clip_predictions_fullbin_test.json")
                      .read_text(encoding="utf-8"))["predictions"]
    rows = [r for r in rows if (CACHE / f"{Path(r['clip_id']).stem}.npz").is_file()]
    pos = [r for r in rows if r["true"] == POS]
    correct = sorted([r for r in pos if r["pred"] == POS],
                     key=lambda r: -r["probs"][POS])[:n_each]
    missed = sorted([r for r in pos if r["pred"] != POS],
                    key=lambda r: r["probs"][POS])[:n_each]
    return correct + missed


def fig_examples(out: Path, class_names: list[str]):
    import cv2

    from src.preprocess import unpack_crops

    chosen = pick_examples(2)
    if not chosen:
        print("  r3_examples.png  skipped: no cached test clips")
        return

    fig, axes = plt.subplots(2, len(chosen), figsize=(3.9 * len(chosen), 5.6),
                             gridspec_kw={"height_ratios": [2.1, 1]})
    if len(chosen) == 1:
        axes = axes.reshape(2, 1)

    for col, row in enumerate(chosen):
        stem = Path(row["clip_id"]).stem
        d = dict(np.load(CACHE / f"{stem}.npz", allow_pickle=True))
        crops = unpack_crops(d["crops_data"], d["crops_offsets"])
        comps = np.stack([d["blur_raw"], d["face_area_fraction"],
                          d["mrs_eye_visibility"], d["mrs_motion_consistency"]])
        found = np.asarray(d["landmarks_found"], bool)

        # the frame the model leaned on most, by detector confidence
        conf = np.nan_to_num(np.asarray(d["detector_confidence"], float))
        t = int(np.argmax(conf))
        img = cv2.cvtColor(crops[t], cv2.COLOR_BGR2RGB)
        ax = axes[0, col]
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        ok = row["pred"] == row["true"]
        # Three short lines rather than one long one: at four columns a single
        # line runs into its neighbour.
        ax.set_title(f"clip {stem}\ntrue: {class_names[row['true']]}\n"
                     f"predicted: {class_names[row['pred']]}"
                     f"  ({'correct' if ok else 'missed'})",
                     fontsize=8.5, color=DEEP if ok else MRSC, linespacing=1.45)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(DEEP if ok else MRSC)
            spine.set_linewidth(2)

        ax = axes[1, col]
        ax.plot(np.arange(len(found)), np.asarray(d["mrs_motion_consistency"]),
                marker="o", ms=3, lw=1.4, color=COMP_COLOURS["motion_consistency"],
                label="motion")
        ax.plot(np.arange(len(found)), np.asarray(d["mrs_eye_visibility"]),
                marker="o", ms=3, lw=1.4, color=COMP_COLOURS["eye_visibility"],
                label="eye vis.")
        ax.axvline(t, color=GREY, ls="--", lw=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Sampled frame")
        if col == 0:
            ax.set_ylabel("Component value")
        ax.set_title(f"P(not engaged) = {row['probs'][POS]:.3f}\n"
                     f"landmarks on {int(found.sum())}/{len(found)} frames",
                     fontsize=8, linespacing=1.4)
        if col == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    fig.suptitle("Worked examples from the DAiSEE test partition \u2014 "
                 "the dashed line marks the frame shown above",
                 fontsize=9, color=GREY, y=1.01)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}  {len(chosen)} clips")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out}")

    names = report("fullbin").get("class_names") or ["Not engaged", "Engaged"]
    fig_roc_pr(out / "r3_roc_pr.png")
    fig_confusion(out / "r3_confusion.png")
    fig_ablation(out / "r3_ablation.png")
    fig_data_scale(out / "r3_data_scale.png")
    fig_mrs_weights(out / "r3_mrs_weights.png")
    try:
        fig_shap_per_dim(out / "r3_shap_per_dim.png")
    except Exception as exc:
        print(f"  r3_shap_per_dim.png  FAILED: {exc!r}")
    fig_examples(out / "r3_examples.png", list(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
