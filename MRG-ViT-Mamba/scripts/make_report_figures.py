"""Generate the figures for the project report from a completed training run.

Every number plotted is read from a committed artifact or log; nothing here
recomputes or estimates anything.

Outputs (artifacts/report/):
    fig1_training_curves.png       loss / accuracy / macro-F1 per epoch
    fig2_mrs_weight_evolution.png  the learned MRS weights across 64 epochs
    fig3_per_class_metrics.png     precision / recall / F1 per engagement class
    fig4_mrs_distributions.png     the five components and the combined score
    fig5_shap_groups.png           feature-group attribution
    fig6_learned_vs_uniform.png    what the learned weighting changed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.mrs import COMPONENTS, MRSCalibration, components_from_arrays, mrs_from_arrays  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.metrics import DEFAULT_CLASS_NAMES  # noqa: E402
from src.preprocess import stage1_key, stage1_path  # noqa: E402
from src.utils import ensure_dir, load_config  # noqa: E402

# One palette for every figure so the report reads as a set.
DEEP, LAND, MRSC = "#1F6F78", "#6B4E9E", "#B4761F"
GREY, LIGHT = "#5A646E", "#C8D0D8"
COMP_COLOURS = {"blur": "#B4761F", "face_visibility": "#1F6F78",
                "head_pose": "#6B4E9E", "eye_visibility": "#7A9E3B",
                "motion_consistency": "#C4553B"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def load_history(run: str) -> list[dict]:
    with open(f"logs/training_history_{run}.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fig_training_curves(rows, best_epoch, out):
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    panels = [
        ("loss", "train_loss", "val_loss", "cross-entropy loss"),
        ("accuracy", "train_accuracy", "val_accuracy", "accuracy"),
        ("macro-F1", "train_macro_f1", "val_macro_f1", "macro-F1"),
    ]
    for ax, (title, ktr, kva, ylab) in zip(axes, panels):
        ax.plot(ep, [float(r[ktr]) for r in rows], color=DEEP, lw=1.7, label="train")
        ax.plot(ep, [float(r[kva]) for r in rows], color=MRSC, lw=1.7, label="validation")
        ax.axvline(best_epoch, color=GREY, ls="--", lw=1.1)
        ax.annotate(f"selected\nepoch {best_epoch}", (best_epoch, ax.get_ylim()[1]),
                    xytext=(4, -4), textcoords="offset points", fontsize=7.5,
                    va="top", color=GREY)
        if title != "loss":
            ax.axhline(0.25, color=LIGHT, lw=1.4, ls=":")
            ax.annotate("chance (0.25)", (ep[-1], 0.25), xytext=(-4, 4),
                        textcoords="offset points", ha="right", fontsize=7.5, color=GREY)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Training and validation over 64 epochs (DAiSEE_mini, 120 train / 80 val clips)",
                 fontsize=11, y=1.04)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_weight_evolution(rows, best_epoch, out):
    ep = [int(r["epoch"]) for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.0),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    for c in COMPONENTS:
        ax.plot(ep, [float(r["w_" + c]) for r in rows], lw=1.9,
                color=COMP_COLOURS[c], label=c.replace("_", " "))
    ax.axhline(0.2, color=GREY, ls="--", lw=1.2)
    ax.annotate("uniform start (0.20)", (0, 0.2), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color=GREY)
    ax.axvline(best_epoch, color=LIGHT, lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("weight  $w_c$  (softmax, sums to 1)")
    ax.set_title("Learned MRS component weights", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_ylim(0, max(0.5, max(float(r["w_face_visibility"]) for r in rows) * 1.15))

    final = {c: float(rows[-1]["w_" + c]) for c in COMPONENTS}
    order = sorted(COMPONENTS, key=lambda c: final[c], reverse=True)
    y = np.arange(len(order))
    ax2.barh(y, [final[c] for c in order], color=[COMP_COLOURS[c] for c in order], height=.62)
    ax2.axvline(0.2, color=GREY, ls="--", lw=1.2)
    ax2.set_yticks(y, [c.replace("_", "\n") for c in order], fontsize=8)
    ax2.invert_yaxis()
    for i, c in enumerate(order):
        d = final[c] - 0.2
        ax2.text(final[c] + .008, i, f"{final[c]:.3f}  ({d:+.3f})", va="center", fontsize=7.8)
    ax2.set_xlim(0, max(final.values()) * 1.42)
    ax2.set_xlabel("final weight")
    ax2.set_title("Final weights vs the 0.20 start", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_per_class(metrics, out, title):
    names = list(DEFAULT_CLASS_NAMES)
    P = [metrics["per_class"][n]["precision"] for n in names]
    R = [metrics["per_class"][n]["recall"] for n in names]
    F = [metrics["per_class"][n]["f1"] for n in names]
    S = [metrics["per_class"][n]["support"] for n in names]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for off, vals, lab, col in ((-w, P, "precision", DEEP), (0, R, "recall", MRSC),
                                (w, F, "F1", LAND)):
        b = ax.bar(x + off, vals, w, label=lab, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1.5)
    ax.axhline(0.25, color=LIGHT, ls=":", lw=1.4)
    ax.annotate("chance", (len(names) - .5, .25), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=7.5, color=GREY)
    ax.set_xticks(x, [f"{n}\n(n={s})" for n, s in zip(names, S)])
    ax.set_ylabel("score"); ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_mrs_distributions(cfg, index, out):
    cal = MRSCalibration.from_dict(
        json.load(open("artifacts/mrs_calibration.json", encoding="utf-8"))["calibration"])
    key = stage1_key(cfg)
    acc = {c: [] for c in COMPONENTS}
    allm = []
    for sp in ("train", "val", "test"):
        for rec in index.clips.get(sp, []):
            p = stage1_path(cfg, key, sp, rec.stem)
            if not p.exists():
                continue
            d = np.load(p)
            a = (d["blur_raw"], d["face_area_fraction"], d["detector_confidence"],
                 d["head_pose_deg"], d["mrs_eye_visibility"], d["mrs_motion_consistency"])
            cm = components_from_arrays(*a, dict(cfg["mrs"]), cal)
            for c in COMPONENTS:
                acc[c].append(cm[c])
            allm.append(mrs_from_arrays(*a, dict(cfg["mrs"]), cal))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.0))
    for ax, name in zip(axes.ravel(), list(COMPONENTS) + ["MRS"]):
        v = np.concatenate(allm) if name == "MRS" else np.concatenate(acc[name])
        col = MRSC if name == "MRS" else COMP_COLOURS[name]
        ax.hist(v, bins=40, range=(0, 1), color=col, edgecolor="white", linewidth=.4)
        ax.set_title(f"{name.replace('_',' ')}\nmean {v.mean():.3f}   std {v.std():.3f}",
                     fontsize=9.5, fontweight="bold" if name == "MRS" else "normal")
        ax.set_xlim(0, 1); ax.set_xlabel("score" if name == "MRS" else "")
    fig.suptitle(f"MRS components over {len(np.concatenate(allm))} sampled frames "
                 f"(equal 0.20 weights)", fontsize=11, y=1.01)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_shap(path, out):
    s = json.load(open(path, encoding="utf-8"))
    share = s["group_share_of_total"]; perdim = s["group_share_per_dimension"]
    order = sorted(share, key=lambda g: share[g], reverse=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    y = np.arange(len(order))
    cols = [DEEP if g == "deep_temporal" else LAND for g in order]
    a1.barh(y, [100 * share[g] for g in order], color=cols, height=.62)
    a1.set_yticks(y, [g.replace("_", " ") for g in order], fontsize=8.5)
    a1.invert_yaxis(); a1.set_xlabel("share of total mean |SHAP|  (%)")
    a1.set_title("Total attribution by group", fontsize=10, fontweight="bold")
    for i, g in enumerate(order):
        a1.text(100 * share[g] + .5, i, f"{100*share[g]:.1f}%  ({s['group_sizes'][g]}d)",
                va="center", fontsize=7.6)
    a1.set_xlim(0, max(100 * v for v in share.values()) * 1.34)

    order2 = sorted(perdim, key=lambda g: perdim[g], reverse=True)
    a2.barh(np.arange(len(order2)), [perdim[g] for g in order2],
            color=[DEEP if g == "deep_temporal" else LAND for g in order2], height=.62)
    a2.set_yticks(np.arange(len(order2)), [g.replace("_", " ") for g in order2], fontsize=8.5)
    a2.invert_yaxis(); a2.set_xlabel("mean |SHAP| per dimension")
    a2.set_title("Per-dimension attribution (size-normalised)", fontsize=10, fontweight="bold")
    fig.suptitle("SHAP attribution over the fusion input  "
                 "(attribution, not causation)", fontsize=11, y=1.03)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_learned_vs_uniform(cfg, index, rows, out):
    cal = MRSCalibration.from_dict(
        json.load(open("artifacts/mrs_calibration.json", encoding="utf-8"))["calibration"])
    key = stage1_key(cfg)
    w = np.array([float(rows[-1]["w_" + c]) for c in COMPONENTS])
    uni, lea = [], []
    for sp in ("train", "val", "test"):
        for rec in index.clips.get(sp, []):
            p = stage1_path(cfg, key, sp, rec.stem)
            if not p.exists():
                continue
            d = np.load(p)
            a = (d["blur_raw"], d["face_area_fraction"], d["detector_confidence"],
                 d["head_pose_deg"], d["mrs_eye_visibility"], d["mrs_motion_consistency"])
            cm = components_from_arrays(*a, dict(cfg["mrs"]), cal)
            M = np.stack([cm[c] for c in COMPONENTS], -1)
            uni.append(M.mean(-1)); lea.append((M * w).sum(-1))
    uni = np.concatenate(uni); lea = np.concatenate(lea)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    a1.hist(uni, bins=45, range=(0, 1), alpha=.62, color=GREY,
            label=f"uniform 0.20  (std {uni.std():.3f})", edgecolor="white", lw=.3)
    a1.hist(lea, bins=45, range=(0, 1), alpha=.72, color=MRSC,
            label=f"learned  (std {lea.std():.3f})", edgecolor="white", lw=.3)
    a1.set_xlabel("MRS"); a1.set_ylabel("frames"); a1.legend(frameon=False, fontsize=8)
    a1.set_title("Reliability distribution", fontsize=10, fontweight="bold")
    a2.scatter(uni, lea, s=5, alpha=.22, color=DEEP, edgecolors="none")
    lim = [0, 1]; a2.plot(lim, lim, color=GREY, ls="--", lw=1.1)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    a2.set_xlabel("MRS with uniform 0.20 weights"); a2.set_ylabel("MRS with learned weights")
    a2.set_title(f"Per-frame agreement  (Pearson r = {np.corrcoef(uni,lea)[0,1]:.3f})",
                 fontsize=10, fontweight="bold")
    fig.suptitle("What the learned weighting changed", fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mini64")
    args = ap.parse_args()

    cfg = load_config(); index = build_index(cfg)
    out = ensure_dir(Path(cfg["paths"]["artifacts_dir"]) / "report")
    rows = load_history(args.run)
    info = json.load(open(f"artifacts/training_{args.run}.json", encoding="utf-8"))
    best = int(info["best"]["epoch"])
    test = json.load(open(f"artifacts/test_metrics_{args.run}_test.json", encoding="utf-8"))

    made = [
        fig_training_curves(rows, best, out / "fig1_training_curves.png"),
        fig_weight_evolution(rows, best, out / "fig2_mrs_weight_evolution.png"),
        fig_per_class(test["metrics"], out / "fig3_per_class_metrics.png",
                      f"Per-class test metrics ({test['num_videos']} clips, "
                      f"{test['num_unique_subjects']} subjects)"),
        fig_mrs_distributions(cfg, index, out / "fig4_mrs_distributions.png"),
        fig_shap(f"artifacts/shap_analysis_{args.run}_test.json",
                 out / "fig5_shap_groups.png"),
        fig_learned_vs_uniform(cfg, index, rows, out / "fig6_learned_vs_uniform.png"),
    ]
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
