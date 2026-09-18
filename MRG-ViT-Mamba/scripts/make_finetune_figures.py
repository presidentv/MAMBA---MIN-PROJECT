"""Report figures for the fine-tuned T=32 run.

Reuses the plotting functions from scripts/make_report_figures.py so the two
runs are drawn in the same visual language and can be put side by side in the
report, and adds two comparisons that only exist once there are two runs:

  * frozen T=16 vs fine-tuned T=32 across the three splits;
  * the learned MRS weights reached by each regime.

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
from src.mrs import COMPONENTS  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ft32")
    ap.add_argument("--baseline", default="mini64",
                    help="frozen-backbone run to compare against; '' to skip")
    ap.add_argument("--out-dir", default="artifacts/report_ft32")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    report = json.loads(
        Path(f"artifacts/finetune_report_{args.run}.json").read_text(encoding="utf-8"))
    rows = load_history(args.run)
    best_epoch = int(report["best_epoch"])

    written = []
    written.append(fig_training_curves(rows, best_epoch,
                                       Path(out_dir) / "ft1_training_curves.png"))
    written.append(fig_weight_evolution(rows, best_epoch,
                                        Path(out_dir) / "ft2_mrs_weight_evolution.png"))
    for split in ("val", "test"):
        written.append(fig_per_class(
            report["splits"][split], Path(out_dir) / f"ft3_per_class_{split}.png",
            f"Per-class {split} metrics, fine-tuned T=32 "
            f"({report['splits'][split]['num_samples']} clips)"))

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
