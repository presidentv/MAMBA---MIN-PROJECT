"""Compare two fine-tuning runs: curves, headline metrics, overfitting gap.

Written for the v1 vs v2 recipe comparison, but takes any two run names. The
point is to make the *shape* of the two training runs visible, not only their
best numbers: a run whose validation loss climbs from epoch 3 is a different
object from one that holds, even if the selected checkpoints score alike.

Run:  python scripts/compare_finetune_runs.py --a ft32 --b ft32_v2
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

from scripts.make_report_figures import DEEP, GREY, LIGHT, MRSC, load_history  # noqa: E402
from src.utils import ensure_dir, resolve_path, save_json  # noqa: E402


def _report(run: str) -> dict:
    return json.loads(
        resolve_path(f"artifacts/finetune_report_{run}.json").read_text(encoding="utf-8"))


def _col(rows, key):
    return [float(r[key]) for r in rows]


def fig_compare(rows_a, rows_b, rep_a, rep_b, label_a, label_b, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    panels = [
        ("validation loss", "val_loss", None),
        ("validation macro-F1", "val_macro_f1", 0.25),
        ("train accuracy", "train_accuracy", 0.25),
    ]
    for ax, (title, key, ref) in zip(axes, panels):
        for rows, rep, lab, col in ((rows_a, rep_a, label_a, LIGHT),
                                    (rows_b, rep_b, label_b, DEEP)):
            ep = [int(r["epoch"]) for r in rows]
            ax.plot(ep, _col(rows, key), lw=1.8, color=col, label=lab)
            if key == "val_macro_f1":
                best = int(rep["best_epoch"])
                ax.plot([best], [float(rows[best][key])], "o", color=col, ms=6)
        if ref is not None:
            ax.axhline(ref, color=MRSC, ls=":", lw=1.2)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.legend(frameon=False, fontsize=8)
    axes[1].annotate("dot = selected checkpoint", (0.02, 0.04), xycoords="axes fraction",
                     fontsize=7.5, color=GREY)
    fig.suptitle("Fine-tuning recipe: v1 vs v2", fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="ft32", help="baseline run name")
    ap.add_argument("--b", default="ft32_v2", help="new run name")
    ap.add_argument("--label-a", default="v1 (no augmentation, flat LR)")
    ap.add_argument("--label-b", default="v2 (augment + layer decay + drop-path)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir or f"artifacts/compare_{args.a}_vs_{args.b}")
    rep_a, rep_b = _report(args.a), _report(args.b)
    rows_a, rows_b = load_history(args.a), load_history(args.b)

    def summarise(run, rep, rows):
        best = int(rep["best_epoch"])
        val_mf1 = _col(rows, "val_macro_f1")
        return {
            "run": run,
            "epochs_run": len(rows),
            "best_epoch": best,
            "train": {k: rep["splits"]["train"][k] for k in ("accuracy", "macro_f1", "loss")},
            "val": {k: rep["splits"]["val"][k] for k in ("accuracy", "macro_f1", "loss")},
            "test": {k: rep["splits"]["test"][k] for k in ("accuracy", "macro_f1", "loss")},
            # How far the run drifts after its peak, and how fast it memorises.
            "val_macro_f1_best": max(val_mf1),
            "val_macro_f1_final": val_mf1[-1],
            "val_macro_f1_last10_mean": float(np.mean(val_mf1[-10:])),
            "val_loss_min": min(_col(rows, "val_loss")),
            "val_loss_final": _col(rows, "val_loss")[-1],
            "first_epoch_at_train_acc_1": next(
                (int(r["epoch"]) for r in rows if float(r["train_accuracy"]) >= 0.999), None),
            "generalisation_gap_at_best": (
                rep["splits"]["train"]["accuracy"] - rep["splits"]["val"]["accuracy"]),
            "learned_mrs_weights": rep.get("learned_mrs_weights"),
        }

    a, b = summarise(args.a, rep_a, rows_a), summarise(args.b, rep_b, rows_b)
    fig = fig_compare(rows_a, rows_b, rep_a, rep_b, args.label_a, args.label_b,
                      Path(out_dir) / "recipe_comparison.png")

    out = {"baseline": a, "candidate": b,
           "labels": {"a": args.label_a, "b": args.label_b},
           "figure": str(fig),
           "note": ("Single seed each, 80 validation and 16 test clips. Differences of a "
                    "few hundredths are not separable from run-to-run noise at this size; "
                    "the shape of the validation curve is the more reliable signal.")}
    path = Path(out_dir) / "recipe_comparison.json"
    save_json(out, path)

    w = 34
    print("\n" + "=" * 78)
    print(f"{'':<{w}} {args.a:>18} {args.b:>18}")
    print("=" * 78)
    rows = [
        ("best epoch", "best_epoch", "{}"),
        ("val accuracy @ best", None, None),
        ("val macro-F1 @ best", None, None),
        ("test accuracy @ best", None, None),
        ("test macro-F1 @ best", None, None),
        ("val macro-F1, mean of last 10", "val_macro_f1_last10_mean", "{:.4f}"),
        ("val loss, minimum", "val_loss_min", "{:.4f}"),
        ("val loss, final", "val_loss_final", "{:.4f}"),
        ("first epoch at train acc 1.000", "first_epoch_at_train_acc_1", "{}"),
        ("train-val accuracy gap @ best", "generalisation_gap_at_best", "{:.4f}"),
    ]
    for label, key, fmt in rows:
        if key is None:
            split, metric = label.split()[0], label.split()[1]
            metric = {"accuracy": "accuracy", "macro-F1": "macro_f1"}[metric]
            va, vb = a[split][metric], b[split][metric]
            print(f"{label:<{w}} {va:>18.4f} {vb:>18.4f}")
        else:
            print(f"{label:<{w}} {str(fmt.format(a[key])):>18} {str(fmt.format(b[key])):>18}")
    print("=" * 78)
    print(f"\nwrote {path}\nwrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
