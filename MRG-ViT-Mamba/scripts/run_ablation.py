"""Ablation study with trivial baselines and multiple seeds.

Two things make a single training run uninterpretable on a corpus this small:

  * **No baseline.** A 4-class accuracy of 0.33 sounds poor and a macro-F1 of
    0.26 sounds meaningless until you know what always-predict-the-majority-class
    scores on the same split. Those baselines are computed here.
  * **No variance.** With 36 validation and 36 test clips, one seed's macro-F1
    moves by a lot for reasons that have nothing to do with the architecture.
    Every arm is therefore run over several seeds and reported as mean +- std.

Arms compared:
    full            MRS weighting + Mamba + landmark fusion   (the proposed model)
    no_mrs          identical, with MRS weighting disabled     (r_t = 1)
    residual_mrs    floor + (1-floor) * r_t weighting
    no_landmarks    deep branch only, no MediaPipe fusion
    bidirectional   Vim-style bidirectional temporal scan

Arms are never selected on test. Each arm's checkpoint is chosen by its own
validation macro-F1, and test is reported once per arm as its final evaluation.

Writes artifacts/ablation_results.json.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import CachedClipDataset, load_calibration  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.evaluate import evaluate_split  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.train import train_model  # noqa: E402
from src.utils import get_logger, load_config, save_json, set_seed  # noqa: E402

# Each arm must differ from "full" in exactly one respect, and the difference has
# to be one the architecture can actually express -- see the normalisation note in
# src/temporal_mamba.py, which is why "multiply_only" and "pool_only" exist as
# separate arms rather than being assumed equivalent to "full".
ARMS = {
    "full": {},                                    # multiply + reliability-weighted pooling
    "no_mrs": {"mrs_mode": "none",                 # r_t = 1 everywhere, both paths off
               "cfg": {"mamba": {"pooling": "mean"},
                       "mrs": {"weighting_mode": "none"}}},
    "multiply_only": {"cfg": {"mamba": {"pooling": "mean"}}},      # spec's literal f'_t = r_t f_t
    "pool_only": {"cfg": {"mrs": {"weighting_mode": "none"}}},     # reliability only in pooling
    "residual_mrs": {"cfg": {"mrs": {"weighting_mode": "residual"}}},
    "no_landmarks": {"cfg": {"fusion": {"use_landmark_branch": False}}},
    "bidirectional": {"cfg": {"mamba": {"bidirectional": True}}},
}


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def trivial_baselines(cfg, index, s1, s2, num_classes: int, seed: int) -> dict:
    """Majority-class and stratified-random baselines on val and test.

    These use only the label arrays. The majority class is taken from TRAIN, so
    the baseline is a legitimate predictor and not an oracle.
    """
    calibration = load_calibration(cfg)
    sets = {s: CachedClipDataset(cfg, index, s, s1, s2, calibration)
            for s in ("train", "val", "test")}
    train_counts = sets["train"].label_counts()
    majority = max(train_counts, key=train_counts.get)
    train_labels = sets["train"].labels()
    prior = np.bincount(train_labels, minlength=num_classes) / len(train_labels)

    rng = np.random.default_rng(seed)
    out = {"majority_class_from_train": int(majority),
           "train_prior": prior.tolist()}
    for split in ("val", "test"):
        y = sets[split].labels()
        maj_pred = np.full_like(y, majority)
        # Average the stochastic baseline over many draws so it is a stable
        # reference rather than one lucky sample.
        strat = [compute_metrics(y, rng.choice(num_classes, size=len(y), p=prior), num_classes)
                 for _ in range(200)]
        out[split] = {
            "n": int(len(y)),
            "majority_class": {k: compute_metrics(y, maj_pred, num_classes)[k]
                               for k in ("accuracy", "macro_f1", "weighted_f1")},
            "stratified_random_mean": {
                k: float(np.mean([m[k] for m in strat]))
                for k in ("accuracy", "macro_f1", "weighted_f1")},
            "stratified_random_std": {
                k: float(np.std([m[k] for m in strat]))
                for k in ("accuracy", "macro_f1", "weighted_f1")},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("ablation", "logs/run_ablation.log")
    base_cfg = load_config(args.config)
    index = build_index(base_cfg)

    from scripts.run_training import resolve_keys  # noqa: E402

    s1, s2, spec = resolve_keys(base_cfg, None)
    num_classes = int(base_cfg["dataset"]["num_classes"])

    print("=" * 90)
    print("ABLATION STUDY")
    print("=" * 90)
    print(f"backbone : {spec.name} (D={spec.embed_dim})")
    print(f"seeds    : {args.seeds}")
    print(f"arms     : {args.arms}\n")

    baselines = trivial_baselines(base_cfg, index, s1, s2, num_classes, int(base_cfg["seed"]))
    print("Trivial baselines on the same splits:")
    for split in ("val", "test"):
        b = baselines[split]
        print(f"  {split} (n={b['n']}): majority-class acc={b['majority_class']['accuracy']:.3f} "
              f"macroF1={b['majority_class']['macro_f1']:.3f}   |   "
              f"stratified-random acc={b['stratified_random_mean']['accuracy']:.3f} "
              f"macroF1={b['stratified_random_mean']['macro_f1']:.3f}")
    print()

    results: dict[str, dict] = {}
    for arm in args.arms:
        spec_arm = ARMS[arm]
        cfg = deep_update(dict(base_cfg), spec_arm.get("cfg", {}))
        mrs_mode = spec_arm.get("mrs_mode")
        runs = []
        print("-" * 90)
        print(f"ARM: {arm}   (weighting={mrs_mode or cfg['mrs']['weighting_mode']}, "
              f"pooling={cfg['mamba']['pooling']}, "
              f"landmarks={cfg['fusion']['use_landmark_branch']}, "
              f"bidirectional={cfg['mamba']['bidirectional']})")
        for seed in args.seeds:
            set_seed(seed)
            cfg_seed = deep_update(cfg, {"seed": seed})
            run_name = f"abl_{arm}_s{seed}"
            res = train_model(cfg_seed, index, s1, s2, run_name=run_name,
                              mrs_mode=mrs_mode, logger=None)
            test = evaluate_split(cfg_seed, index, s1, s2, res.best_checkpoint,
                                  split="test", run_name=run_name, mrs_mode=mrs_mode,
                                  plots=False, per_clip=False)
            m = test["metrics"]
            runs.append({
                "seed": seed, "best_epoch": res.best_epoch,
                "val_macro_f1": res.best_val_macro_f1,
                "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"],
                "test_weighted_f1": m["weighted_f1"],
                "test_classes_never_predicted": m["classes_present_but_never_predicted"],
            })
            print(f"  seed {seed}: best_ep={res.best_epoch:>3} "
                  f"val_mF1={res.best_val_macro_f1:.4f}  "
                  f"test acc={m['accuracy']:.4f} mF1={m['macro_f1']:.4f} "
                  f"wF1={m['weighted_f1']:.4f}")

        def agg(key):
            vals = [r[key] for r in runs]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals))}

        summary = {k: agg(k) for k in ("val_macro_f1", "test_accuracy",
                                       "test_macro_f1", "test_weighted_f1")}
        results[arm] = {"runs": runs, "summary": summary,
                        "config": {"mrs_mode": mrs_mode or cfg["mrs"]["weighting_mode"],
                                   "pooling": cfg["mamba"]["pooling"],
                                   "use_landmark_branch": cfg["fusion"]["use_landmark_branch"],
                                   "bidirectional": cfg["mamba"]["bidirectional"]}}
        print(f"  MEAN over {len(runs)} seeds: val mF1={summary['val_macro_f1']['mean']:.4f}"
              f"+-{summary['val_macro_f1']['std']:.4f}  "
              f"test mF1={summary['test_macro_f1']['mean']:.4f}"
              f"+-{summary['test_macro_f1']['std']:.4f}")

    # ------------------------------------------------------------------ table
    print("\n" + "=" * 90)
    print("SUMMARY (mean +- std over seeds)")
    print("=" * 90)
    print(f"{'arm':<16}{'val macro-F1':>22}{'test acc':>20}{'test macro-F1':>22}")
    for arm, r in results.items():
        s = r["summary"]
        print(f"{arm:<16}"
              f"{s['val_macro_f1']['mean']:>14.4f} +-{s['val_macro_f1']['std']:<7.4f}"
              f"{s['test_accuracy']['mean']:>12.4f} +-{s['test_accuracy']['std']:<7.4f}"
              f"{s['test_macro_f1']['mean']:>14.4f} +-{s['test_macro_f1']['std']:<7.4f}")
    b = baselines["test"]
    print(f"{'[majority]':<16}{'-':>22}"
          f"{b['majority_class']['accuracy']:>20.4f}{b['majority_class']['macro_f1']:>22.4f}")
    print(f"{'[random]':<16}{'-':>22}"
          f"{b['stratified_random_mean']['accuracy']:>20.4f}"
          f"{b['stratified_random_mean']['macro_f1']:>22.4f}")

    save_json({
        "protocol": "Each arm is trained per seed; its checkpoint is chosen by that run's own "
                    "validation macro-F1. Test is read once per run as the final evaluation. "
                    "No arm was selected using test performance.",
        "backbone": spec.as_dict(),
        "seeds": args.seeds,
        "baselines": baselines,
        "arms": results,
    }, "artifacts/ablation_results.json")
    print("\nwrote artifacts/ablation_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
