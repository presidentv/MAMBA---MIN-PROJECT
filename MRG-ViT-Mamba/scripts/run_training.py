"""Train the MRG-ViT--Mamba model (Stage B) and evaluate on the test split.

Training uses train + validation only. The test split is read once, at the end,
by the evaluation step, using the checkpoint that validation macro-F1 selected.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --run-name no_mrs --mrs-mode none
    python scripts/run_training.py --backbone timm:vit_base_patch16_clip_224.openai
    python scripts/run_training.py --no-test          # stop before touching test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_index import build_index  # noqa: E402
from src.evaluate import evaluate_split  # noqa: E402
from src.preprocess import stage1_key, stage2_key  # noqa: E402
from src.train import train_model  # noqa: E402
from src.utils import Timer, get_logger, load_config, save_json, set_seed  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


def resolve_keys(cfg, backbone: str | None):
    """Resolve the Stage 1/2 cache keys for a backbone without a GPU forward pass."""
    name = backbone or str(cfg["vit"]["model_name"])
    spec = ViTFrameEncoder(name, pretrained=True, freeze=True).spec
    s1 = stage1_key(cfg)
    return s1, stage2_key(cfg, s1, spec.name, spec.input_size), spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--mrs-mode", default=None, choices=[None, "real", "none"],
                    help="dataset-level reliability switch: 'real' (default) uses the "
                         "computed MRS, 'none' feeds an all-ones vector (no-MRS ablation)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-test", action="store_true",
                    help="train and validate only; do not read the test split")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger(f"train.{args.run_name}", f"logs/run_training_{args.run_name}.log")
    cfg = load_config(args.config)
    if args.backbone:
        cfg["vit"]["model_name"] = args.backbone
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    set_seed(seed)

    index = build_index(cfg)
    if not index.is_usable():
        log.error("dataset index has fatal problems; run scripts/audit_dataset.py")
        return 1

    s1, s2, spec = resolve_keys(cfg, args.backbone)
    log.info("run=%s backbone=%s (D=%d, input=%d)", args.run_name, spec.name,
             spec.embed_dim, spec.input_size)
    log.info("cache keys: stage1=%s stage2=%s", s1, s2)

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr

    with Timer() as t:
        result = train_model(cfg, index, s1, s2, run_name=args.run_name,
                             mrs_mode=args.mrs_mode, logger=log, overrides=overrides)
    log.info("training finished in %.1fs; best epoch %d with val macro-F1 %.4f",
             t.elapsed, result.best_epoch, result.best_val_macro_f1)
    log.info("checkpoint: %s", result.best_checkpoint)

    summary = {"run_name": args.run_name, "seed": seed,
               "training_seconds": round(t.elapsed, 1),
               "best_epoch": result.best_epoch,
               "best_val_macro_f1": result.best_val_macro_f1,
               "checkpoint": result.best_checkpoint,
               "backbone": spec.as_dict(),
               "cache_keys": {"stage1": s1, "stage2": s2}}

    if args.no_test:
        log.info("--no-test given: the test split was not read")
        summary["test"] = "NOT RUN"
    else:
        log.info("evaluating the validation-selected checkpoint on the TEST split")
        test_result = evaluate_split(cfg, index, s1, s2, result.best_checkpoint,
                                     split="test", run_name=args.run_name,
                                     mrs_mode=args.mrs_mode, logger=log)
        m = test_result["metrics"]
        summary["test"] = {
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "num_videos": test_result["num_videos"],
            "num_unique_subjects": test_result["num_unique_subjects"],
        }
        print("\n" + "=" * 72)
        print(f"TEST RESULTS - run '{args.run_name}'")
        print("=" * 72)
        print(f"clips            : {test_result['num_videos']}")
        print(f"unique subjects  : {test_result['num_unique_subjects']}")
        print(f"Acc-4            : {m['accuracy']:.4f}")
        print(f"Weighted F1      : {m['weighted_f1']:.4f}")
        print(f"Macro-F1         : {m['macro_f1']:.4f}")
        print("\nper class:")
        for name, d in m["per_class"].items():
            print(f"  {name:<11} P={d['precision']:.3f} R={d['recall']:.3f} "
                  f"F1={d['f1']:.3f} support={d['support']}")
        print(f"\n{m['macro_f1_note']}")
        if m["classes_present_but_never_predicted"]:
            print(f"classes never predicted: {m['classes_present_but_never_predicted']}")
        print("=" * 72)

    save_json(summary, f"artifacts/run_summary_{args.run_name}.json")
    log.info("wrote artifacts/run_summary_%s.json", args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
