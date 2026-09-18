"""SHAP explainability (spec section 32).

Attributes over the representation entering fusion, so the landmark half keeps
named features and can be grouped into eye openness, iris/gaze, head pose, brow,
mouth/jaw and face scale.

Writes artifacts/shap_analysis_<run>.json plus two plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import CachedClipDataset, collate, load_calibration  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.explain import run_shap  # noqa: E402
from src.preprocess import stage1_key, stage2_key  # noqa: E402
from src.utils import get_device, get_logger, load_config, set_seed  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/main/best.pt")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="which split to explain; the checkpoint is not changed either way")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("shap", "logs/run_shap.log")
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    index = build_index(cfg)
    dev_info = get_device()
    device = torch.device(dev_info.device)

    spec = ViTFrameEncoder(str(cfg["vit"]["model_name"]), pretrained=True, freeze=True).spec
    s1 = stage1_key(cfg)
    s2 = stage2_key(cfg, s1, spec.name, spec.input_size)

    ckpt_path = Path(__file__).resolve().parent.parent / args.checkpoint
    if not ckpt_path.is_file():
        log.error("checkpoint not found: %s. Train a model first.", ckpt_path)
        return 1

    model, ckpt = load_checkpoint(cfg, args.checkpoint, device)
    calibration = load_calibration(cfg)
    mrs_mode = ckpt.get("mrs_mode")

    train_ds = CachedClipDataset(cfg, index, "train", s1, s2, calibration, mrs_mode=mrs_mode)
    eval_ds = CachedClipDataset(cfg, index, args.split, s1, s2, calibration, mrs_mode=mrs_mode)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False, collate_fn=collate)

    print("=" * 78)
    print("SHAP ANALYSIS")
    print("=" * 78)
    print(f"checkpoint  : {args.checkpoint} (epoch {ckpt.get('epoch')})")
    print(f"explaining  : {args.split} split, {len(eval_ds)} clips")
    print(f"background  : train split, {len(train_ds)} clips")

    result = run_shap(model, train_loader, eval_loader, device, cfg,
                      out_prefix=f"{args.run_name}_{args.split}")

    print(f"\nattribution input : {result['attribution_input']}")
    print(f"background/eval   : {result['background_samples']} / {result['evaluated_samples']}")
    print(f"\n{'group':<22}{'dims':>7}{'share of total':>17}{'per-dimension':>16}")
    ordered = sorted(result["group_share_of_total"].items(), key=lambda kv: kv[1], reverse=True)
    for group, share in ordered:
        print(f"{group:<22}{result['group_sizes'][group]:>7}{100*share:>16.2f}%"
              f"{result['group_share_per_dimension'][group]:>16.5f}")

    print("\ntop 15 named landmark features by mean |SHAP|:")
    for item in result["top_landmark_features"][:15]:
        print(f"  {item['feature']:<40}{item['mean_abs_shap']:.5f}")

    print(f"\nplots: {result['plots']['groups']}")
    print(f"       {result['plots']['top_features']}")
    print(f"\nCAVEAT: {result['caveat']}")
    print(f"\nwrote artifacts/shap_analysis_{args.run_name}_{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
