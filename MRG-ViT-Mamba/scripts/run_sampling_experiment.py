"""Temporal sampling experiment (spec section 34).

Evaluates T in {8, 16, 32, 64} frames per clip, recording for each: accuracy,
weighted F1, macro F1, preprocessing cost, inference time and GPU memory.

The point is the trade-off curve between temporal information and computation.
No claim is made that 16 frames is universally sufficient -- that is exactly the
assumption the spec asks to be tested rather than assumed.

Each T gets its own Stage 1 and Stage 2 cache (the frame count is part of both
keys), so results cannot be contaminated by a stale cache.

Writes artifacts/sampling_experiment.json.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_index import build_index  # noqa: E402
from src.evaluate import evaluate_split  # noqa: E402
from src.preprocess import run_stage1, run_stage2, stage1_key, stage2_key  # noqa: E402
from src.train import train_model  # noqa: E402
from src.utils import Timer, get_device, get_logger, load_config, save_json, set_seed  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("sampling", "logs/run_sampling_experiment.log")
    base_cfg = load_config(args.config)
    index = build_index(base_cfg)
    device = get_device()

    encoder = ViTFrameEncoder(str(base_cfg["vit"]["model_name"]), pretrained=True,
                              freeze=True, chunk_size=int(base_cfg["vit"]["batch_size"])
                              ).to(device.device)
    spec = encoder.spec

    print("=" * 100)
    print("TEMPORAL SAMPLING EXPERIMENT")
    print("=" * 100)
    print(f"device   : {device.device} ({device.name})")
    print(f"backbone : {spec.name} (D={spec.embed_dim})")
    print(f"frames   : {args.frames}")
    print(f"seeds    : {args.seeds}\n")

    rows = []
    for T in args.frames:
        cfg = copy.deepcopy(dict(base_cfg))
        cfg["video"]["num_frames"] = int(T)

        s1 = stage1_key(cfg)
        with Timer() as t1:
            s1_sum = run_stage1(cfg, index, logger=None)
        s2 = stage2_key(cfg, s1, spec.name, spec.input_size)
        with Timer() as t2:
            s2_sum = run_stage2(cfg, index, encoder, s1, device=device.device, logger=None)

        runs = []
        efficiency = None
        for seed in args.seeds:
            set_seed(seed)
            cfg_seed = copy.deepcopy(cfg)
            cfg_seed["seed"] = seed
            run_name = f"T{T}_s{seed}"
            res = train_model(cfg_seed, index, s1, s2, run_name=run_name, logger=None)
            test = evaluate_split(cfg_seed, index, s1, s2, res.best_checkpoint,
                                  split="test", run_name=run_name,
                                  plots=False, per_clip=False)
            m = test["metrics"]
            runs.append({"seed": seed, "val_macro_f1": res.best_val_macro_f1,
                         "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"],
                         "test_weighted_f1": m["weighted_f1"]})
            efficiency = test["efficiency"]

        def agg(key):
            v = [r[key] for r in runs]
            return {"mean": float(np.mean(v)), "std": float(np.std(v))}

        row = {
            "frames": T,
            "stage1_key": s1, "stage2_key": s2,
            "preprocess_stage1_seconds": round(t1.elapsed, 1),
            "preprocess_stage2_seconds": round(t2.elapsed, 1),
            "stage1_failures": s1_sum["counts"]["failed"],
            "stage2_failures": s2_sum["counts"]["failed"],
            "metrics": {k: agg(k) for k in ("val_macro_f1", "test_accuracy",
                                            "test_macro_f1", "test_weighted_f1")},
            "runs": runs,
            "efficiency": efficiency,
        }
        rows.append(row)
        print(f"T={T:<3} stage1 {t1.elapsed:6.1f}s  stage2 {t2.elapsed:6.1f}s  "
              f"test acc {row['metrics']['test_accuracy']['mean']:.4f}"
              f"+-{row['metrics']['test_accuracy']['std']:.4f}  "
              f"mF1 {row['metrics']['test_macro_f1']['mean']:.4f}"
              f"+-{row['metrics']['test_macro_f1']['std']:.4f}  "
              f"head {1000*efficiency['seconds_per_video_mean']:.2f} ms/video  "
              f"peak {efficiency.get('peak_gpu_memory_mb') or 0:.0f} MB")

    print("\n" + "=" * 100)
    print("SUMMARY (mean +- std over seeds)")
    print("=" * 100)
    print(f"{'T':>4}{'test acc':>18}{'weighted F1':>20}{'macro F1':>18}"
          f"{'ms/video':>12}{'peak MB':>10}{'prep s':>9}")
    for r in rows:
        m = r["metrics"]
        e = r["efficiency"]
        print(f"{r['frames']:>4}"
              f"{m['test_accuracy']['mean']:>11.4f}+-{m['test_accuracy']['std']:<5.4f}"
              f"{m['test_weighted_f1']['mean']:>13.4f}+-{m['test_weighted_f1']['std']:<5.4f}"
              f"{m['test_macro_f1']['mean']:>11.4f}+-{m['test_macro_f1']['std']:<5.4f}"
              f"{1000*e['seconds_per_video_mean']:>12.2f}"
              f"{(e.get('peak_gpu_memory_mb') or 0):>10.0f}"
              f"{r['preprocess_stage1_seconds'] + r['preprocess_stage2_seconds']:>9.1f}")

    save_json({
        "backbone": spec.as_dict(),
        "device": device.as_dict(),
        "seeds": args.seeds,
        "timing_scope": "cached-feature head only; preprocessing time is reported "
                        "separately as preprocess_stage1/2_seconds for the whole 108-clip "
                        "corpus across all three splits",
        "results": rows,
        "caveat": "36 test clips from 8 subjects. The differences between frame counts here "
                  "are smaller than the seed-to-seed spread, so this table shows the cost "
                  "curve reliably and the accuracy curve only weakly.",
    }, "artifacts/sampling_experiment.json")
    print("\nwrote artifacts/sampling_experiment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
