"""Robustness experiment (spec section 33).

Builds degraded copies of the TEST split in a separate cache, then evaluates the
already-trained checkpoints under each degradation. The original dataset files
are never modified.

Two arms are compared, both trained on clean data:

    baseline  : ViT -> Mamba, no reliability signal      (--mrs-mode none arm)
    proposed  : MRS -> ViT -> MRS weighting -> Mamba     (the full model)

The hypothesis being tested is that reliability weighting helps *more* as the
input degrades, because MRS should notice the degradation. Whether it actually
does is what the numbers say; the existence of an MRS term is not evidence of
robustness (spec section 33 says this explicitly).

Also reports how MRS itself responds to each corruption, which is a direct check
of whether the reliability score is measuring what it claims to.

Writes artifacts/robustness_results.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruptions import CORRUPTIONS, build, describe  # noqa: E402
from src.dataset import load_calibration  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.evaluate import evaluate_split  # noqa: E402
from src.mrs import mrs_from_arrays  # noqa: E402
from src.preprocess import (  # noqa: E402
    run_stage1, run_stage2, stage1_key, stage1_path, stage2_key,
)
from src.utils import Timer, get_device, get_logger, load_config, save_json  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


def mrs_summary(cfg, index, s1_key, calibration, split="test") -> dict:
    vals = []
    for rec in index.clips.get(split, []):
        p = stage1_path(cfg, s1_key, split, rec.stem)
        if not p.exists():
            continue
        d = np.load(p)
        vals.append(mrs_from_arrays(d["blur_raw"], d["face_area_fraction"],
                                    d["detector_confidence"], d["head_pose_deg"],
                                    d["mrs_eye_visibility"], d["mrs_motion_consistency"],
                                    dict(cfg["mrs"]), calibration))
    if not vals:
        return {}
    a = np.concatenate(vals)
    faces = []
    for rec in index.clips.get(split, []):
        p = stage1_path(cfg, s1_key, split, rec.stem)
        if p.exists():
            faces.append(np.load(p)["face_found"])
    face_rate = float(np.concatenate(faces).mean()) if faces else None
    return {"mrs_mean": float(a.mean()), "mrs_std": float(a.std()),
            "mrs_min": float(a.min()), "mrs_max": float(a.max()),
            "face_detection_rate": face_rate}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS))
    ap.add_argument("--baseline-checkpoint", default="checkpoints/abl_no_mrs_s42/best.pt")
    ap.add_argument("--proposed-checkpoint", default="checkpoints/abl_full_s42/best.pt")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("robustness", "logs/run_robustness.log")
    cfg = load_config(args.config)
    index = build_index(cfg)
    device = get_device()
    calibration = load_calibration(cfg)

    for name, path in (("baseline", args.baseline_checkpoint),
                       ("proposed", args.proposed_checkpoint)):
        if not (Path(__file__).resolve().parent.parent / path).is_file():
            log.error("%s checkpoint not found: %s. Run scripts/run_ablation.py first.",
                      name, path)
            return 1

    encoder = ViTFrameEncoder(str(cfg["vit"]["model_name"]), pretrained=True,
                              freeze=True, chunk_size=int(cfg["vit"]["batch_size"])
                              ).to(device.device)
    spec = encoder.spec

    print("=" * 96)
    print("ROBUSTNESS EXPERIMENT  (degraded TEST split; original files untouched)")
    print("=" * 96)
    print(f"backbone : {spec.name}")
    print(f"baseline : {args.baseline_checkpoint}  (no reliability signal)")
    print(f"proposed : {args.proposed_checkpoint}  (MRS weighting + reliability pooling)\n")

    rows = []
    for corruption in args.corruptions:
        transform = build(corruption)
        name = None if corruption == "clean" else corruption
        s1 = stage1_key(cfg, name)
        with Timer() as t1:
            s1_summary = run_stage1(cfg, index, splits=("test",), logger=None,
                                    frame_transform=transform, corruption_name=name)
        s2 = stage2_key(cfg, s1, spec.name, spec.input_size)
        with Timer() as t2:
            s2_summary = run_stage2(cfg, index, encoder, s1, splits=("test",),
                                    device=device.device, logger=None)

        stats = mrs_summary(cfg, index, s1, calibration)
        row = {"corruption": corruption, "description": describe(corruption),
               "stage1_key": s1, "stage2_key": s2,
               "stage1_seconds": round(t1.elapsed, 1), "stage2_seconds": round(t2.elapsed, 1),
               "stage1_failures": s1_summary["counts"]["failed"],
               "stage2_failures": s2_summary["counts"]["failed"],
               "mrs": stats}

        for arm, ckpt in (("baseline", args.baseline_checkpoint),
                          ("proposed", args.proposed_checkpoint)):
            # mrs_mode is left to the checkpoint: each arm is evaluated with the
            # reliability setting it was actually trained under.
            res = evaluate_split(cfg, index, s1, s2, ckpt, split="test",
                                 run_name=f"robust_{arm}",
                                 artifact_prefix=f"robust_{arm}_{corruption}",
                                 plots=False, per_clip=False)
            m = res["metrics"]
            row[arm] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        "weighted_f1": m["weighted_f1"]}

        rows.append(row)
        print(f"{corruption:<18} MRS mean={stats.get('mrs_mean', float('nan')):.3f} "
              f"face_det={100*stats.get('face_detection_rate', 0):.1f}%  |  "
              f"baseline acc={row['baseline']['accuracy']:.3f} mF1={row['baseline']['macro_f1']:.3f}"
              f"  |  proposed acc={row['proposed']['accuracy']:.3f} "
              f"mF1={row['proposed']['macro_f1']:.3f}")

    # ------------------------------------------------------------------- table
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"{'corruption':<18}{'MRS mean':>10}{'face det':>10}"
          f"{'base acc':>10}{'base mF1':>10}{'prop acc':>10}{'prop mF1':>10}{'d mF1':>9}")
    clean = next((r for r in rows if r["corruption"] == "clean"), None)
    for r in rows:
        d = r["proposed"]["macro_f1"] - r["baseline"]["macro_f1"]
        print(f"{r['corruption']:<18}{r['mrs'].get('mrs_mean', 0):>10.3f}"
              f"{100*r['mrs'].get('face_detection_rate', 0):>9.1f}%"
              f"{r['baseline']['accuracy']:>10.3f}{r['baseline']['macro_f1']:>10.3f}"
              f"{r['proposed']['accuracy']:>10.3f}{r['proposed']['macro_f1']:>10.3f}"
              f"{d:>+9.3f}")

    if clean:
        print("\nDegradation relative to clean (proposed arm):")
        for r in rows:
            if r["corruption"] == "clean":
                continue
            drop = clean["proposed"]["macro_f1"] - r["proposed"]["macro_f1"]
            mrs_drop = clean["mrs"]["mrs_mean"] - r["mrs"].get("mrs_mean", 0)
            print(f"  {r['corruption']:<18} macro-F1 drop {drop:+.3f}   MRS drop {mrs_drop:+.3f}")

    save_json({
        "protocol": "Degradations applied in memory to decoded frames during Stage 1; the "
                    "DAiSEE files on disk are unmodified. Both arms were trained on CLEAN "
                    "data only and are evaluated unchanged under each degradation.",
        "backbone": spec.as_dict(),
        "baseline_checkpoint": args.baseline_checkpoint,
        "proposed_checkpoint": args.proposed_checkpoint,
        "results": rows,
        "caveat": "A single test split of 36 clips from 8 subjects. Differences of a few "
                  "hundredths of macro-F1 are not distinguishable from noise at this size.",
    }, "artifacts/robustness_results.json")
    print("\nwrote artifacts/robustness_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
