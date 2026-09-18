"""TEST 5 + CHECKPOINT 5 - landmark feature sanity.

Verifies that the MediaPipe branch produces a fixed-width, finite, sanely-ranged
feature row for every cached frame, and records per-feature statistics.

Writes artifacts/landmark_feature_stats.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_index import build_index  # noqa: E402
from src.landmarks import (  # noqa: E402
    BLENDSHAPE_NAMES, CLIP_FEATURE_DIM, CLIP_FEATURE_NAMES, FEATURE_DIM,
    FEATURE_GROUPS, FEATURE_NAMES, GEOMETRIC_FEATURE_NAMES,
)
from src.preprocess import stage1_key, stage1_path  # noqa: E402
from src.utils import load_config, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    index = build_index(cfg)
    key = stage1_key(cfg)

    print("=" * 72)
    print("TEST 5 / CHECKPOINT 5 - LANDMARK FEATURES")
    print("=" * 72)
    print(f"geometric features : {len(GEOMETRIC_FEATURE_NAMES)}")
    print(f"blendshapes        : {len(BLENDSHAPE_NAMES)}")
    print(f"per-frame dim L    : {FEATURE_DIM}")
    print(f"per-clip dim 3L    : {CLIP_FEATURE_DIM}   (mean || std || mean|delta|)")

    per_frame, per_clip = [], []
    found_flags, problems = [], []
    clips = 0
    for split in ("train", "val", "test"):
        for rec in index.clips.get(split, []):
            path = stage1_path(cfg, key, split, rec.stem)
            if not path.exists():
                continue
            d = np.load(path)
            frame_feats = np.asarray(d["landmark_features"], dtype=np.float64)
            clip_feats = np.asarray(d["landmark_clip_features"], dtype=np.float64)

            if frame_feats.shape[1] != FEATURE_DIM:
                problems.append(f"{rec.clip_id}: per-frame dim {frame_feats.shape[1]} != {FEATURE_DIM}")
            if clip_feats.shape[0] != CLIP_FEATURE_DIM:
                problems.append(f"{rec.clip_id}: per-clip dim {clip_feats.shape[0]} != {CLIP_FEATURE_DIM}")
            if not np.all(np.isfinite(frame_feats)):
                problems.append(f"{rec.clip_id}: non-finite per-frame features")
            if not np.all(np.isfinite(clip_feats)):
                problems.append(f"{rec.clip_id}: non-finite per-clip features")

            per_frame.append(frame_feats)
            per_clip.append(clip_feats)
            found_flags.append(np.asarray(d["landmarks_found"], dtype=bool))
            clips += 1

    if not per_frame:
        print("No cached clips found. Run Stage 1 first.")
        return 1

    frames = np.concatenate(per_frame, axis=0)
    clip_matrix = np.stack(per_clip)
    found = np.concatenate(found_flags)

    print(f"\nclips              : {clips}")
    print(f"frames             : {frames.shape[0]}")
    print(f"per-frame matrix   : {frames.shape}")
    print(f"per-clip matrix    : {clip_matrix.shape}")
    print(f"landmarks found    : {found.sum()}/{found.size} ({100*found.mean():.1f}%)")
    print(f"frames with no face: {(~found).sum()}  "
          f"(their feature row is all zeros - the explicit missing-landmark policy)")

    # Blendshape scores are model probabilities and must lie in [0, 1].
    bs = frames[:, len(GEOMETRIC_FEATURE_NAMES):]
    if bs.size:
        lo, hi = float(bs.min()), float(bs.max())
        print(f"\nblendshape range   : [{lo:.4f}, {hi:.4f}]")
        if lo < -1e-6 or hi > 1 + 1e-6:
            problems.append(f"blendshape scores outside [0,1]: [{lo}, {hi}]")

    # Geometry sanity on frames that actually had a face.
    valid = frames[found]
    geo = valid[:, :len(GEOMETRIC_FEATURE_NAMES)]
    print("\n--- geometric features (frames with a detected face) ---")
    print(f"{'feature':<24}{'min':>10}{'mean':>10}{'max':>10}{'std':>10}")
    for i, name in enumerate(GEOMETRIC_FEATURE_NAMES):
        col = geo[:, i]
        print(f"{name:<24}{col.min():>10.4f}{col.mean():>10.4f}{col.max():>10.4f}{col.std():>10.4f}")
        if abs(col).max() > 1e4:
            problems.append(f"{name}: implausible magnitude {abs(col).max():.1f} "
                            f"- a normalisation denominator may be collapsing")

    constant = [FEATURE_NAMES[i] for i in range(FEATURE_DIM) if valid[:, i].std() < 1e-8]
    if constant:
        print(f"\nNOTE: {len(constant)} features are constant across every frame with a face "
              f"and carry no information: {constant[:12]}{' ...' if len(constant) > 12 else ''}")

    stats = {
        "cache_key": key,
        "per_frame_dim": FEATURE_DIM,
        "per_clip_dim": CLIP_FEATURE_DIM,
        "num_geometric": len(GEOMETRIC_FEATURE_NAMES),
        "num_blendshapes": len(BLENDSHAPE_NAMES),
        "clips": clips,
        "frames": int(frames.shape[0]),
        "landmark_detection_rate_pct": round(100.0 * float(found.mean()), 2),
        "frames_without_landmarks": int((~found).sum()),
        "missing_landmark_policy": "feature row set to all zeros; the temporal position is "
                                   "kept and MRS eye-visibility scores that frame 0",
        "aggregation": "per-clip = concat(mean, std, mean absolute temporal difference)",
        "feature_names": list(FEATURE_NAMES),
        "clip_feature_names": list(CLIP_FEATURE_NAMES),
        "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
        "per_frame_statistics": {
            name: {"min": float(valid[:, i].min()), "mean": float(valid[:, i].mean()),
                   "max": float(valid[:, i].max()), "std": float(valid[:, i].std())}
            for i, name in enumerate(FEATURE_NAMES)
        },
        "per_clip_statistics": {
            "min": clip_matrix.min(axis=0).tolist(),
            "mean": clip_matrix.mean(axis=0).tolist(),
            "max": clip_matrix.max(axis=0).tolist(),
            "std": clip_matrix.std(axis=0).tolist(),
        },
        "constant_features": constant,
        "problems": problems,
    }
    save_json(stats, "artifacts/landmark_feature_stats.json")
    print("\nwrote artifacts/landmark_feature_stats.json")

    print("\n" + "=" * 72)
    if problems:
        print(f"CHECKPOINT 5 FAILED - {len(problems)} problems:")
        for p in problems[:20]:
            print(f"  - {p}")
        print("=" * 72)
        return 1
    print("CHECKPOINT 5 PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
