"""Fit the MRS calibration on the TRAINING split only (spec Rule 8).

Two of the five MRS signals have no dataset-independent scale:

  * blur      raw variance-of-Laplacian, which depends on camera, codec and crop
  * face size the face-box area fraction, which depends on how far the subject
              sits from a webcam

Both are mapped to [0, 1] through percentiles of the *training* frames. Reading
validation or test clips here would fit a preprocessing statistic on evaluation
data, so this script never opens them.

Writes artifacts/mrs_calibration.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_index import build_index  # noqa: E402
from src.mrs import MRSCalibration  # noqa: E402
from src.preprocess import stage1_key, stage1_path  # noqa: E402
from src.utils import load_config, save_json  # noqa: E402


def _describe(name: str, a: np.ndarray) -> dict:
    return {"name": name, "min": float(a.min()), "p5": float(np.percentile(a, 5)),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    index = build_index(cfg)
    key = stage1_key(cfg)
    low, high = (float(v) for v in cfg["mrs"]["calibration_percentiles"])

    blur, area = [], []
    used, missing = 0, 0
    for rec in index.clips.get("train", []):
        path = stage1_path(cfg, key, "train", rec.stem)
        if not path.exists():
            missing += 1
            continue
        d = np.load(path)
        blur.append(d["blur_raw"])
        area.append(d["face_area_fraction"])
        used += 1
    if not blur:
        print(f"No cached training clips under {key}. Run Stage 1 first.")
        return 1

    blur = np.concatenate(blur)
    area = np.concatenate(area)
    calibration = MRSCalibration.fit(blur, area, percentiles=(low, high))

    print("=" * 72)
    print("MRS CALIBRATION (fitted on the TRAIN split only)")
    print("=" * 72)
    print(f"cache key   : {key}")
    print(f"train clips : {used} used, {missing} missing")
    print(f"frames      : {blur.size}")
    print(f"percentiles : {low} / {high}")

    b_stats = _describe("blur_raw", blur)
    a_stats = _describe("face_area_fraction", area)
    for st in (b_stats, a_stats):
        print(f"\n{st['name']}:")
        print(f"  min={st['min']:.5f} p5={st['p5']:.5f} median={st['median']:.5f} "
              f"p95={st['p95']:.5f} max={st['max']:.5f}")

    print(f"\nfitted bounds:")
    print(f"  blur (log) : [{calibration.blur_log_low:.4f}, {calibration.blur_log_high:.4f}]")
    print(f"  area       : [{calibration.area_low:.5f}, {calibration.area_high:.5f}]")

    b_mapped = calibration.blur_score_array(blur)
    a_mapped = calibration.area_score_array(area)
    print(f"\nmapped B: min={b_mapped.min():.3f} mean={b_mapped.mean():.3f} "
          f"max={b_mapped.max():.3f} std={b_mapped.std():.3f}")
    print(f"mapped area term: min={a_mapped.min():.3f} mean={a_mapped.mean():.3f} "
          f"max={a_mapped.max():.3f} std={a_mapped.std():.3f}")

    warnings = []
    if b_mapped.std() < 0.05:
        warnings.append("mapped blur reliability is nearly constant; B carries little information")
    if a_mapped.std() < 0.05:
        warnings.append("mapped face-area term is nearly constant; F reduces to detector confidence")
    for w in warnings:
        print(f"\nWARNING: {w}")

    save_json({
        "calibration": calibration.to_dict(),
        "fitted_on": {"split": "train", "cache_key": key, "clips": used,
                      "frames": int(blur.size)},
        "raw_statistics": {"blur_raw": b_stats, "face_area_fraction": a_stats},
        "mapped_statistics": {
            "blur": {"min": float(b_mapped.min()), "mean": float(b_mapped.mean()),
                     "max": float(b_mapped.max()), "std": float(b_mapped.std())},
            "area_term": {"min": float(a_mapped.min()), "mean": float(a_mapped.mean()),
                          "max": float(a_mapped.max()), "std": float(a_mapped.std())},
        },
        "warnings": warnings,
        "note": "Percentiles are fitted on training frames only; validation and test "
                "clips are never read by this script.",
    }, cfg["mrs"]["calibration_file"])
    print(f"\nwrote {cfg['mrs']['calibration_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
