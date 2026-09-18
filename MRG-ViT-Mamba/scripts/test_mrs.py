"""TEST 4 + CHECKPOINT 4 - MRS sanity.

Validates every component and the combined score over the whole cached corpus,
plots the distributions, and writes out the highest- and lowest-MRS face crops so
the score can be checked against what the frames actually look like.

Outputs:
    artifacts/mrs_report.json
    artifacts/mrs_histograms.png
    artifacts/mrs_extremes.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.dataset_index import build_index  # noqa: E402
from src.mrs import COMPONENTS, MRSCalibration, components_from_arrays, mrs_from_arrays  # noqa: E402
from src.preprocess import stage1_key, stage1_path, unpack_crops  # noqa: E402
from src.utils import ensure_dir, load_config, load_json, save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    index = build_index(cfg)
    key = stage1_key(cfg)
    cfg_mrs = dict(cfg["mrs"])

    calib_path = Path(__file__).resolve().parent.parent / cfg["mrs"]["calibration_file"]
    if calib_path.is_file():
        calibration = MRSCalibration.from_dict(load_json(calib_path)["calibration"])
    else:
        print("WARNING: no fitted MRS calibration; using the uncalibrated fallback.")
        calibration = MRSCalibration.uncalibrated()

    print("=" * 72)
    print("TEST 4 / CHECKPOINT 4 - MRS SANITY")
    print("=" * 72)
    print(f"cache key            : {key}")
    print(f"calibrated           : {calibration.calibrated}")
    print(f"MRS weights          : {cfg_mrs['weights']}")

    rows: dict[str, list] = {c: [] for c in COMPONENTS}
    mrs_all: list[np.ndarray] = []
    frame_ref: list[tuple[Path, int]] = []
    per_split: dict[str, list] = {}
    problems: list[str] = []
    motion_sources: dict[str, int] = {}
    total_frames = 0

    for split in ("train", "val", "test"):
        split_scores = []
        for rec in index.clips.get(split, []):
            path = stage1_path(cfg, key, split, rec.stem)
            if not path.exists():
                continue
            d = np.load(path)
            raw_args = (d["blur_raw"], d["face_area_fraction"], d["detector_confidence"],
                        d["head_pose_deg"], d["mrs_eye_visibility"], d["mrs_motion_consistency"])
            mrs = mrs_from_arrays(*raw_args, cfg_mrs, calibration)
            comp = components_from_arrays(*raw_args, cfg_mrs, calibration)
            for name, arr in comp.items():
                arr = np.asarray(arr, dtype=np.float64)
                rows[name].append(arr)
                if not np.all(np.isfinite(arr)):
                    problems.append(f"{rec.clip_id}: non-finite {name}")
                if arr.min() < -1e-6 or arr.max() > 1 + 1e-6:
                    problems.append(f"{rec.clip_id}: {name} out of [0,1] "
                                    f"(min={arr.min():.4f} max={arr.max():.4f})")
            if not np.all(np.isfinite(mrs)):
                problems.append(f"{rec.clip_id}: non-finite MRS")
            if mrs.min() < -1e-6 or mrs.max() > 1 + 1e-6:
                problems.append(f"{rec.clip_id}: MRS out of [0,1]")

            for src in d["motion_source"]:
                motion_sources[str(src)] = motion_sources.get(str(src), 0) + 1

            mrs_all.append(mrs)
            split_scores.append(mrs)
            for t in range(len(mrs)):
                frame_ref.append((path, t))
            total_frames += len(mrs)
        if split_scores:
            per_split[split] = np.concatenate(split_scores)

    if total_frames < 100:
        print(f"Only {total_frames} frames available; CHECKPOINT 4 asks for at least 100.")
        return 1

    mrs_flat = np.concatenate(mrs_all)
    comp_flat = {c: np.concatenate(rows[c]) for c in COMPONENTS}

    print(f"\nframes examined      : {total_frames}")
    print(f"\n{'component':<22}{'min':>9}{'mean':>9}{'max':>9}{'std':>9}")
    print("-" * 58)
    for c in COMPONENTS:
        a = comp_flat[c]
        print(f"{c:<22}{a.min():>9.4f}{a.mean():>9.4f}{a.max():>9.4f}{a.std():>9.4f}")
    print("-" * 58)
    print(f"{'MRS':<22}{mrs_flat.min():>9.4f}{mrs_flat.mean():>9.4f}"
          f"{mrs_flat.max():>9.4f}{mrs_flat.std():>9.4f}")

    print("\nper split (MRS):")
    for split, arr in per_split.items():
        print(f"  {split:<6} n={arr.size:<6} min={arr.min():.4f} mean={arr.mean():.4f} "
              f"max={arr.max():.4f} std={arr.std():.4f}")

    print(f"\nmotion measurement sources: {motion_sources}")

    # CHECKPOINT 4 explicitly says: if nearly all frames share one MRS, investigate.
    degenerate = mrs_flat.std() < 0.01
    if degenerate:
        problems.append(f"MRS is nearly constant across the corpus (std={mrs_flat.std():.5f}); "
                        f"the reliability signal would carry no information")
    flat_components = [c for c in COMPONENTS if comp_flat[c].std() < 0.01]
    if flat_components:
        print(f"\nNOTE: these components are nearly constant and contribute almost nothing "
              f"to MRS variation: {flat_components}")

    # ------------------------------------------------------------- histograms
    art = ensure_dir("artifacts")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=130)
    for ax, name in zip(axes.ravel(), list(COMPONENTS) + ["MRS"]):
        data = mrs_flat if name == "MRS" else comp_flat[name]
        ax.hist(data, bins=40, range=(0, 1), color="#3b6ea5", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{name}\nmean={data.mean():.3f} std={data.std():.3f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25, linewidth=0.5)
    fig.suptitle(f"MRS components over {total_frames} cached frames "
                 f"(calibrated={calibration.calibrated})", fontsize=12)
    fig.tight_layout()
    hist_path = art / "mrs_histograms.png"
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"\nwrote {hist_path}")

    # -------------------------------------------------- high / low MRS examples
    order = np.argsort(mrs_flat)
    lowest = order[:8]
    highest = order[-8:][::-1]

    def strip(indices):
        imgs = []
        for i in indices:
            path, t = frame_ref[int(i)]
            d = np.load(path)
            crop = unpack_crops(d["crops_data"], d["crops_offsets"])[t]
            crop = cv2.resize(crop, (160, 160))
            cv2.putText(crop, f"{mrs_flat[int(i)]:.2f}", (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            imgs.append(crop)
        return np.concatenate(imgs, axis=1)

    label_bar = np.full((26, 160 * 8, 3), 255, dtype=np.uint8)
    top = label_bar.copy()
    cv2.putText(top, "LOWEST MRS", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    bot = label_bar.copy()
    cv2.putText(bot, "HIGHEST MRS", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    montage = np.concatenate([top, strip(lowest), bot, strip(highest)], axis=0)
    ex_path = art / "mrs_extremes.png"
    cv2.imwrite(str(ex_path), montage)
    print(f"wrote {ex_path}")

    save_json({
        "cache_key": key,
        "frames_examined": int(total_frames),
        "calibration": calibration.to_dict(),
        "weights": {k: float(v) for k, v in cfg_mrs["weights"].items()},
        "components": {
            c: {"min": float(comp_flat[c].min()), "mean": float(comp_flat[c].mean()),
                "max": float(comp_flat[c].max()), "std": float(comp_flat[c].std())}
            for c in COMPONENTS
        },
        "mrs": {"min": float(mrs_flat.min()), "mean": float(mrs_flat.mean()),
                "max": float(mrs_flat.max()), "std": float(mrs_flat.std()),
                "percentiles": {str(p): float(np.percentile(mrs_flat, p))
                                for p in (1, 5, 25, 50, 75, 95, 99)}},
        "per_split": {s: {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std())}
                      for s, a in per_split.items()},
        "motion_sources": motion_sources,
        "nearly_constant_components": flat_components,
        "problems": problems,
        "histograms": str(hist_path),
        "extremes_image": str(ex_path),
    }, "artifacts/mrs_report.json")
    print("wrote artifacts/mrs_report.json")

    print("\n" + "=" * 72)
    if problems:
        print(f"CHECKPOINT 4 FAILED - {len(problems)} problems:")
        for p in problems[:20]:
            print(f"  - {p}")
        print("=" * 72)
        return 1
    print("CHECKPOINT 4 PASSED (also inspect artifacts/mrs_extremes.png visually)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
