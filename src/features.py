"""Step 1 -- gaze feature engineering from the Phase 2 landmark tables.

Reads ``outputs/phase2_landmarks/<split>/<subject>/<clip>/landmarks.parquet`` and
produces, per clip, both

  * a per-frame feature sequence (the fusion transformer needs a sequence), and
  * clip-level aggregate statistics (mean/std/min/max plus rate scalars, which
    the logistic-regression baseline and the writeup need).

Feature groups, matching the project plan:

  fixation proxies    I-VT classification of each inter-frame gaze shift into
                      fixation vs saccade, plus fixation ratio and mean
                      fixation run length.
  gaze dispersion     rolling-window spread of the gaze point (I-DT dispersion
                      and per-axis standard deviations).
  blink rate          EAR-based. SUBSTITUTION: the project plan called for
                      OpenFace AU45 (blink), but OpenFace is not installed and
                      its Windows build is a heavy non-pip dependency. The eye
                      aspect ratio computed from the MediaPipe eye contours is
                      the standard substitute (Soukupova & Cech 2016); the
                      threshold is per-clip adaptive rather than a global
                      constant, because EAR baseline varies with face shape and
                      eyewear.
  off-screen ratio    fraction of frames whose combined head+iris gaze angle
                      exceeds a screen-plausibility bound.
  head pose stability rolling standard deviation of yaw / pitch / roll.

Run standalone:
    python src/features.py --output-root outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GazeFeatureConfig, load_experiment_config  # noqa: E402

logger = logging.getLogger("features")

# The per-frame columns that make up the gaze modality, in a fixed order. The
# dataset class relies on this order being stable across clips.
GAZE_FEATURE_COLUMNS: Tuple[str, ...] = (
    "gaze_yaw",
    "gaze_pitch",
    "gaze_velocity",
    "is_fixation",
    "gaze_dispersion",
    "gaze_std_yaw",
    "gaze_std_pitch",
    "ear",
    "ear_normalised",
    "is_blink",
    "blink_rate_window",
    "off_screen",
    "head_std_yaw",
    "head_std_pitch",
    "head_std_roll",
    "yaw",
    "pitch",
    "roll",
    "iris_rel_x",
    "iris_rel_y",
    "mrs",
)


def _rolling(series: pd.Series, window: int, fn: str) -> pd.Series:
    return getattr(series.rolling(window=window, min_periods=1, center=True), fn)()


def compute_gaze_features(
    landmarks: pd.DataFrame, cfg: GazeFeatureConfig
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return ``(per_frame_features, clip_level_aggregates)`` for one clip.

    ``landmarks`` must be the Phase 2 table for a single clip, sorted by
    ``sample_index``.
    """
    df = landmarks.sort_values("sample_index").reset_index(drop=True)
    n = len(df)
    out = pd.DataFrame(index=range(n))

    out["sample_index"] = df["sample_index"].to_numpy()
    out["frame_index"] = df["frame_index"].to_numpy()
    out["timestamp"] = df["timestamp"].to_numpy()
    out["mrs"] = df["mrs"].to_numpy() if "mrs" in df else 1.0

    # --- raw head pose -------------------------------------------------- #
    for axis in ("yaw", "pitch", "roll"):
        out[axis] = df[axis].to_numpy(dtype=float)

    # --- gaze direction = head pose + eye-in-socket rotation ------------- #
    # iris_rel_x is 0 at the outer eye corner and 1 at the inner corner, so it
    # is mirrored between the two eyes; centring at 0.5 and averaging gives a
    # signed "eye turned nasally/temporally" measure. iris_rel_y is already an
    # offset perpendicular to the eye axis, normalised by eye width.
    left_rel_x = df["left_iris_rel_x"].to_numpy(dtype=float) - 0.5
    right_rel_x = df["right_iris_rel_x"].to_numpy(dtype=float) - 0.5
    # Right eye's inner corner points the other way, hence the sign flip.
    iris_rel_x = np.nanmean(np.vstack([left_rel_x, -right_rel_x]), axis=0)
    iris_rel_y = np.nanmean(
        np.vstack(
            [df["left_iris_rel_y"].to_numpy(dtype=float), df["right_iris_rel_y"].to_numpy(dtype=float)]
        ),
        axis=0,
    )
    out["iris_rel_x"] = iris_rel_x
    out["iris_rel_y"] = iris_rel_y
    out["gaze_yaw"] = out["yaw"] + cfg.iris_gain_deg * iris_rel_x
    out["gaze_pitch"] = out["pitch"] + cfg.iris_gain_deg * iris_rel_y

    # --- angular velocity and I-VT fixation classification --------------- #
    dt = np.diff(out["timestamp"].to_numpy(dtype=float), prepend=np.nan)
    dyaw = np.diff(out["gaze_yaw"].to_numpy(dtype=float), prepend=np.nan)
    dpitch = np.diff(out["gaze_pitch"].to_numpy(dtype=float), prepend=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity = np.hypot(dyaw, dpitch) / dt
    velocity[~np.isfinite(velocity)] = 0.0
    out["gaze_velocity"] = velocity
    # See the docstring caveat: at 5 fps this separates "gaze shifted between
    # samples" from "gaze held", not true saccades from true fixations.
    out["is_fixation"] = (velocity < cfg.saccade_velocity_threshold).astype(float)

    # --- gaze dispersion over a rolling window --------------------------- #
    window = max(2, int(cfg.rolling_window))
    yaw_s, pitch_s = out["gaze_yaw"], out["gaze_pitch"]
    # I-DT dispersion: (max-min) summed over axes within the window.
    dispersion = (
        _rolling(yaw_s, window, "max") - _rolling(yaw_s, window, "min")
    ) + (_rolling(pitch_s, window, "max") - _rolling(pitch_s, window, "min"))
    out["gaze_dispersion"] = dispersion.fillna(0.0).to_numpy()
    out["gaze_std_yaw"] = _rolling(yaw_s, window, "std").fillna(0.0).to_numpy()
    out["gaze_std_pitch"] = _rolling(pitch_s, window, "std").fillna(0.0).to_numpy()

    # --- blink detection from EAR (AU45 substitute) ---------------------- #
    ear = df["mean_ear"].to_numpy(dtype=float)
    out["ear"] = ear
    baseline = float(np.nanmedian(ear)) if np.isfinite(ear).any() else 0.0
    out["ear_normalised"] = ear / baseline if baseline > 1e-9 else 0.0
    closed = out["ear_normalised"].to_numpy() < cfg.blink_ear_ratio

    # Count a blink only on the falling edge, so one long closure that spans
    # several samples is one blink, not three.
    is_blink = np.zeros(n, dtype=float)
    last_blink = -10_000
    for i in range(n):
        if closed[i] and (i == 0 or not closed[i - 1]) and (i - last_blink) > cfg.blink_min_separation:
            is_blink[i] = 1.0
            last_blink = i
    out["is_blink"] = is_blink

    duration = float(out["timestamp"].iloc[-1] - out["timestamp"].iloc[0]) if n > 1 else 0.0
    fps = (n - 1) / duration if duration > 0 else 5.0
    # Blinks per minute inside the rolling window.
    out["blink_rate_window"] = (
        pd.Series(is_blink).rolling(window=window, min_periods=1, center=True).sum()
        * (60.0 * fps / window)
    ).to_numpy()

    # --- off-screen ------------------------------------------------------ #
    off = (out["gaze_yaw"].abs() > cfg.off_screen_yaw_deg) | (
        out["gaze_pitch"].abs() > cfg.off_screen_pitch_deg
    )
    out["off_screen"] = off.astype(float)

    # --- head pose stability --------------------------------------------- #
    for axis in ("yaw", "pitch", "roll"):
        out[f"head_std_{axis}"] = _rolling(out[axis], window, "std").fillna(0.0).to_numpy()

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ------------------------------------------------------------------ #
    # Clip-level aggregates
    # ------------------------------------------------------------------ #
    aggregates: Dict[str, float] = {}
    for column in GAZE_FEATURE_COLUMNS:
        values = out[column].to_numpy(dtype=float)
        aggregates[f"{column}_mean"] = float(np.mean(values))
        aggregates[f"{column}_std"] = float(np.std(values))
        aggregates[f"{column}_min"] = float(np.min(values))
        aggregates[f"{column}_max"] = float(np.max(values))

    total_blinks = float(is_blink.sum())
    aggregates.update(
        {
            "n_frames": float(n),
            "duration_s": duration,
            "fixation_ratio": float(out["is_fixation"].mean()),
            "mean_fixation_run": _mean_run_length(out["is_fixation"].to_numpy() > 0.5),
            "saccade_rate_per_s": float((1.0 - out["is_fixation"].mean()) * fps),
            "blink_count": total_blinks,
            "blink_rate_per_min": (total_blinks / duration * 60.0) if duration > 0 else 0.0,
            "off_screen_ratio": float(out["off_screen"].mean()),
            "ear_baseline": baseline,
            "head_stability_yaw": float(np.std(out["yaw"])),
            "head_stability_pitch": float(np.std(out["pitch"])),
            "head_stability_roll": float(np.std(out["roll"])),
            "gaze_dispersion_mean": float(out["gaze_dispersion"].mean()),
        }
    )
    return out, aggregates


def _mean_run_length(mask: np.ndarray) -> float:
    """Mean length of consecutive True runs; 0 when there are none."""
    if mask.size == 0 or not mask.any():
        return 0.0
    runs: List[int] = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def discover_landmark_files(phase2_root: Path, splits) -> List[Tuple[Path, str, str, str]]:
    found = []
    for split in splits:
        split_dir = phase2_root / split
        if not split_dir.is_dir():
            logger.warning("No Phase 2 output for split %s", split)
            continue
        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for clip_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                for name in ("landmarks.parquet", "landmarks.csv"):
                    candidate = clip_dir / name
                    if candidate.is_file():
                        found.append((candidate, split, subject_dir.name, clip_dir.name))
                        break
    return found


def build_gaze_features(output_root: Path, cfg: GazeFeatureConfig, splits) -> pd.DataFrame:
    phase2_root = Path(output_root) / "phase2_landmarks"
    if not phase2_root.is_dir():
        raise SystemExit(f"Phase 2 output missing at {phase2_root}. Run phase2_landmarks.py first.")

    files = discover_landmark_files(phase2_root, splits)
    if not files:
        raise SystemExit(f"No landmark tables found under {phase2_root}")
    logger.info("Computing gaze features for %d clips", len(files))

    feature_root = Path(output_root) / "features" / "gaze"
    clip_rows: List[Dict[str, object]] = []
    skipped: List[str] = []

    label_columns = ["Boredom", "Engagement", "Confusion", "Frustration"]
    for path, split, subject_id, clip_id in files:
        table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if len(table) < cfg.min_frames:
            logger.warning("Skipping %s: only %d frames (< %d)", clip_id, len(table), cfg.min_frames)
            skipped.append(clip_id)
            continue

        per_frame, aggregates = compute_gaze_features(table, cfg)
        per_frame.insert(0, "ClipID", clip_id)
        per_frame.insert(1, "SubjectID", subject_id)
        per_frame.insert(2, "split", split)
        # Carry the clip's labels onto every frame row. The dataset class reads
        # them from here, so leaving them only on the clip-level CSV would make
        # every clip look unlabelled.
        for column in label_columns:
            if column in table.columns:
                value = table[column].iloc[0]
                per_frame[column] = pd.NA if pd.isna(value) else int(value)

        out_dir = feature_root / split / subject_id / clip_id
        out_dir.mkdir(parents=True, exist_ok=True)
        per_frame.to_parquet(out_dir / "gaze.parquet", index=False)

        row: Dict[str, object] = {
            "ClipID": clip_id,
            "SubjectID": subject_id,
            "split": split,
            "gaze_file": str((out_dir / "gaze.parquet").relative_to(output_root)),
        }
        row.update(aggregates)
        for column in label_columns:
            if column in table.columns:
                value = table[column].iloc[0]
                row[column] = None if pd.isna(value) else int(value)
        clip_rows.append(row)

    clip_df = pd.DataFrame(clip_rows)
    target = Path(output_root) / "features" / "gaze_clip_features.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    clip_df.to_csv(target, index=False)
    logger.info("Wrote %d clip-level gaze rows -> %s", len(clip_df), target)
    if skipped:
        logger.warning("Skipped %d clip(s) with too few frames: %s", len(skipped), skipped)

    manifest = {
        "n_clips": len(clip_df),
        "per_frame_feature_columns": list(GAZE_FEATURE_COLUMNS),
        "n_per_frame_features": len(GAZE_FEATURE_COLUMNS),
        "skipped_clips": skipped,
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
    }
    (Path(output_root) / "features" / "gaze_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return clip_df


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--rolling-window", type=int, default=None)
    parser.add_argument("--saccade-velocity-threshold", type=float, default=None)
    parser.add_argument("--blink-ear-ratio", type=float, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s",
        datefmt="%H:%M:%S",
    )
    experiment = load_experiment_config(args.config)
    overrides = {
        "rolling_window": args.rolling_window,
        "saccade_velocity_threshold": args.saccade_velocity_threshold,
        "blink_ear_ratio": args.blink_ear_ratio,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(experiment.gaze, key, value)

    build_gaze_features(args.output_root, experiment.gaze, experiment.splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
