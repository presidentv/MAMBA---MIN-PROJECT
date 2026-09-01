"""Dataset for MRG-VM: reliable face crops + geometric features + MRS + label.

One item is one clip:

    frames     (T, 3, H, W)  normalised aligned face crops from Phase 1
    mrs        (T,)          per-frame Motion Reliability Score
    geometric  (T, D_g)      landmark geometric + gaze features from Phase 2
    mask       (T,)          True where the frame is real
    label      scalar        Engagement 0-3

Frames are decoded once into an in-RAM uint8 cache at construction. The sample's
5,384 retained frames at 112x112 come to ~200 MB as uint8, which is far cheaper
than re-decoding PNGs every epoch; normalisation to float happens per batch.

Split assignment always comes from the DAiSEE directory layout and is never
re-shuffled, and subject-disjointness is asserted, exactly as in src/datasets.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import MRGVMDataConfig
from .geometric import GEOMETRIC_FEATURE_COLUMNS, compute_geometric_features

logger = logging.getLogger("mrgvm.data")

LABEL_COLUMNS = ("Boredom", "Engagement", "Confusion", "Frustration")

# The five MRS components, carried per frame from Phase 1. Order must match
# reliability.SUBSCORE_NAMES -- the conditioner indexes this vector positionally.
MRS_SUBSCORE_COLUMNS: Tuple[str, ...] = (
    "blur", "face_visibility", "head_rotation", "eye_visibility", "motion_consistency",
)

# Gaze descriptors produced by src/features.py, folded into the geometric stream.
GAZE_FEATURE_SUBSET: Tuple[str, ...] = (
    "gaze_yaw", "gaze_pitch", "gaze_velocity", "is_fixation", "gaze_dispersion",
    "gaze_std_yaw", "gaze_std_pitch", "ear", "ear_normalised", "is_blink",
    "blink_rate_window", "off_screen", "head_std_yaw", "head_std_pitch",
    "head_std_roll", "yaw", "pitch", "roll", "iris_rel_x", "iris_rel_y",
)


@dataclass
class MRGVMClip:
    clip_id: str
    subject_id: str
    split: str
    frames: np.ndarray        # (T, H, W, 3) uint8, RGB
    mrs: np.ndarray           # (T,) float32  combined scalar (v1)
    sub_scores: np.ndarray    # (T, 5) float32  the components (v2 conditioner)
    geometric: np.ndarray     # (T, D_g) float32
    labels: Dict[str, int]


def discover_clips(output_root: Path, splits: Sequence[str]) -> List[Tuple[str, str, Path, Path, Path]]:
    """Find (split, subject, phase1_dir, landmark_file, gaze_file) for each clip."""
    output_root = Path(output_root)
    phase1_root = output_root / "phase1_reliable_frames"
    phase2_root = output_root / "phase2_landmarks"
    gaze_root = output_root / "features" / "gaze"
    if not phase1_root.is_dir() or not phase2_root.is_dir():
        raise SystemExit(
            f"Phases 1-2 outputs required. Looked in {phase1_root} and {phase2_root}."
        )

    found = []
    for split in splits:
        split_dir = phase2_root / split
        if not split_dir.is_dir():
            logger.warning("No Phase 2 output for split %s", split)
            continue
        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for clip_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                landmark_file = clip_dir / "landmarks.parquet"
                if not landmark_file.is_file():
                    continue
                phase1_dir = phase1_root / split / subject_dir.name / clip_dir.name
                gaze_file = gaze_root / split / subject_dir.name / clip_dir.name / "gaze.parquet"
                found.append((split, subject_dir.name, phase1_dir, landmark_file, gaze_file))
    return found


def load_clips(
    output_root: Path, cfg: MRGVMDataConfig, splits: Sequence[str]
) -> Tuple[List[MRGVMClip], List[str]]:
    """Load every clip into memory. Returns ``(clips, geometric_column_names)``."""
    entries = discover_clips(output_root, splits)
    if not entries:
        raise SystemExit("No clips discovered; run Phases 1-2 first.")

    clips: List[MRGVMClip] = []
    geometric_columns: List[str] = []
    skipped: List[str] = []

    for split, subject_id, phase1_dir, landmark_file, gaze_file in entries:
        clip_id = landmark_file.parent.name
        landmarks = pd.read_parquet(landmark_file).sort_values("sample_index").reset_index(drop=True)
        if landmarks.empty:
            skipped.append(f"{clip_id} (empty landmarks)")
            continue

        geometry = compute_geometric_features(landmarks)
        columns = list(GEOMETRIC_FEATURE_COLUMNS)
        geometry["sample_index"] = landmarks["sample_index"].to_numpy()
        # Carry the per-frame MRS through: it drives both reliability-guidance
        # mechanisms, so it must survive the gaze merge below.
        geometry["mrs"] = landmarks["mrs"].to_numpy(dtype=np.float32)
        for column in MRS_SUBSCORE_COLUMNS:
            if column in landmarks.columns:
                geometry[column] = landmarks[column].to_numpy(dtype=np.float32)
            else:
                logger.warning("Sub-score %s missing for %s; filling 1.0", column, clip_id)
                geometry[column] = np.float32(1.0)

        if cfg.include_gaze_features and gaze_file.is_file():
            gaze = pd.read_parquet(gaze_file)
            available = [c for c in GAZE_FEATURE_SUBSET if c in gaze.columns]
            geometry = geometry.merge(
                gaze[["sample_index"] + available], on="sample_index", how="inner", validate="1:1"
            )
            columns += available
        elif cfg.include_gaze_features:
            logger.warning("Gaze features missing for %s; geometry only", clip_id)

        if not geometric_columns:
            geometric_columns = columns

        # ---- frames ---------------------------------------------------- #
        frames_dir = phase1_dir / "frames"
        images: List[np.ndarray] = []
        keep_index: List[int] = []
        for position, sample_index in enumerate(geometry["sample_index"].to_numpy()):
            row = landmarks[landmarks["sample_index"] == sample_index]
            if row.empty:
                continue
            frame_index = int(row["frame_index"].iloc[0])
            stem = f"s{int(sample_index):04d}_f{frame_index:06d}"
            matches = sorted(frames_dir.glob(stem + ".*"))
            if not matches:
                continue
            bgr = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if bgr.shape[0] != cfg_image_size(cfg):
                bgr = cv2.resize(
                    bgr, (cfg_image_size(cfg), cfg_image_size(cfg)), interpolation=cv2.INTER_AREA
                )
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            keep_index.append(position)

        if not images:
            skipped.append(f"{clip_id} (no frame images)")
            continue

        geometry = geometry.iloc[keep_index].reset_index(drop=True)
        mrs = geometry["mrs"].to_numpy(dtype=np.float32)
        sub_scores = geometry[list(MRS_SUBSCORE_COLUMNS)].to_numpy(dtype=np.float32)
        if not cfg.use_mrs_gate:
            # Phase 6 ablation: pretend every retained frame is perfect. Both the
            # scalar and the component vector must be neutralised, or the
            # conditioner would still see the real signal.
            mrs = np.ones_like(mrs)
            sub_scores = np.ones_like(sub_scores)

        labels = {}
        for column in LABEL_COLUMNS:
            if column in landmarks.columns and not pd.isna(landmarks[column].iloc[0]):
                labels[column] = int(landmarks[column].iloc[0])
        if cfg.target not in labels:
            skipped.append(f"{clip_id} (no {cfg.target} label)")
            continue

        clips.append(
            MRGVMClip(
                clip_id=clip_id,
                subject_id=subject_id,
                split=split,
                frames=np.stack(images).astype(np.uint8),
                mrs=mrs,
                sub_scores=sub_scores,
                geometric=geometry[geometric_columns].to_numpy(dtype=np.float32),
                labels=labels,
            )
        )

    if skipped:
        logger.warning("Skipped %d clip(s): %s", len(skipped), skipped[:8])
    total_frames = sum(len(c.frames) for c in clips)
    logger.info(
        "Loaded %d clips, %d frames, geometric dim %d (~%.0f MB cached)",
        len(clips), total_frames, len(geometric_columns),
        sum(c.frames.nbytes for c in clips) / 1e6,
    )
    return clips, geometric_columns


def cfg_image_size(cfg: MRGVMDataConfig) -> int:
    """Image size is owned by the Vision Mamba config; mirrored here for loading."""
    return getattr(cfg, "_image_size", 112)


def verify_split_integrity(clips: Sequence[MRGVMClip]) -> Dict[str, List[str]]:
    by_subject: Dict[str, set] = {}
    for clip in clips:
        by_subject.setdefault(clip.subject_id, set()).add(clip.split)
    offenders = {s: sorted(v) for s, v in by_subject.items() if len(v) > 1}
    if offenders:
        raise ValueError(f"Subject leakage across splits: {offenders}")
    logger.info("Split integrity OK: %d subjects, none shared across splits", len(by_subject))
    return {
        split: sorted({c.subject_id for c in clips if c.split == split})
        for split in sorted({c.split for c in clips})
    }


class GeometricScaler:
    """Z-score scaler for the geometric stream, fitted on TRAIN frames only."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, clips: Sequence[MRGVMClip]) -> "GeometricScaler":
        stacked = np.concatenate([c.geometric for c in clips], axis=0)
        self.mean = stacked.mean(axis=0)
        self.std = stacked.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, clip: MRGVMClip) -> MRGVMClip:
        if self.mean is None:
            return clip
        clip.geometric = ((clip.geometric - self.mean) / self.std).astype(np.float32)
        return clip

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)


class MRGVMDataset(Dataset):
    def __init__(
        self,
        clips: Sequence[MRGVMClip],
        max_frames: int = 50,
        target: str = "Engagement",
        norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.clips = list(clips)
        self.max_frames = max_frames
        self.target = target
        self.norm_mean = np.asarray(norm_mean, dtype=np.float32)
        self.norm_std = np.asarray(norm_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def geometric_dim(self) -> int:
        return self.clips[0].geometric.shape[1] if self.clips else 0

    def labels(self) -> np.ndarray:
        return np.array([c.labels[self.target] for c in self.clips], dtype=np.int64)

    def __getitem__(self, index: int) -> Dict[str, object]:
        clip = self.clips[index]
        length = min(len(clip.frames), self.max_frames)
        height, width = clip.frames.shape[1:3]

        frames = np.zeros((self.max_frames, 3, height, width), dtype=np.float32)
        raw = clip.frames[:length].astype(np.float32) / 255.0
        raw = (raw - self.norm_mean) / self.norm_std
        frames[:length] = raw.transpose(0, 3, 1, 2)

        geometric = np.zeros((self.max_frames, clip.geometric.shape[1]), dtype=np.float32)
        geometric[:length] = clip.geometric[:length]

        mrs = np.zeros(self.max_frames, dtype=np.float32)
        mrs[:length] = clip.mrs[:length]

        sub_scores = np.zeros((self.max_frames, clip.sub_scores.shape[1]), dtype=np.float32)
        sub_scores[:length] = clip.sub_scores[:length]

        mask = np.zeros(self.max_frames, dtype=bool)
        mask[:length] = True

        return {
            "frames": torch.from_numpy(frames),
            "geometric": torch.from_numpy(geometric),
            "mrs": torch.from_numpy(mrs),
            "sub_scores": torch.from_numpy(sub_scores),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(clip.labels[self.target], dtype=torch.long),
            "clip_id": clip.clip_id,
            "subject_id": clip.subject_id,
            "split": clip.split,
        }


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "geometric": torch.stack([b["geometric"] for b in batch]),
        "mrs": torch.stack([b["mrs"] for b in batch]),
        "sub_scores": torch.stack([b["sub_scores"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "clip_id": [b["clip_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
        "split": [b["split"] for b in batch],
    }


def build_dataloaders(
    output_root: Path, cfg: MRGVMDataConfig, splits: Sequence[str], image_size: int
) -> Tuple[Dict[str, DataLoader], Dict[str, MRGVMDataset], Dict[str, object]]:
    setattr(cfg, "_image_size", image_size)
    clips, geometric_columns = load_clips(output_root, cfg, splits)
    subjects = verify_split_integrity(clips)

    by_split: Dict[str, List[MRGVMClip]] = {}
    for clip in clips:
        by_split.setdefault(clip.split, []).append(clip)

    scaler = GeometricScaler()
    if cfg.standardize:
        if "Train" not in by_split:
            raise SystemExit("Cannot standardise: no Train clips.")
        scaler.fit(by_split["Train"])
        for split_clips in by_split.values():
            for clip in split_clips:
                scaler.transform(clip)

    datasets: Dict[str, MRGVMDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    for split, split_clips in by_split.items():
        dataset = MRGVMDataset(
            split_clips, cfg.max_frames, cfg.target, cfg.norm_mean, cfg.norm_std
        )
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=(split == "Train"),
            collate_fn=collate, num_workers=cfg.num_workers, drop_last=False,
        )

    info = {
        "geometric_columns": geometric_columns,
        "geometric_dim": len(geometric_columns),
        "subjects_per_split": subjects,
        "n_clips_per_split": {k: len(v) for k, v in datasets.items()},
        "scaler": scaler,
    }
    return loaders, datasets, info
