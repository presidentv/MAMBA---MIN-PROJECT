"""Step 3 -- PyTorch Dataset / DataLoader over the per-clip feature sequences.

One item == one clip:

    gaze     (max_len, n_gaze)    padded/truncated gaze feature sequence
    affect   (max_len, n_affect)  padded/truncated affect feature sequence
    mask     (max_len,)           True where the timestep is real, False = padding
    label    scalar long          Engagement, 0-3, the primary target
    aux      (3,) long            Boredom / Confusion / Frustration, kept for later
    meta     dict                 ClipID, SubjectID, split, n_frames

The two modalities are joined on ``sample_index``, not on position: both derive
from the same Phase 1 retained-frame list, but an explicit inner join means a
frame the mesh dropped in one branch can never silently shift the other.

Split handling: the Train/Validation/Test assignment comes from the DAiSEE
directory layout and is never re-shuffled. :func:`verify_split_integrity` asserts
that no subject appears in more than one split and raises if that is violated.

Standardisation uses TRAIN statistics only, fitted once and reused for
validation and test, so no target-split information leaks into the scaler.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from affect import AFFECT_PROB_COLUMNS, EMBEDDING_DIM
from config import DataConfig
from features import GAZE_FEATURE_COLUMNS

logger = logging.getLogger("datasets")

LABEL_COLUMNS = ("Boredom", "Engagement", "Confusion", "Frustration")


@dataclass
class ClipSample:
    """One clip's assembled feature sequences, before tensorisation."""

    clip_id: str
    subject_id: str
    split: str
    gaze: np.ndarray      # (T, n_gaze)
    affect: np.ndarray    # (T, n_affect)
    labels: Dict[str, int]
    sample_indices: np.ndarray


def affect_columns(feature_set: str) -> List[str]:
    """Column names for the requested affect representation."""
    if feature_set == "probs":
        return list(AFFECT_PROB_COLUMNS)
    embedding = [f"emb_{i:04d}" for i in range(EMBEDDING_DIM)]
    if feature_set == "embedding":
        return embedding
    if feature_set == "both":
        return list(AFFECT_PROB_COLUMNS) + embedding
    raise ValueError(f"Unknown affect_feature_set {feature_set!r}; use probs|embedding|both")


def load_clip_samples(
    output_root: Path, splits: Sequence[str], affect_feature_set: str = "probs"
) -> List[ClipSample]:
    """Assemble every clip that has BOTH a gaze and an affect feature table."""
    output_root = Path(output_root)
    gaze_root = output_root / "features" / "gaze"
    affect_root = output_root / "features" / "affect"
    if not gaze_root.is_dir() or not affect_root.is_dir():
        raise SystemExit(
            "Feature tables missing. Run src/features.py and src/affect.py first "
            f"(looked in {gaze_root} and {affect_root})."
        )

    wanted_affect = affect_columns(affect_feature_set)
    samples: List[ClipSample] = []
    skipped: List[str] = []

    for split in splits:
        split_dir = gaze_root / split
        if not split_dir.is_dir():
            logger.warning("No gaze features for split %s", split)
            continue
        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for clip_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                clip_id, subject_id = clip_dir.name, subject_dir.name
                gaze_file = clip_dir / "gaze.parquet"
                affect_file = affect_root / split / subject_id / clip_id / "affect.parquet"
                if not gaze_file.is_file() or not affect_file.is_file():
                    skipped.append(f"{split}/{clip_id} (missing modality)")
                    continue

                gaze_df = pd.read_parquet(gaze_file)
                affect_df = pd.read_parquet(
                    affect_file, columns=["sample_index"] + wanted_affect
                )
                merged = gaze_df.merge(affect_df, on="sample_index", how="inner", validate="1:1")
                merged = merged.sort_values("sample_index")
                if merged.empty:
                    skipped.append(f"{split}/{clip_id} (empty after join)")
                    continue

                labels: Dict[str, int] = {}
                for column in LABEL_COLUMNS:
                    if column in gaze_df.columns and not pd.isna(gaze_df[column].iloc[0]):
                        labels[column] = int(gaze_df[column].iloc[0])
                if "Engagement" not in labels:
                    skipped.append(f"{split}/{clip_id} (no Engagement label)")
                    continue

                samples.append(
                    ClipSample(
                        clip_id=clip_id,
                        subject_id=subject_id,
                        split=split,
                        gaze=merged[list(GAZE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32),
                        affect=merged[wanted_affect].to_numpy(dtype=np.float32),
                        labels=labels,
                        sample_indices=merged["sample_index"].to_numpy(),
                    )
                )

    if skipped:
        logger.warning("Skipped %d clip(s): %s", len(skipped), skipped[:10])
    logger.info(
        "Loaded %d clips (gaze dim %d, affect dim %d)",
        len(samples),
        samples[0].gaze.shape[1] if samples else 0,
        samples[0].affect.shape[1] if samples else 0,
    )
    return samples


def verify_split_integrity(samples: Sequence[ClipSample]) -> Dict[str, object]:
    """Assert no subject spans two splits; DAiSEE is split by subject by design.

    Raises rather than warns: a subject leaking across splits would make every
    reported number meaningless, and that must not be recoverable by accident.
    """
    by_subject: Dict[str, set] = {}
    for sample in samples:
        by_subject.setdefault(sample.subject_id, set()).add(sample.split)
    offenders = {s: sorted(v) for s, v in by_subject.items() if len(v) > 1}
    if offenders:
        raise ValueError(
            f"Subject leakage across splits: {offenders}. "
            "The DAiSEE folder split must be subject-disjoint."
        )
    report = {
        "n_subjects": len(by_subject),
        "subjects_per_split": {
            split: sorted({s.subject_id for s in samples if s.split == split})
            for split in sorted({s.split for s in samples})
        },
        "leakage": {},
    }
    logger.info(
        "Split integrity OK: %d subjects, none shared across splits", len(by_subject)
    )
    return report


def class_distribution(samples: Sequence[ClipSample], target: str = "Engagement") -> pd.DataFrame:
    """Per-split class counts -- printed up front so tiny classes are visible."""
    rows = [
        {"split": s.split, "class": s.labels.get(target), "ClipID": s.clip_id} for s in samples
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["split", "class"])["ClipID"].count().unstack(fill_value=0).sort_index()
    )


class FeatureScaler:
    """Z-score scaler fitted on TRAIN timesteps only."""

    def __init__(self) -> None:
        self.gaze_mean: Optional[np.ndarray] = None
        self.gaze_std: Optional[np.ndarray] = None
        self.affect_mean: Optional[np.ndarray] = None
        self.affect_std: Optional[np.ndarray] = None

    def fit(self, samples: Sequence[ClipSample]) -> "FeatureScaler":
        gaze = np.concatenate([s.gaze for s in samples], axis=0)
        affect = np.concatenate([s.affect for s in samples], axis=0)
        self.gaze_mean, self.gaze_std = gaze.mean(0), gaze.std(0)
        self.affect_mean, self.affect_std = affect.mean(0), affect.std(0)
        # A constant feature has zero variance; dividing by 1 leaves it at 0
        # after centring rather than producing inf/nan.
        self.gaze_std[self.gaze_std < 1e-6] = 1.0
        self.affect_std[self.affect_std < 1e-6] = 1.0
        return self

    def transform(self, sample: ClipSample) -> ClipSample:
        if self.gaze_mean is None:
            return sample
        return ClipSample(
            clip_id=sample.clip_id,
            subject_id=sample.subject_id,
            split=sample.split,
            gaze=((sample.gaze - self.gaze_mean) / self.gaze_std).astype(np.float32),
            affect=((sample.affect - self.affect_mean) / self.affect_std).astype(np.float32),
            labels=sample.labels,
            sample_indices=sample.sample_indices,
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            gaze_mean=self.gaze_mean, gaze_std=self.gaze_std,
            affect_mean=self.affect_mean, affect_std=self.affect_std,
        )


class EngagementClipDataset(Dataset):
    """Sequence dataset over clips, padded to a fixed length with a mask."""

    def __init__(
        self,
        samples: Sequence[ClipSample],
        max_sequence_length: int = 50,
        target: str = "Engagement",
        auxiliary_targets: Sequence[str] = ("Boredom", "Confusion", "Frustration"),
    ) -> None:
        self.samples = list(samples)
        self.max_len = int(max_sequence_length)
        self.target = target
        self.auxiliary_targets = list(auxiliary_targets)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def gaze_dim(self) -> int:
        return self.samples[0].gaze.shape[1] if self.samples else 0

    @property
    def affect_dim(self) -> int:
        return self.samples[0].affect.shape[1] if self.samples else 0

    def labels(self) -> np.ndarray:
        return np.array([s.labels[self.target] for s in self.samples], dtype=np.int64)

    def _pad(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        length = min(array.shape[0], self.max_len)
        padded = np.zeros((self.max_len, array.shape[1]), dtype=np.float32)
        padded[:length] = array[:length]
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:length] = True
        return padded, mask

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        gaze, mask = self._pad(sample.gaze)
        affect, _ = self._pad(sample.affect)
        aux = [sample.labels.get(name, -1) for name in self.auxiliary_targets]
        return {
            "gaze": torch.from_numpy(gaze),
            "affect": torch.from_numpy(affect),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(sample.labels[self.target], dtype=torch.long),
            "aux": torch.tensor(aux, dtype=torch.long),
            "clip_id": sample.clip_id,
            "subject_id": sample.subject_id,
            "split": sample.split,
            "n_frames": int(min(sample.gaze.shape[0], self.max_len)),
        }


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack tensors, keep metadata as plain Python lists."""
    return {
        "gaze": torch.stack([b["gaze"] for b in batch]),
        "affect": torch.stack([b["affect"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "aux": torch.stack([b["aux"] for b in batch]),
        "clip_id": [b["clip_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
        "split": [b["split"] for b in batch],
        "n_frames": [b["n_frames"] for b in batch],
    }


def build_dataloaders(
    output_root: Path, cfg: DataConfig, splits: Sequence[str]
) -> Tuple[Dict[str, DataLoader], Dict[str, EngagementClipDataset], Dict[str, object]]:
    """Assemble datasets and loaders for every split, with integrity checks."""
    samples = load_clip_samples(output_root, splits, cfg.affect_feature_set)
    if not samples:
        raise SystemExit("No clips with both modalities and a label were found.")

    integrity = verify_split_integrity(samples)
    distribution = class_distribution(samples, cfg.target)

    by_split: Dict[str, List[ClipSample]] = {split: [] for split in splits}
    for sample in samples:
        by_split.setdefault(sample.split, []).append(sample)

    scaler = FeatureScaler()
    if cfg.standardize:
        train_samples = by_split.get("Train", [])
        if not train_samples:
            raise SystemExit("Cannot standardise: no Train clips found.")
        scaler.fit(train_samples)
        by_split = {k: [scaler.transform(s) for s in v] for k, v in by_split.items()}

    datasets: Dict[str, EngagementClipDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    for split, split_samples in by_split.items():
        if not split_samples:
            continue
        dataset = EngagementClipDataset(
            split_samples, cfg.max_sequence_length, cfg.target, cfg.auxiliary_targets
        )
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == "Train"),
            collate_fn=collate,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

    info: Dict[str, object] = {
        "integrity": integrity,
        "class_distribution": distribution.to_dict() if not distribution.empty else {},
        "gaze_dim": datasets[next(iter(datasets))].gaze_dim,
        "affect_dim": datasets[next(iter(datasets))].affect_dim,
        "n_clips_per_split": {k: len(v) for k, v in datasets.items()},
        "scaler": scaler,
    }
    return loaders, datasets, info


def summarise_class_distribution(
    datasets: Dict[str, EngagementClipDataset], num_classes: int = 4
) -> Tuple[pd.DataFrame, List[str]]:
    """Return the per-split class table plus explicit warnings for tiny classes.

    On a sample this small, a class with one or two clips in the test split makes
    its per-class F1 essentially a coin flip. Those cases are surfaced as
    warnings so results are never read as if they were stable.
    """
    rows = []
    warnings: List[str] = []
    for split, dataset in datasets.items():
        labels = dataset.labels()
        counts = np.bincount(labels, minlength=num_classes)
        rows.append({"split": split, **{f"class_{i}": int(counts[i]) for i in range(num_classes)},
                     "total": int(len(labels))})
        for class_index, count in enumerate(counts):
            if count == 0:
                warnings.append(f"{split}: class {class_index} has NO clips -- metrics for it are undefined.")
            elif count < 5:
                warnings.append(
                    f"{split}: class {class_index} has only {count} clip(s) -- "
                    "its per-class F1 is not statistically meaningful."
                )
    return pd.DataFrame(rows).set_index("split"), warnings
