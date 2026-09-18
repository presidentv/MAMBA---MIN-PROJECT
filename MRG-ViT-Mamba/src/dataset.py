"""Torch datasets over the Stage A cache.

Training reads cached ViT features, cached landmark features and the cached raw
MRS signals, then combines the MRS on the fly using a blur calibration fitted on
the training split. Nothing here decodes video, so an epoch is cheap and the
temporal/fusion components can be iterated on quickly (spec section 24, Stage B).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .mrs import COMPONENTS, MRSCalibration, component_matrix_from_arrays, mrs_from_arrays
from .preprocess import stage1_path, stage2_path
from .utils import missing_clip_policy, resolve_path


def _enforce_missing_policy(cfg, split: str, n_total: int, missing: list[str],
                            require_all: bool | None, what: str) -> None:
    """Raise if clips lack a cache entry, unless the config tolerates it.

    ``require_all`` given explicitly by a caller wins; left as None, the
    decision comes from ``dataset.missing_clip_policy`` so that one setting
    governs preprocessing, auditing and training alike.
    """
    if not missing:
        return
    allow, max_frac = missing_clip_policy(cfg)
    if require_all is True:
        allow = False
    elif require_all is False:
        allow, max_frac = True, 1.0
    frac = len(missing) / n_total if n_total else 1.0
    preview = f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
    if not allow:
        raise FileNotFoundError(
            f"{len(missing)} clips in split '{split}' have no {what} entry: {preview}. "
            f"Run scripts/run_preprocessing.py first, or set "
            f"dataset.missing_clip_policy: skip to leave unprocessable clips out.")
    if frac > max_frac:
        raise FileNotFoundError(
            f"{len(missing)}/{n_total} clips ({100 * frac:.2f}%) in split '{split}' have no "
            f"{what} entry, above dataset.max_missing_fraction={100 * max_frac:.2f}%. "
            f"That is too many to be a few corrupt videos - preprocessing is probably "
            f"incomplete. First missing: {preview}")
    import warnings
    warnings.warn(
        f"split '{split}': skipping {len(missing)}/{n_total} clips ({100 * frac:.2f}%) with no "
        f"{what} entry (within the {100 * max_frac:.2f}% tolerance): {preview}",
        stacklevel=3)


class CachedClipDataset(Dataset):
    """One item per clip, joining the Stage 1 (faces/MRS) and Stage 2 (ViT) caches.

    Returns:
        vit_features       [T, D]  float32
        mrs                [T]     float32 in [0, 1]
        landmark_features  [3L]    float32
        label              scalar  int64
    """

    def __init__(self, cfg, index, split: str, s1_key: str, s2_key: str,
                 calibration: MRSCalibration,
                 mrs_mode: str | None = None,
                 require_all: bool | None = None):
        self.cfg = cfg
        self.split = split
        self.s1_key = s1_key
        self.s2_key = s2_key
        self.calibration = calibration
        # "real" (default) returns the computed reliability; "none" returns an
        # all-ones vector, which is the no-MRS ablation baseline.
        #
        # This deliberately does NOT read cfg.mrs.weighting_mode. That key selects
        # how the *model* applies reliability to the features; reading it here too
        # coupled the two, so turning off the feature multiplication also silently
        # stopped any reliability reaching the reliability-weighted pooling, making
        # two distinct ablation arms come out identical.
        if mrs_mode not in (None, "real", "none"):
            raise ValueError(f"dataset mrs_mode must be 'real' or 'none', got {mrs_mode!r}")
        self.mrs_mode = mrs_mode or "real"
        self.cfg_mrs = dict(cfg["mrs"])

        self.records = []
        missing = []
        for rec in index.clips.get(split, []):
            p1 = stage1_path(cfg, s1_key, split, rec.stem)
            p2 = stage2_path(cfg, s2_key, split, rec.stem)
            if p1.exists() and p2.exists():
                self.records.append((rec, p1, p2))
            else:
                missing.append(rec.clip_id)
        _enforce_missing_policy(cfg, split, len(index.clips.get(split, [])), missing,
                                require_all, f"cache ({s1_key} / {s2_key})")
        self.missing = missing
        if not self.records:
            raise RuntimeError(f"split '{split}' has no fully cached clips")

        # Discover the dimensions instead of assuming them.
        s1 = np.load(self.records[0][1])
        s2 = np.load(self.records[0][2])
        self.num_frames = int(s2.shape[0])
        self.vit_dim = int(s2.shape[1])
        self.landmark_dim = int(s1["landmark_clip_features"].shape[0])

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> np.ndarray:
        return np.array([rec.engagement for rec, _, _ in self.records], dtype=np.int64)

    def label_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in self.labels():
            counts[int(label)] = counts.get(int(label), 0) + 1
        return counts

    def subjects(self) -> list[str]:
        return [rec.subject_id for rec, _, _ in self.records]

    def landmark_matrix(self) -> np.ndarray:
        """[N, 3L] for fitting the training-split scaler."""
        return np.stack([
            np.load(p1)["landmark_clip_features"] for _, p1, _ in self.records
        ]).astype(np.float32)

    def blur_raw_values(self) -> np.ndarray:
        return np.concatenate([np.load(p1)["blur_raw"] for _, p1, _ in self.records])

    def __getitem__(self, idx: int) -> dict:
        rec, p1, p2 = self.records[idx]
        data = np.load(p1)

        vit = np.asarray(np.load(p2), dtype=np.float32)
        land = np.asarray(data["landmark_clip_features"], dtype=np.float32)

        raw_args = (data["blur_raw"], data["face_area_fraction"], data["detector_confidence"],
                    data["head_pose_deg"], data["mrs_eye_visibility"],
                    data["mrs_motion_consistency"])
        if self.mrs_mode == "none":
            mrs = np.ones(vit.shape[0], dtype=np.float32)
            comps = np.full((vit.shape[0], len(COMPONENTS)), 1.0, dtype=np.float32)
        else:
            mrs = mrs_from_arrays(*raw_args, self.cfg_mrs, self.calibration)
            comps = component_matrix_from_arrays(*raw_args, self.cfg_mrs, self.calibration)

        return {
            "vit_features": torch.from_numpy(vit),
            "mrs": torch.from_numpy(np.asarray(mrs, dtype=np.float32)),
            "mrs_components": torch.from_numpy(np.asarray(comps, dtype=np.float32)),
            "landmark_features": torch.from_numpy(land),
            "label": torch.tensor(int(rec.engagement), dtype=torch.long),
            "clip_id": rec.clip_id,
            "subject_id": rec.subject_id,
        }


class FineTuneClipDataset(Dataset):
    """One item per clip, returning decoded face crops instead of ViT features.

    The Stage 2 cache is a snapshot of a *frozen* backbone's output. The moment
    the backbone is trained, those vectors describe a model that no longer
    exists, so fine-tuning cannot read them. This dataset therefore reads only
    Stage 1 and hands back the crops themselves, normalised with the backbone's
    own mean/std, for the ViT to encode inside the training graph.

    Everything else -- the MRS components, the landmark clip features, the
    labels -- is identical to CachedClipDataset, so the two paths stay
    comparable.

    Returns:
        frames             [T, 3, S, S]  float32, backbone-normalised
        mrs                [T]           float32 in [0, 1]
        mrs_components     [T, 5]        float32
        landmark_features  [3L]          float32
        label              scalar        int64
    """

    def __init__(self, cfg, index, split: str, s1_key: str, spec,
                 calibration: MRSCalibration, mrs_mode: str | None = None,
                 require_all: bool | None = None):
        self.cfg = cfg
        self.split = split
        self.s1_key = s1_key
        self.spec = spec
        self.calibration = calibration
        if mrs_mode not in (None, "real", "none"):
            raise ValueError(f"dataset mrs_mode must be 'real' or 'none', got {mrs_mode!r}")
        self.mrs_mode = mrs_mode or "real"
        self.cfg_mrs = dict(cfg["mrs"])

        self.records = []
        missing = []
        for rec in index.clips.get(split, []):
            p1 = stage1_path(cfg, s1_key, split, rec.stem)
            if p1.exists():
                self.records.append((rec, p1))
            else:
                missing.append(rec.clip_id)
        _enforce_missing_policy(cfg, split, len(index.clips.get(split, [])), missing,
                                require_all, f"Stage 1 cache ({s1_key})")
        self.missing = missing
        if not self.records:
            raise RuntimeError(f"split '{split}' has no cached clips")

        probe = np.load(self.records[0][1])
        self.num_frames = int(probe["crops_offsets"].shape[0] - 1)
        self.landmark_dim = int(probe["landmark_clip_features"].shape[0])
        self.vit_dim = int(spec.embed_dim)

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> np.ndarray:
        return np.array([rec.engagement for rec, _ in self.records], dtype=np.int64)

    def label_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in self.labels():
            counts[int(label)] = counts.get(int(label), 0) + 1
        return counts

    def subjects(self) -> list[str]:
        return [rec.subject_id for rec, _ in self.records]

    def clip_ids(self) -> list[str]:
        return [rec.clip_id for rec, _ in self.records]

    def landmark_matrix(self) -> np.ndarray:
        return np.stack([
            np.load(p1)["landmark_clip_features"] for _, p1 in self.records
        ]).astype(np.float32)

    def blur_raw_values(self) -> np.ndarray:
        return np.concatenate([np.load(p1)["blur_raw"] for _, p1 in self.records])

    def __getitem__(self, idx: int) -> dict:
        from .preprocess import unpack_crops
        from .vit_encoder import preprocess_crops

        rec, p1 = self.records[idx]
        data = np.load(p1)

        crops = unpack_crops(data["crops_data"], data["crops_offsets"])
        frames = preprocess_crops(crops, self.spec)          # [T, 3, S, S] float32
        land = np.asarray(data["landmark_clip_features"], dtype=np.float32)

        raw_args = (data["blur_raw"], data["face_area_fraction"], data["detector_confidence"],
                    data["head_pose_deg"], data["mrs_eye_visibility"],
                    data["mrs_motion_consistency"])
        if self.mrs_mode == "none":
            mrs = np.ones(frames.shape[0], dtype=np.float32)
            comps = np.full((frames.shape[0], len(COMPONENTS)), 1.0, dtype=np.float32)
        else:
            mrs = mrs_from_arrays(*raw_args, self.cfg_mrs, self.calibration)
            comps = component_matrix_from_arrays(*raw_args, self.cfg_mrs, self.calibration)

        return {
            "frames": frames,
            "mrs": torch.from_numpy(np.asarray(mrs, dtype=np.float32)),
            "mrs_components": torch.from_numpy(np.asarray(comps, dtype=np.float32)),
            "landmark_features": torch.from_numpy(land),
            "label": torch.tensor(int(rec.engagement), dtype=torch.long),
            "clip_id": rec.clip_id,
            "subject_id": rec.subject_id,
        }


def collate_finetune(batch: list[dict]) -> dict:
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "mrs": torch.stack([b["mrs"] for b in batch]),
        "mrs_components": torch.stack([b["mrs_components"] for b in batch]),
        "landmark_features": torch.stack([b["landmark_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "clip_id": [b["clip_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
    }


def collate(batch: list[dict]) -> dict:
    return {
        "vit_features": torch.stack([b["vit_features"] for b in batch]),
        "mrs": torch.stack([b["mrs"] for b in batch]),
        "mrs_components": torch.stack([b["mrs_components"] for b in batch]),
        "landmark_features": torch.stack([b["landmark_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "clip_id": [b["clip_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
    }


def load_calibration(cfg) -> MRSCalibration:
    """Load the fitted MRS calibration, or fall back with an explicit warning.

    An uncalibrated mapping is usable but is flagged ``calibrated=False`` and
    carried into every artifact, so a run made before scripts/fit_mrs_stats.py
    can always be told apart from one made after it.
    """
    path = resolve_path(cfg["mrs"]["calibration_file"])
    if Path(path).is_file():
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return MRSCalibration.from_dict(json.load(fh)["calibration"])
    return MRSCalibration.uncalibrated()
