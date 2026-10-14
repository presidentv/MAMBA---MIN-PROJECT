"""Training-time augmentation of cached face crops.

Fine-tuning an 86M-parameter ViT on ~120 clips with no augmentation drove
training accuracy to 1.000 by epoch 16 while validation loss rose from 1.33 to
2.69 (logs/training_history_ft32.csv). This module supplies the missing
regularisation.

Two decisions worth stating:

*   **One transform per clip, not per frame.** A clip is a temporal sequence;
    re-cropping or re-lighting each frame independently would inject motion and
    flicker that are not in the data, and the Mamba encoder reads exactly that
    kind of frame-to-frame change. Every frame of a clip gets the same crop box,
    the same photometric shift and the same erase box.

*   **MRS is not recomputed.** The reliability score is cached from the
    unaugmented frames, so it keeps describing the real observation quality of
    the original video rather than the quality of a synthetic brightness shift.
    Augmentation changes what the ViT sees; it does not change how trustworthy
    the underlying observation was.

Horizontal flip is deliberately absent. The landmark branch carries signed
geometry (iris offset x, nose offset x, yaw, left/right asymmetry) computed in
Stage 1 from unflipped frames. Flipping the crops without mirroring those
features would feed the two branches contradictory evidence, and mirroring them
correctly means hand-maintaining a left/right index map for 478 landmarks and
52 blendshapes -- a large surface for a silent error.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ClipAugment:
    """Clip-level augmentation of a [T, H, W, 3] uint8 BGR crop stack."""

    scale: tuple[float, float] = (0.80, 1.0)      # area kept by the random crop
    ratio: tuple[float, float] = (0.90, 1.11)     # aspect jitter of that crop
    brightness: float = 0.20                       # +/- fraction
    contrast: float = 0.20                         # +/- fraction
    erase_prob: float = 0.25
    erase_scale: tuple[float, float] = (0.02, 0.12)

    @classmethod
    def from_config(cls, cfg_aug: dict | None) -> "ClipAugment | None":
        """Build from the ``augmentation`` config block; None when disabled."""
        if not cfg_aug or not bool(cfg_aug.get("enabled", False)):
            return None
        return cls(
            scale=tuple(cfg_aug.get("scale", (0.80, 1.0))),
            ratio=tuple(cfg_aug.get("ratio", (0.90, 1.11))),
            brightness=float(cfg_aug.get("brightness", 0.20)),
            contrast=float(cfg_aug.get("contrast", 0.20)),
            erase_prob=float(cfg_aug.get("erase_prob", 0.25)),
            erase_scale=tuple(cfg_aug.get("erase_scale", (0.02, 0.12))),
        )

    def describe(self) -> dict:
        return {
            "scale": list(self.scale), "ratio": list(self.ratio),
            "brightness": self.brightness, "contrast": self.contrast,
            "erase_prob": self.erase_prob, "erase_scale": list(self.erase_scale),
            "applies_to": "training split only",
            "granularity": "one sampled transform per clip, identical across its frames",
            "horizontal_flip": False,
        }

    # ------------------------------------------------------------------ apply
    def __call__(self, crops: np.ndarray) -> np.ndarray:
        """[T, H, W, 3] uint8 -> [T, H, W, 3] uint8, same shape.

        Uses the module-level numpy RNG, which the DataLoader reseeds per worker
        per epoch, so a clip is augmented differently on every pass.
        """
        if crops.ndim != 4 or crops.shape[-1] != 3:
            raise ValueError(f"expected [T, H, W, 3] uint8, got {crops.shape}")
        t, h, w, _ = crops.shape

        out = self._random_resized_crop(crops, h, w)
        out = self._photometric(out)
        out = self._random_erase(out, h, w)
        assert out.shape == crops.shape and out.dtype == np.uint8
        return out

    def _random_resized_crop(self, crops, h, w):
        area = h * w
        for _ in range(10):
            target = area * np.random.uniform(*self.scale)
            aspect = np.exp(np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1])))
            cw = int(round(np.sqrt(target * aspect)))
            ch = int(round(np.sqrt(target / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                x0 = np.random.randint(0, w - cw + 1)
                y0 = np.random.randint(0, h - ch + 1)
                return np.stack([
                    cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
                    for f in crops
                ])
        return crops  # degenerate aspect draw; leave the clip untouched

    def _photometric(self, crops):
        alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        beta = 255.0 * np.random.uniform(-self.brightness, self.brightness)
        return np.clip(crops.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    def _random_erase(self, crops, h, w):
        if np.random.rand() >= self.erase_prob:
            return crops
        target = h * w * np.random.uniform(*self.erase_scale)
        aspect = np.exp(np.random.uniform(np.log(0.5), np.log(2.0)))
        ew = max(1, min(w, int(round(np.sqrt(target * aspect)))))
        eh = max(1, min(h, int(round(np.sqrt(target / aspect)))))
        x0 = np.random.randint(0, w - ew + 1)
        y0 = np.random.randint(0, h - eh + 1)
        fill = np.random.randint(0, 256)
        out = crops.copy()
        out[:, y0:y0 + eh, x0:x0 + ew, :] = fill
        return out


@dataclass
class CenterCropView:
    """One deterministic view of a clip, for test-time augmentation.

    TTA averages a model's probabilities over several views of the same clip.
    The views must be *deterministic*, or the reported score depends on an RNG
    and the evaluation stops being reproducible -- so this is a fixed centre
    crop at a given scale, not a sample from ClipAugment's distribution. No
    photometric shift and no erasing either: those exist to make training
    harder, and at evaluation time they would only add noise.

    ``scale`` is the fraction of the frame area kept. 1.0 returns the clip
    untouched, which is the view the model is normally evaluated on, so a TTA
    set should always include it.

    Horizontal flip is absent here for the same reason as in ClipAugment: the
    landmark branch carries signed geometry computed from unflipped frames.
    """

    scale: float = 1.0

    def __post_init__(self):
        if not 0.0 < float(self.scale) <= 1.0:
            raise ValueError(f"CenterCropView scale must be in (0, 1], got {self.scale}")

    @staticmethod
    def set_from_config(cfg_eval: dict | None) -> "list[CenterCropView]":
        """Views named by ``evaluation.tta_scales``; empty list when unset.

        An empty list means "no TTA" rather than "one default view", so the
        plain single-pass evaluation stays the default everywhere.
        """
        scales = (cfg_eval or {}).get("tta_scales") or []
        return [CenterCropView(float(s)) for s in scales]

    def describe(self) -> dict:
        return {"view": "center_crop", "scale": float(self.scale)}

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        if crops.ndim != 4 or crops.shape[-1] != 3:
            raise ValueError(f"expected [T, H, W, 3] uint8, got {crops.shape}")
        if self.scale >= 1.0:
            return crops
        t, h, w, _ = crops.shape
        side = np.sqrt(float(self.scale))
        ch, cw = max(1, int(round(h * side))), max(1, int(round(w * side)))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return np.stack([
            cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
            for f in crops
        ])
