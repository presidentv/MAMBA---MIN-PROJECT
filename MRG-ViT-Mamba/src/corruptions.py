"""Controlled degradations for the robustness experiment (spec section 33).

Corruptions are applied **in memory** to decoded frames during Stage 1. The
original DAiSEE files on disk are never touched, and each corrupted sweep gets
its own cache key so a degraded cache can never be mistaken for a clean one.

Each corruption is deterministic given (clip, frame index) so that a rerun
reproduces exactly the same degraded input.
"""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(sigma: float):
    def apply(frame: np.ndarray) -> np.ndarray:
        k = max(3, int(2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(frame, (k, k), sigma)
    return apply


def downscale(factor: float):
    """Reduce resolution then restore the original size, losing detail."""
    def apply(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, int(w * factor)), max(1, int(h * factor))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return apply


def occlusion(fraction: float, seed: int = 0):
    """Grey rectangle covering ``fraction`` of the frame area, at a fixed location.

    The position is derived from a seeded RNG created per corruption rather than
    per frame, so the occluder is stable within a clip -- which is what a real
    occluder (a hand, a mug, a badly placed webcam) looks like.
    """
    rng = np.random.default_rng(seed)
    pos = (float(rng.uniform(0.15, 0.55)), float(rng.uniform(0.15, 0.55)))

    def apply(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        side = int(np.sqrt(fraction * h * w))
        x0 = int(pos[0] * (w - side))
        y0 = int(pos[1] * (h - side))
        out = frame.copy()
        out[y0:y0 + side, x0:x0 + side] = 128
        return out
    return apply


def brightness_contrast(alpha: float, beta: float):
    """out = alpha * frame + beta   (alpha is contrast gain, beta a brightness shift)."""
    def apply(frame: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return apply


def jpeg_artifacts(quality: int):
    def apply(frame: np.ndarray) -> np.ndarray:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else frame
    return apply


# name -> (factory, description). Severities are chosen to span "barely
# noticeable" to "clearly damaging" rather than to make any method look good.
CORRUPTIONS: dict[str, tuple] = {
    "clean":            (None, "no degradation (reference)"),
    "blur_s2":          (lambda: gaussian_blur(2.0), "Gaussian blur, sigma=2"),
    "blur_s4":          (lambda: gaussian_blur(4.0), "Gaussian blur, sigma=4"),
    "downscale_4x":     (lambda: downscale(0.25), "resolution reduced 4x then restored"),
    "downscale_8x":     (lambda: downscale(0.125), "resolution reduced 8x then restored"),
    "occlusion_10pct":  (lambda: occlusion(0.10, seed=1), "grey box over 10% of the frame"),
    "occlusion_25pct":  (lambda: occlusion(0.25, seed=1), "grey box over 25% of the frame"),
    "dark":             (lambda: brightness_contrast(0.5, -30), "halved contrast, darkened"),
    "bright":           (lambda: brightness_contrast(1.4, 45), "raised contrast, brightened"),
    "jpeg_q15":         (lambda: jpeg_artifacts(15), "heavy JPEG compression, quality 15"),
}


def build(name: str):
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; choose from {sorted(CORRUPTIONS)}")
    factory, _ = CORRUPTIONS[name]
    return factory() if factory else None


def describe(name: str) -> str:
    return CORRUPTIONS[name][1]
