"""Phase 6b -- corruption robustness, the experiment that tests the novelty.

WHY THIS EXISTS
---------------
Every attempt so far to show that the Motion Reliability Score helps has come
back null, and the reason is the corpus rather than the mechanism. DAiSEE is
seated webcam footage in controlled conditions: the Phase 1 gate rejected 1 frame
out of 5,403, and post-gating reliability sits at 0.93 +/- 0.049. A mechanism
that decides *which frames to trust* has nothing to do when every frame is
trustworthy.

So stop asking "does reliability guidance raise accuracy on clean data" -- it
cannot, and that is not what it is for. Ask the question the mechanism is
actually built to answer: **when the video degrades, does knowing which frames
degraded help?**

THE PROTOCOL
------------
Corrupt the test frames at increasing severity, then measure how fast macro-F1
falls under three conditions that share the same corrupted pixels:

    guided   the model recomputes MRS on the corrupted frames, so it can see
             which frames went bad and condition on that
    blind    the same trained weights, but MRS is pinned to 1.0 -- the model is
             told every frame is perfect when it is not
    unguided a separately trained model with no conditioning at all

``guided`` vs ``blind`` isolates the mechanism at inference with training held
exactly constant, which is the cleanest comparison available. ``unguided`` is the
architectural control.

TWO THINGS THAT ARE EASY TO GET WRONG, BOTH OF WHICH THE FIRST DRAFT GOT WRONG
------------------------------------------------------------------------------
1. **The MRS must be recomputed on the corrupted frames.** Keeping the original
   scores would be meaningless: the score would claim the frame is fine, the
   mechanism would have nothing to react to, and the experiment would measure
   nothing.

2. **The landmark stream must degrade too.** The first version of this file
   corrupted only the pixels and left the Phase 2 geometric features untouched,
   which produced an exactly null result -- clean 0.363, severely blurred 0.363.
   The cause is obvious in hindsight: SHAP attributes only 4.5% of the decision
   to appearance, so corrupting appearance alone changes almost nothing, and the
   "robustness" being measured was really the landmark stream being immune by
   construction. That is not a property of the real system, where a blurred
   frame degrades the face mesh as well.

Both are therefore recomputed from the corrupted pixels: the five MRS components
via re-detection, and the 32-dimensional geometric vector via re-running the face
mesh. When the mesh fails outright on a badly degraded frame -- which it does,
and which is the realistic failure -- that frame contributes zeros to the
geometric stream and a floor reliability score, as it would in deployment.

Usage:
    python -m mrgvm.robustness --output-root outputs_ungated \
        --checkpoint outputs_ungated/checkpoints/mrgvm_v2.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from metrics import evaluate  # noqa: E402

from .config import load_mrgvm_config  # noqa: E402
from .data import MRS_SUBSCORE_COLUMNS, MRGVMDataset, collate, build_dataloaders  # noqa: E402
from .geometric import compute_geometric_features  # noqa: E402
from .model import MRGVMModel  # noqa: E402

logger = logging.getLogger("mrgvm.robustness")


# --------------------------------------------------------------------------- #
# Corruptions. Each takes an RGB uint8 frame and a severity in 1..5.
# Chosen to match the failure modes the five MRS components were designed to
# detect, so the experiment probes the score rather than arbitrary noise.
# --------------------------------------------------------------------------- #
def gaussian_blur(frame: np.ndarray, severity: int) -> np.ndarray:
    """Defocus. Targets the `blur` component."""
    sigma = [0.6, 1.2, 2.0, 3.0, 4.5][severity - 1]
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)


def motion_blur(frame: np.ndarray, severity: int) -> np.ndarray:
    """Directional smear, as from fast head movement."""
    size = [3, 5, 9, 13, 19][severity - 1]
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(frame, -1, kernel)


def occlusion(frame: np.ndarray, severity: int) -> np.ndarray:
    """Grey box over part of the face -- a hand, a mug, hair.

    Placed over the centre, where the face is after Phase 1 alignment, so it
    genuinely occludes features rather than background.
    """
    height, width = frame.shape[:2]
    fraction = [0.10, 0.18, 0.28, 0.40, 0.55][severity - 1]
    box_h, box_w = int(height * fraction), int(width * fraction)
    top = max(0, height // 2 - box_h // 2)
    left = max(0, width // 2 - box_w // 2)
    out = frame.copy()
    out[top : top + box_h, left : left + box_w] = 128
    return out


def darkness(frame: np.ndarray, severity: int) -> np.ndarray:
    """Underexposure, as in a poorly lit room."""
    factor = [0.75, 0.55, 0.40, 0.28, 0.18][severity - 1]
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def jpeg_artefacts(frame: np.ndarray, severity: int) -> np.ndarray:
    """Compression damage, as from a low-bandwidth stream."""
    quality = [45, 30, 20, 12, 7][severity - 1]
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return frame
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def sensor_noise(frame: np.ndarray, severity: int) -> np.ndarray:
    """Gaussian sensor noise, as from a cheap webcam at high gain."""
    sigma = [4, 9, 16, 26, 40][severity - 1]
    noisy = frame.astype(np.float32) + np.random.normal(0, sigma, frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


CORRUPTIONS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "occlusion": occlusion,
    "darkness": darkness,
    "jpeg": jpeg_artefacts,
    "sensor_noise": sensor_noise,
}


def corrupt_frames(frames: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    """Apply one corruption to every frame of a clip. ``frames`` is (T,H,W,3) uint8.

    A clip-level fraction is NOT used: partial corruption would confound "the
    mechanism found the bad frames" with "some frames happened to stay clean".
    Whole-clip corruption at a known severity is the cleaner probe.
    """
    rng = np.random.RandomState(seed)
    state = np.random.get_state()
    np.random.seed(rng.randint(0, 2**31 - 1))
    try:
        function = CORRUPTIONS[name]
        return np.stack([function(frame, severity) for frame in frames])
    finally:
        np.random.set_state(state)


# --------------------------------------------------------------------------- #
# MRS recomputation on corrupted pixels -- the heart of the protocol
# --------------------------------------------------------------------------- #
class CorruptedClipScorer:
    """Recompute BOTH the MRS components and the geometric stream from corrupted pixels.

    Reuses the Phase 1 scoring functions and the Phase 2 mesh, so the definitions
    are identical to training and only the pixels differ.
    """

    def __init__(self, model_dir: Optional[Path] = None, blur_reference: float = 150.0,
                 max_flow: float = 8.0, max_head_deviation: float = 60.0) -> None:
        from engagement_pipeline import faces, mrs as mrs_module, pose
        from engagement_pipeline.models import ensure_model

        self._faces = faces
        self._mrs = mrs_module
        self._pose = pose
        self.detector = faces.FaceDetectorWrapper(
            ensure_model("face_detector", model_dir), min_detection_confidence=0.2
        )
        self.landmarker = faces.FaceLandmarkerWrapper(
            ensure_model("face_landmarker", model_dir),
            min_face_detection_confidence=0.2, min_face_presence_confidence=0.2,
        )
        self.blur_reference = blur_reference
        self.max_flow = max_flow
        self.max_head_deviation = max_head_deviation

    def score_clip(self, frames_rgb, geometric_columns):
        """(T,H,W,3) uint8 RGB -> (sub_scores (T,5), geometric (T,D), mesh_failure_rate)."""
        from phase2_landmarks import _derived_eye_features, landmark_column_names

        length = len(frames_rgb)
        sub_scores = np.zeros((length, 5), dtype=np.float32)
        mesh_rows = []
        mesh_failures = 0
        previous = None

        for index, rgb in enumerate(frames_rgb):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detection = self.detector.detect(bgr)

            if detection is None:
                # A detector failure IS the reliability signal, not an error.
                sub_scores[index] = 0.0
                previous = None
            else:
                roi = self._mrs.face_roi(bgr, detection.bbox_xyxy)
                head = self._pose.pose_from_keypoints(
                    detection.keypoints, bgr.shape[1], bgr.shape[0]
                )
                motion, _, _ = self._mrs.motion_consistency_score(bgr, previous, self.max_flow)
                sub_scores[index] = np.array([
                    self._mrs.blur_score(roi, self.blur_reference),
                    self._mrs.face_visibility_score(detection.confidence),
                    self._mrs.head_rotation_score(head.deviation, self.max_head_deviation),
                    self._mrs.eye_visibility_score(
                        detection.right_eye, detection.left_eye,
                        detection.bbox_xyxy, bgr.shape,
                    ),
                    motion,
                ], dtype=np.float32)
                previous = bgr

            # --- geometric stream, recomputed from the corrupted pixels ----- #
            mesh = self.landmarker.detect(bgr)
            if mesh is None:
                mesh_failures += 1
                mesh_rows.append(None)
                continue
            row = dict(zip(landmark_column_names(), mesh.points.reshape(-1).astype(float)))
            row.update(_derived_eye_features(mesh.points))
            estimate = self._pose.pose_from_transformation_matrix(mesh.transformation_matrix)
            row["yaw"], row["pitch"], row["roll"] = estimate.yaw, estimate.pitch, estimate.roll
            mesh_rows.append(row)

        geometric = self._build_geometric(mesh_rows, sub_scores, geometric_columns, length)
        return sub_scores, geometric, mesh_failures / max(length, 1)

    def _build_geometric(self, mesh_rows, sub_scores, geometric_columns, length):
        """Rebuild the geometric vector, zeroing frames where the mesh failed.

        A failed mesh contributes a zero row rather than an imputed one: the real
        system has no landmarks for such a frame, and forward-filling would
        invent information the deployed pipeline would not have.
        """
        valid = [i for i, r in enumerate(mesh_rows) if r is not None]
        out = np.zeros((length, len(geometric_columns)), dtype=np.float32)
        if not valid:
            return out

        frame = pd.DataFrame([mesh_rows[i] for i in valid])
        frame["sample_index"] = valid
        frame["mrs"] = sub_scores[valid].mean(axis=1)
        derived = compute_geometric_features(frame)

        for position, name in enumerate(geometric_columns):
            if name in derived.columns:
                out[valid, position] = derived[name].to_numpy(dtype=np.float32)
            elif name in frame.columns:
                out[valid, position] = frame[name].to_numpy(dtype=np.float32)
        return out

    def close(self) -> None:
        self.detector.close()
        self.landmarker.close()


def build_corruption_bank(
    clips,
    geometric_columns,
    scaler,
    n_variants: int = 2,
    corruptions: Optional[Sequence[str]] = None,
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    seed: int = 1234,
    image_size: int = 112,
):
    """Generate corrupted training variants with reliability recomputed.

    Returns a list of new clips; the caller appends them to the training split.

    Each variant draws one corruption and one severity uniformly, so across the
    bank the model sees the full reliability range rather than the narrow band
    the clean corpus provides. Validation and test are never augmented -- the
    point is to teach the conditioner what unreliable input looks like, not to
    make the evaluation easier.
    """
    import copy as _copy

    corruptions = list(corruptions or CORRUPTIONS)
    rng = np.random.RandomState(seed)
    scorer = CorruptedClipScorer()
    out = []
    try:
        for clip_index, clip in enumerate(clips):
            for variant in range(n_variants):
                name = corruptions[rng.randint(len(corruptions))]
                severity = int(severities[rng.randint(len(severities))])
                new_clip = _copy.copy(clip)
                new_clip.clip_id = f"{clip.clip_id}__{name}_s{severity}"
                new_clip.frames = corrupt_frames(
                    clip.frames, name, severity, seed=seed + clip_index * 17 + variant
                )
                sub, geo, _ = scorer.score_clip(new_clip.frames, geometric_columns)
                new_clip.sub_scores = sub
                new_clip.mrs = sub.mean(axis=1).astype(np.float32)
                if scaler.mean is not None:
                    geo = ((geo - scaler.mean) / scaler.std).astype(np.float32)
                new_clip.geometric = geo
                out.append(new_clip)
    finally:
        scorer.close()

    if out:
        reliability = np.concatenate([c.mrs for c in out])
        logger.info(
            "Corruption bank: %d clips, reliability mean %.3f std %.3f (clean std was ~0.05)",
            len(out), float(reliability.mean()), float(reliability.std()),
        )
    return out


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _evaluate(model, dataset, device, num_classes: int, blind: bool) -> Dict[str, object]:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)
    model.eval()
    trues, preds = [], []
    coral = model.cfg.train.loss == "coral"
    if coral:
        from models import coral_predict

    for batch in loader:
        sub_scores = batch["sub_scores"].to(device)
        mrs = batch["mrs"].to(device)
        if blind:
            # Tell the model every frame is perfect, though the pixels are not.
            sub_scores = torch.ones_like(sub_scores)
            mrs = torch.ones_like(mrs)
        logits = model(
            batch["frames"].to(device), batch["geometric"].to(device),
            mrs, batch["mask"].to(device), sub_scores,
        )["logits"]
        preds.append((coral_predict(logits) if coral else logits.argmax(1)).cpu().numpy())
        trues.append(batch["label"].numpy())

    return evaluate(np.concatenate(trues), np.concatenate(preds), num_classes)


def run_robustness(
    output_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    corruptions: Sequence[str],
    severities: Sequence[int],
    split: str = "Test",
    model_dir: Optional[Path] = None,
    recompute: bool = True,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = load_mrgvm_config(None, **checkpoint["config"])
    _, datasets, info = build_dataloaders(
        output_root, cfg.data, cfg.splits, cfg.vision_mamba.image_size
    )
    if split not in datasets:
        raise SystemExit(f"Split {split} not available; have {list(datasets)}")

    model = MRGVMModel(cfg, info["geometric_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    base = datasets[split]
    num_classes = cfg.data.num_classes
    recomputer = CorruptedClipScorer(model_dir) if recompute else None
    scaler = info["scaler"]
    geometric_columns = list(info["geometric_columns"])
    rows: List[Dict[str, object]] = []

    # severity 0 == the clean baseline
    clean = _evaluate(model, base, device, num_classes, blind=False)
    for condition in ("guided", "blind"):
        rows.append({
            "corruption": "none", "severity": 0, "condition": condition,
            "macro_f1": clean["macro_f1"], "accuracy": clean["accuracy"],
            "quadratic_weighted_kappa": clean["quadratic_weighted_kappa"],
        })
    logger.info("Clean baseline: macro-F1 %.3f", clean["macro_f1"])

    for name in corruptions:
        for severity in severities:
            corrupted_clips = []
            mesh_failure_rates = []
            for clip in base.clips:
                new_clip = copy.copy(clip)
                new_clip.frames = corrupt_frames(clip.frames, name, severity)
                if recomputer is not None:
                    sub, geo, failure_rate = recomputer.score_clip(
                        new_clip.frames, geometric_columns
                    )
                    new_clip.sub_scores = sub
                    new_clip.mrs = sub.mean(axis=1).astype(np.float32)
                    # Standardise with the TRAIN-split statistics, exactly as the
                    # clean data was, so the only difference is the pixels.
                    if scaler.mean is not None:
                        geo = ((geo - scaler.mean) / scaler.std).astype(np.float32)
                    new_clip.geometric = geo
                    mesh_failure_rates.append(failure_rate)
                corrupted_clips.append(new_clip)

            dataset = MRGVMDataset(
                corrupted_clips, base.max_frames, base.target, base.norm_mean, base.norm_std
            )
            for condition in ("guided", "blind"):
                metrics = _evaluate(
                    model, dataset, device, num_classes, blind=(condition == "blind")
                )
                rows.append({
                    "corruption": name, "severity": severity, "condition": condition,
                    "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                    "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
                    "mean_recomputed_mrs": float(
                        np.mean([c.mrs.mean() for c in corrupted_clips])
                    ) if recomputer else float("nan"),
                    "mesh_failure_rate": float(np.mean(mesh_failure_rates))
                    if mesh_failure_rates else 0.0,
                })
            guided = rows[-2]["macro_f1"]
            blind_score = rows[-1]["macro_f1"]
            logger.info(
                "%-14s sev %d  guided %.3f  blind %.3f  delta %+.3f  "
                "(MRS %.3f, mesh failed %.0f%%)",
                name, severity, guided, blind_score, guided - blind_score,
                rows[-1].get("mean_recomputed_mrs", float("nan")),
                100 * rows[-1].get("mesh_failure_rate", 0.0),
            )

    if recomputer is not None:
        recomputer.close()

    table = pd.DataFrame(rows)
    results_dir = output_root / "results_mrgvm"
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "robustness_results.csv", index=False)
    return table


def summarise(table: pd.DataFrame) -> Dict[str, object]:
    """Area-under-degradation-curve per condition, plus the guided-blind gap."""
    corrupted = table[table.severity > 0]
    summary: Dict[str, object] = {}
    for condition in ("guided", "blind"):
        subset = corrupted[corrupted.condition == condition]
        summary[f"{condition}_mean_macro_f1"] = float(subset["macro_f1"].mean())
    summary["guided_minus_blind"] = (
        summary["guided_mean_macro_f1"] - summary["blind_mean_macro_f1"]
    )
    per_corruption = {}
    for name, group in corrupted.groupby("corruption"):
        g = group[group.condition == "guided"]["macro_f1"].mean()
        b = group[group.condition == "blind"]["macro_f1"].mean()
        per_corruption[name] = {"guided": float(g), "blind": float(b), "delta": float(g - b)}
    summary["per_corruption"] = per_corruption
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--split", default="Test")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--no-recompute", dest="recompute", action="store_false",
                        help="Skip MRS recomputation (makes the experiment meaningless; "
                             "provided only as a negative control).")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(output_root / "results_mrgvm" / "robustness.log",
                                      mode="w", encoding="utf-8")],
    )
    checkpoint = args.checkpoint or (output_root / "checkpoints" / "mrgvm_v2.pt")
    if not Path(checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    table = run_robustness(
        output_root, Path(checkpoint), device, args.corruptions, args.severities,
        args.split, args.model_dir, args.recompute,
    )
    summary = summarise(table)

    logger.info("=" * 76)
    logger.info("ROBUSTNESS SUMMARY (mean macro-F1 across all corrupted conditions)")
    logger.info("  guided (MRS recomputed) : %.3f", summary["guided_mean_macro_f1"])
    logger.info("  blind  (MRS pinned to 1): %.3f", summary["blind_mean_macro_f1"])
    logger.info("  guided - blind          : %+.3f", summary["guided_minus_blind"])
    logger.info("-" * 76)
    for name, values in sorted(
        summary["per_corruption"].items(), key=lambda kv: -kv[1]["delta"]
    ):
        logger.info("  %-14s guided %.3f  blind %.3f  delta %+.3f",
                    name, values["guided"], values["blind"], values["delta"])

    (output_root / "results_mrgvm" / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
