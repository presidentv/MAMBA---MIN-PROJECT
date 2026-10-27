"""Score a single video file, with no dataset index and no cache.

Every other entry point into this model -- scripts/run_finetune.py,
src/evaluate.py, scripts/explain_finetuned.py -- starts from ``build_index``
and the Stage 1 cache, which together need a DAiSEE-shaped directory tree and
a labels CSV. That is the right shape for an experiment and the wrong shape
for a demo: a front end has one clip, no label, and nowhere to put a cache.

This module closes that gap without reimplementing anything. The pipeline is
exactly the one the reported numbers came from:

    read_frames -> FacePreprocessor.process_clip -> the item a
    FineTuneClipDataset would have returned -> ViT -> MRGViTMamba

The preprocessing is literally ``FacePreprocessor.process_clip``, invoked on a
stand-in record, so face detection, the landmark mesh, the MRS components and
the clip-level landmark aggregation cannot drift from the training path. The
item assembly mirrors ``FineTuneClipDataset.__getitem__``; ``_build_item``
below is the one place where this file duplicates logic, and it is kept
deliberately short so it can be read against its original.

Two things are enforced rather than assumed, because both fail silently:

  * The MRS calibration must be the fitted one. An uncalibrated fallback still
    produces plausible reliability scores from entirely different percentile
    bounds, so it is refused unless explicitly allowed.
  * Test-time augmentation matches ``evaluation.tta_scales`` from the
    checkpoint's own config, which is what run_finetune.py averaged over when
    it produced the published metrics.

Usage:

    predictor = EngagementPredictor.from_checkpoint(path_to_best_pt)
    result = predictor.predict("my_clip.mp4")
    print(result.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .augment import CenterCropView
from .class_merge import class_names as merged_class_names
from .dataset import load_calibration
from .dataset_index import ClipRecord
from .model import MRGViTMamba
from .mrs import COMPONENTS, component_matrix_from_arrays, mrs_from_arrays
from .preprocess import FacePreprocessor, unpack_crops
from .utils import resolve_path
from .vit_encoder import ViTFrameEncoder, preprocess_crops

# process_clip writes record.engagement into its output. A clip handed to a
# front end has no label, and -1 is carried through to the item builder, which
# drops it. It never reaches the model.
UNLABELLED = -1


@dataclass
class ClipPrediction:
    """One clip's result, in the terms the front end needs to display."""

    video: str
    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float]
    # The probability of the *detection* class, i.e. the one the ROC-AUC in the
    # paper ranks on. None when the checkpoint is not a binary detector.
    detection_score: float | None
    per_frame_reliability: list[float]
    per_frame_components: list[dict[str, float]]
    learned_mrs_weights: dict[str, float]
    pooling_weights: list[float] | None
    tta_views: list[dict] | None
    quality: dict = field(default_factory=dict)
    explanation: dict | None = None

    def summary(self) -> str:
        parts = [f"{self.video}: {self.predicted_class}"]
        parts.append("  " + "  ".join(f"{k} {v:.3f}"
                                      for k, v in self.probabilities.items()))
        r = np.asarray(self.per_frame_reliability)
        parts.append(f"  reliability: mean {r.mean():.3f}  min {r.min():.3f}  "
                     f"max {r.max():.3f}")
        q = self.quality
        parts.append(f"  face detected in {q.get('frames_with_face')}/"
                     f"{q.get('num_frames')} sampled frames, landmarks in "
                     f"{q.get('frames_with_landmarks')}")
        if q.get("warnings"):
            for w in q["warnings"]:
                parts.append(f"  ! {w}")
        return "\n".join(parts)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["explanation"] = self.explanation
        return d


class EngagementPredictor:
    """A loaded model plus the preprocessing it was trained with.

    Construction is expensive -- a ViT-B/16 and two MediaPipe graphs -- so a
    server should build one of these at startup and call ``predict`` per
    request, not build one per request.
    """

    def __init__(self, model, encoder, cfg, calibration, class_names,
                 device, tta_views, num_frames, checkpoint_path=None):
        self.model = model
        self.encoder = encoder
        self.cfg = cfg
        self.calibration = calibration
        self.class_names = tuple(class_names)
        self.device = device
        self.tta_views = list(tta_views)
        self.num_frames = int(num_frames)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self._pre = FacePreprocessor(cfg)
        self.learned_mrs_weights = (
            model.learnable_mrs.weights().detach().cpu().numpy()
            if model.learnable_mrs is not None
            else np.full(len(COMPONENTS), 1.0 / len(COMPONENTS)))

    # ------------------------------------------------------------- loading
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, use_tta=True,
                        allow_uncalibrated_mrs=False) -> "EngagementPredictor":
        """Rebuild the evaluated model from a fine-tuning checkpoint.

        The architecture, the backbone spec, the landmark standardiser and the
        class names all come from the checkpoint itself rather than from a
        config file supplied by the caller, so a mismatched config cannot
        quietly produce a different model than the one that was evaluated.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"no checkpoint at {checkpoint_path}. The fine-tuned detectors are "
                f"written outside the repository; see the run's "
                f"artifacts/finetune_report_<run>.json for its checkpoint_dir.")

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not ckpt.get("finetuned"):
            raise ValueError(
                f"{checkpoint_path} is not a fine-tuning checkpoint. Frozen-backbone "
                f"checkpoints store features from a backbone that no longer exists "
                f"and cannot be used to score raw video.")

        cfg = ckpt["config"]
        num_classes = int(cfg["dataset"]["num_classes"])
        class_names = tuple(ckpt.get("class_names")
                            or merged_class_names(cfg, num_classes))

        encoder = ViTFrameEncoder(model_name=cfg["vit"]["model_name"], pretrained=False,
                                  freeze=True, chunk_size=int(cfg["vit"]["batch_size"]))
        model = MRGViTMamba(vit_dim=int(ckpt["vit_dim"]),
                            landmark_dim=int(ckpt["landmark_dim"]),
                            cfg=cfg, vit_encoder=encoder).to(device)
        # strict: a missing key here means the landmark standardiser or the MRS
        # weights were not restored, which changes the prediction silently.
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        calibration = load_calibration(cfg)
        if not calibration.calibrated and not allow_uncalibrated_mrs:
            raise RuntimeError(
                f"the MRS calibration file named by this checkpoint's config "
                f"({cfg['mrs']['calibration_file']}) was not found, so the reliability "
                f"score would be computed from generic fallback bounds rather than the "
                f"fitted ones. Scores would not match the reported run. Pass "
                f"allow_uncalibrated_mrs=True only if you accept that.")

        tta_views = (CenterCropView.set_from_config(dict(cfg.get("evaluation", {})))
                     if use_tta else [])

        return cls(model, encoder, cfg, calibration, class_names, device,
                   tta_views, int(ckpt["num_frames"]), checkpoint_path)

    # ------------------------------------------------------------ the item
    def _build_item(self, stage1: dict, view=None) -> dict:
        """The tensors FineTuneClipDataset.__getitem__ would have returned.

        Kept in step with that method: crops are optionally passed through a
        deterministic view, then normalised with the backbone's own statistics,
        and the MRS components are derived from the same raw arrays.
        """
        crops = unpack_crops(stage1["crops_data"], stage1["crops_offsets"])
        if view is not None:
            crops = view(crops)
        frames = preprocess_crops(crops, self.encoder.spec)
        land = np.asarray(stage1["landmark_clip_features"], dtype=np.float32)

        raw_args = (stage1["blur_raw"], stage1["face_area_fraction"],
                    stage1["detector_confidence"], stage1["head_pose_deg"],
                    stage1["mrs_eye_visibility"], stage1["mrs_motion_consistency"])
        cfg_mrs = dict(self.cfg["mrs"])
        mrs = mrs_from_arrays(*raw_args, cfg_mrs, self.calibration)
        comps = component_matrix_from_arrays(*raw_args, cfg_mrs, self.calibration)

        return {
            "frames": frames,
            "mrs": torch.from_numpy(np.asarray(mrs, dtype=np.float32)),
            "mrs_components": torch.from_numpy(np.asarray(comps, dtype=np.float32)),
            "landmark_features": torch.from_numpy(land),
            # explain_clip reads item["label"] to report whether the prediction
            # was right. A clip from a front end has no ground truth, so the
            # sentinel goes in and _explanation_payload drops what comes back
            # out; nothing downstream of the model ever sees it.
            "label": torch.tensor(UNLABELLED, dtype=torch.long),
        }

    @torch.no_grad()
    def _forward(self, item: dict) -> np.ndarray:
        frames = item["frames"][None].to(self.device)
        mrs = item["mrs"][None].to(self.device)
        comps = item["mrs_components"][None].to(self.device)
        land = item["landmark_features"][None].to(self.device)
        logits = self.model(self.encoder(frames), mrs, land,
                            mrs_components=comps).float()
        return torch.softmax(logits, -1)[0].cpu().numpy()

    # ------------------------------------------------------------- scoring
    def preprocess(self, video_path) -> dict:
        """Run face detection, landmarks and MRS over one video file."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"no video at {video_path}")
        record = ClipRecord(clip_id=video_path.name, stem=video_path.stem,
                            split="predict", subject_id="unknown",
                            path=str(video_path), labels={"Engagement": UNLABELLED})
        return self._pre.process_clip(record)

    def predict(self, video_path, explain: bool = False,
                ig_steps: int = 64) -> ClipPrediction:
        """Score one video.

        Probabilities are averaged over the checkpoint's TTA views, which is
        what produced the published metrics. ``explain`` additionally runs
        integrated gradients and Grad-CAM on the untransformed view; it needs
        gradients and roughly doubles the time.
        """
        stage1 = self.preprocess(video_path)

        base_item = self._build_item(stage1)
        probs = self._forward(base_item)
        extra = [v for v in self.tta_views if float(v.scale) < 1.0]
        if extra:
            stacked = [probs]
            for view in extra:
                stacked.append(self._forward(self._build_item(stage1, view)))
            probs = np.mean(np.stack(stacked), axis=0)

        pred = int(np.argmax(probs))

        explanation = None
        if explain:
            from .explain import explain_clip

            ex = explain_clip(self.model, base_item, self.device, ig_steps=ig_steps)
            explanation = self._explanation_payload(ex)
            # The explanation is computed on the single untransformed view, so a
            # clip near the decision boundary can be explained as one class and
            # scored as the other. Say so rather than showing a contradiction.
            if int(ex["pred"]) != pred:
                explanation["disagrees_with_tta"] = (
                    f"explained as {self.class_names[int(ex['pred'])]} on the "
                    f"untransformed view but scored {self.class_names[pred]} after "
                    f"averaging over {len(extra) + 1} views; this clip sits near the "
                    f"decision boundary")

        mrs = base_item["mrs"].numpy()
        comps = base_item["mrs_components"].numpy()
        pooling = self._pooling_weights(mrs)

        return ClipPrediction(
            video=str(video_path),
            predicted_class=self.class_names[pred],
            predicted_index=pred,
            probabilities={n: float(p) for n, p in zip(self.class_names, probs)},
            detection_score=self._detection_score(probs),
            per_frame_reliability=[float(v) for v in mrs],
            per_frame_components=[
                {c: float(v) for c, v in zip(COMPONENTS, row)} for row in comps],
            learned_mrs_weights={c: float(v) for c, v
                                 in zip(COMPONENTS, self.learned_mrs_weights)},
            pooling_weights=pooling,
            tta_views=([{"view": "center_crop", "scale": 1.0}]
                       + [v.describe() for v in extra]) if extra else None,
            quality=self._quality(stage1),
            explanation=explanation,
        )

    # ------------------------------------------------------------- helpers
    def _pooling_weights(self, mrs: np.ndarray) -> list[float] | None:
        pooling = self.model.temporal.pooling
        if pooling == "mrs_weighted":
            return [float(v) for v in mrs / (mrs.sum() + 1e-6)]
        if pooling == "mean":
            return [1.0 / len(mrs)] * len(mrs)
        return None            # "last"/"attention" have no fixed per-frame weight

    def _detection_score(self, probs: np.ndarray) -> float | None:
        """P(not engaged) for a binary detector, which is what ROC-AUC ranks on."""
        if len(self.class_names) != 2:
            return None
        for i, name in enumerate(self.class_names):
            if "not" in name.lower():
                return float(probs[i])
        return float(probs[0])

    def _quality(self, stage1: dict) -> dict:
        """Observation quality, so a bad recording is visible rather than scored
        as if it were fine."""
        face = np.asarray(stage1["face_found"], dtype=bool)
        land = np.asarray(stage1["landmarks_found"], dtype=bool)
        n = int(face.shape[0])
        sources, counts = np.unique(np.asarray(stage1["bbox_source"]), return_counts=True)
        warnings = []
        if face.sum() < n:
            warnings.append(
                f"the face detector missed {n - int(face.sum())} of {n} sampled frames; "
                f"a fallback box was used there")
        if land.sum() < n:
            warnings.append(
                f"no landmark mesh on {n - int(land.sum())} of {n} sampled frames, so "
                f"the landmark branch saw zeros for those")
        res = np.asarray(stage1["source_resolution"])
        return {
            "num_frames": n,
            "frames_with_face": int(face.sum()),
            "frames_with_landmarks": int(land.sum()),
            "bbox_sources": {str(s): int(c) for s, c in zip(sources, counts)},
            "fps": float(stage1["fps"]),
            "source_frame_count": int(stage1["source_frame_count"]),
            "resolution": [int(res[0]), int(res[1])],
            "warnings": warnings,
        }

    def _explanation_payload(self, ex: dict) -> dict:
        """The parts of explain_clip a front end can draw, without the images."""
        out = {
            "logits": [float(v) for v in ex["logits"]],
            "pooling": ex["pooling"],
            "logit_decomposition": {k: float(v) for k, v
                                    in ex["logit_decomposition"].items()},
        }
        la = ex.get("landmark_attribution")
        if la is not None:
            order = np.argsort(-np.abs(la["attribution"]))[:10]
            out["landmark_groups"] = {g: float(v) for g, v
                                      in la["group_attribution"].items()}
            out["top_features"] = [
                {"feature": la["names"][i],
                 "contribution": float(la["attribution"][i]),
                 "value": float(la["value"][i]),
                 "train_mean": float(la["train_mean"][i])}
                for i in order]
        return out

    def describe(self) -> dict:
        return {
            "checkpoint": self.checkpoint_path,
            "classes": list(self.class_names),
            "num_frames": self.num_frames,
            "device": str(self.device),
            "backbone": self.cfg["vit"]["model_name"],
            "pooling": self.model.temporal.pooling,
            "tta_views": [v.describe() for v in self.tta_views] or None,
            "mrs_calibrated": bool(self.calibration.calibrated),
            "learned_mrs_weights": {c: float(v) for c, v
                                    in zip(COMPONENTS, self.learned_mrs_weights)},
        }

    def close(self) -> None:
        self._pre.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    """Score one or more videos from the command line.

    This doubles as the equivalence check for the module: ``--expect`` compares
    the probabilities against a row of a per-clip predictions file, which is
    how a drift between this path and the evaluated path gets caught.
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+", help="video files to score")
    ap.add_argument("--checkpoint", required=True, help="path to a fine-tuned best.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tta", action="store_true",
                    help="single untransformed view instead of the evaluated TTA set")
    ap.add_argument("--explain", action="store_true",
                    help="also run integrated gradients over the landmark features")
    ap.add_argument("--json", default=None, help="write the full results here")
    ap.add_argument("--expect", default=None, metavar="PREDICTIONS_JSON",
                    help="an artifacts/per_clip_predictions_*.json to check against; "
                         "each video is matched by file name")
    args = ap.parse_args(argv)

    predictor = EngagementPredictor.from_checkpoint(
        args.checkpoint, device=args.device, use_tta=not args.no_tta)
    print(json.dumps(predictor.describe(), indent=2))

    expected = {}
    if args.expect:
        rows = json.loads(Path(resolve_path(args.expect)).read_text(encoding="utf-8"))
        expected = {r["clip_id"]: r for r in rows["predictions"]}

    results, failures = [], 0
    try:
        for video in args.videos:
            result = predictor.predict(video, explain=args.explain)
            print()
            print(result.summary())
            results.append(result.as_dict())

            row = expected.get(Path(video).name)
            if row is not None:
                theirs = np.asarray(row["probs"], dtype=np.float64)
                ours = np.asarray(list(result.probabilities.values()), dtype=np.float64)
                gap = float(np.max(np.abs(theirs - ours)))
                ok = gap < 1e-4 and int(row["pred"]) == result.predicted_index
                print(f"  vs evaluation: max |delta p| = {gap:.2e}  "
                      f"{'MATCH' if ok else 'MISMATCH'}")
                if not ok:
                    failures += 1
                    print(f"    evaluated: pred {row['pred']} probs "
                          f"{[round(p, 6) for p in row['probs']]}")
                    print(f"    this path: pred {result.predicted_index} probs "
                          f"{[round(p, 6) for p in ours]}")
    finally:
        predictor.close()

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")

    if expected and failures:
        print(f"\n{failures} clip(s) did not reproduce the evaluated probabilities.")
        return 1
    if expected:
        print("\nall checked clips reproduce the evaluated probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
