"""Step 2 -- frozen facial-expression features over the Phase 1 reliable frames.

SUBSTITUTION NOTE
-----------------
The project plan called for OpenFace 2.0 action-unit intensities as the affect
channel. OpenFace is not a pip package -- it needs a compiled binary plus model
downloads -- so it is unavailable here. The agreed fallback, HSEmotion, is used
instead: a frozen EfficientNet-B0 trained on AffectNet, exposing

  * an 8-class emotion distribution
    (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise), and
  * the 1280-d penultimate embedding.

Both are written per frame; ``DataConfig.affect_feature_set`` decides which the
model consumes. On a sample this small the 8 probabilities are the sane default
-- 1280 dimensions over ~100 clips would be pure memorisation.

Two environment fixes are applied and explained inline: PyTorch >= 2.6 refuses
the pickled checkpoint by default, and the checkpoint predates the installed
timm's module layout.

The frames are already 224x224, ImageNet-normalised, eye-line aligned crops from
Phase 1 -- exactly HSEmotion's expected input -- so no re-cropping happens here.

Run standalone:
    python src/affect.py --output-root outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AffectFeatureConfig, load_experiment_config  # noqa: E402

logger = logging.getLogger("affect")

EMOTION_CLASSES: Tuple[str, ...] = (
    "Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
)
AFFECT_PROB_COLUMNS: Tuple[str, ...] = tuple(f"emo_{name.lower()}" for name in EMOTION_CLASSES)
EMBEDDING_DIM = 1280


@contextmanager
def _allow_full_pickle_load():
    """Temporarily restore ``torch.load(weights_only=False)``.

    PyTorch 2.6 flipped ``weights_only`` to True by default. The HSEmotion
    checkpoint is a whole pickled ``nn.Module``, not a state dict, so it cannot
    load under the safe reader. The file is fetched by the ``hsemotion`` package
    itself from its own upstream repository, which is the same trust boundary as
    the pip install, so full unpickling is scoped to this one call and restored
    immediately afterwards.
    """
    import torch

    original = torch.load

    def permissive(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = permissive
    try:
        yield
    finally:
        torch.load = original


def load_recognizer(cfg: AffectFeatureConfig):
    """Construct a frozen HSEmotion recogniser, with a clear error if it cannot.

    NOTE ON PINNING: the published checkpoints were pickled against timm < 1.0.
    Under timm >= 1.0 the unpickled blocks lack attributes the newer ``forward``
    expects (``DepthwiseSeparableConv.conv_s2d``), which fails only at inference
    time, not at load time. ``timm==0.9.16`` is therefore pinned in
    requirements.txt; this check surfaces the problem early and loudly.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("timm is required for the affect branch: pip install timm==0.9.16") from exc

    if tuple(int(p) for p in timm.__version__.split(".")[:2]) >= (1, 0):
        logger.warning(
            "timm %s is newer than the HSEmotion checkpoints expect; if inference "
            "raises AttributeError on conv_s2d, pin timm==0.9.16",
            timm.__version__,
        )

    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer
    except ImportError as exc:
        raise SystemExit(
            "hsemotion is not installed. pip install hsemotion  (or set a different "
            "affect backbone in the config)."
        ) from exc

    with _allow_full_pickle_load():
        recognizer = HSEmotionRecognizer(model_name=cfg.model_name, device=cfg.device)
    logger.info("Loaded frozen affect backbone %s on %s", cfg.model_name, cfg.device)
    return recognizer


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def extract_clip_affect(
    frame_paths: List[Path], recognizer, batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities (N, 8), embeddings (N, 1280))`` for one clip.

    The backbone is frozen and used purely as a feature extractor; HSEmotion
    swaps the classifier for Identity at load time and keeps the head weights as
    numpy, so ``extract_multi_features`` yields the embedding and ``get_probab``
    turns it into logits.
    """
    embeddings: List[np.ndarray] = []
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("Unreadable frame skipped: %s", path)
                continue
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not images:
            continue
        embeddings.append(np.asarray(recognizer.extract_multi_features(images)))

    if not embeddings:
        return np.zeros((0, len(EMOTION_CLASSES))), np.zeros((0, EMBEDDING_DIM))

    stacked = np.vstack(embeddings)
    probabilities = _softmax(np.asarray(recognizer.get_probab(stacked)))
    return probabilities, stacked


def build_affect_features(output_root: Path, cfg: AffectFeatureConfig, splits) -> pd.DataFrame:
    """Extract affect features for every clip, aligned to Phase 1 retained frames.

    Alignment to the gaze branch is by ``sample_index``: both branches are keyed
    off the same Phase 1 retained-frame list, and the dataset class performs an
    explicit inner join on it rather than trusting positional order.
    """
    output_root = Path(output_root)
    phase1_root = output_root / "phase1_reliable_frames"
    phase2_root = output_root / "phase2_landmarks"
    if not phase1_root.is_dir():
        raise SystemExit(f"Phase 1 output missing at {phase1_root}")

    recognizer = load_recognizer(cfg)
    affect_root = output_root / "features" / "affect"
    clip_rows: List[Dict[str, object]] = []
    total_frames = 0

    for split in splits:
        split_dir = phase1_root / split
        if not split_dir.is_dir():
            continue
        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for clip_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                clip_id, subject_id = clip_dir.name, subject_dir.name

                # Drive the frame list from the Phase 2 table when it exists, so
                # the two modalities cover exactly the same frames. Fall back to
                # the Phase 1 scores table otherwise.
                landmark_file = phase2_root / split / subject_id / clip_id / "landmarks.parquet"
                if landmark_file.is_file():
                    index = pd.read_parquet(
                        landmark_file, columns=["sample_index", "frame_index", "timestamp"]
                    ).sort_values("sample_index")
                else:
                    scores_file = clip_dir / "mrs_scores.parquet"
                    if not scores_file.is_file():
                        logger.warning("No frame index for %s/%s; skipping", split, clip_id)
                        continue
                    scores = pd.read_parquet(scores_file)
                    index = scores[scores["retained"].astype(bool)][
                        ["sample_index", "frame_index", "timestamp"]
                    ].sort_values("sample_index")

                frames_dir = clip_dir / "frames"
                paths: List[Path] = []
                keep_rows: List[dict] = []
                for record in index.to_dict(orient="records"):
                    stem = f"s{int(record['sample_index']):04d}_f{int(record['frame_index']):06d}"
                    matches = sorted(frames_dir.glob(stem + ".*"))
                    if not matches:
                        continue
                    paths.append(matches[0])
                    keep_rows.append(record)

                if not paths:
                    logger.warning("No frame images for %s/%s", split, clip_id)
                    continue

                probabilities, embeddings = extract_clip_affect(paths, recognizer, cfg.batch_size)
                if probabilities.shape[0] != len(keep_rows):
                    logger.warning(
                        "%s: %d features for %d frames; truncating to the shorter",
                        clip_id, probabilities.shape[0], len(keep_rows),
                    )
                    n = min(probabilities.shape[0], len(keep_rows))
                    probabilities, embeddings, keep_rows = probabilities[:n], embeddings[:n], keep_rows[:n]

                frame_df = pd.DataFrame(keep_rows)
                frame_df.insert(0, "ClipID", clip_id)
                frame_df.insert(1, "SubjectID", subject_id)
                frame_df.insert(2, "split", split)
                for i, column in enumerate(AFFECT_PROB_COLUMNS):
                    frame_df[column] = probabilities[:, i]
                frame_df["emo_argmax"] = [EMOTION_CLASSES[i] for i in probabilities.argmax(axis=1)]
                if cfg.save_embeddings:
                    for i in range(embeddings.shape[1]):
                        frame_df[f"emb_{i:04d}"] = embeddings[:, i]

                out_dir = affect_root / split / subject_id / clip_id
                out_dir.mkdir(parents=True, exist_ok=True)
                frame_df.to_parquet(out_dir / "affect.parquet", index=False)
                total_frames += len(frame_df)

                row: Dict[str, object] = {
                    "ClipID": clip_id,
                    "SubjectID": subject_id,
                    "split": split,
                    "n_frames": len(frame_df),
                    "affect_file": str((out_dir / "affect.parquet").relative_to(output_root)),
                    "dominant_emotion": frame_df["emo_argmax"].mode().iloc[0],
                }
                for i, column in enumerate(AFFECT_PROB_COLUMNS):
                    row[f"{column}_mean"] = float(probabilities[:, i].mean())
                    row[f"{column}_std"] = float(probabilities[:, i].std())
                clip_rows.append(row)
                logger.info("%s/%s/%s: %d frames -> affect", split, subject_id, clip_id, len(frame_df))

    clip_df = pd.DataFrame(clip_rows)
    target = output_root / "features" / "affect_clip_features.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    clip_df.to_csv(target, index=False)

    manifest = {
        "backbone": cfg.model_name,
        "substitution": "HSEmotion EfficientNet-B0 in place of OpenFace 2.0 action units",
        "emotion_classes": list(EMOTION_CLASSES),
        "prob_columns": list(AFFECT_PROB_COLUMNS),
        "embedding_dim": EMBEDDING_DIM if cfg.save_embeddings else 0,
        "n_clips": len(clip_df),
        "n_frames": total_frames,
    }
    (output_root / "features" / "affect_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("Affect features: %d clips, %d frames -> %s", len(clip_df), total_frames, target)
    return clip_df


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-embeddings", dest="save_embeddings", action="store_false", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    experiment = load_experiment_config(args.config)
    for key in ("model_name", "device", "batch_size", "save_embeddings"):
        value = getattr(args, key)
        if value is not None:
            setattr(experiment.affect, key, value)

    build_affect_features(args.output_root, experiment.affect, experiment.splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
