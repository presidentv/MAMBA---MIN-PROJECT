"""Configuration for the engagement-classifier stack.

Same precedence rule as the preprocessing pipeline: CLI > JSON file > default.
Nothing downstream hardcodes a path or a hyperparameter; everything routes
through these dataclasses.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Step 1 -- gaze features
# --------------------------------------------------------------------------- #
@dataclass
class GazeFeatureConfig:
    iris_gain_deg: float = 40.0
    # Degrees of eye rotation corresponding to the iris traversing its full
    # socket width. Converts the scale-free iris offset from Phase 2 into an
    # angle that can be added to head yaw/pitch to form a gaze direction.

    saccade_velocity_threshold: float = 30.0
    # I-VT threshold in deg/s separating fixation from saccade.
    #
    # CAVEAT, stated loudly because it matters for interpretation: Phase 1
    # samples at 5 fps (200 ms between frames) while a real saccade lasts
    # 30-80 ms. At this rate we cannot observe saccades directly -- what this
    # flag actually measures is the rate of GAZE SHIFTS BETWEEN SAMPLES. It is a
    # legitimate engagement signal, but it must not be reported as a saccade
    # rate. Raising --sample-fps in Phase 1 narrows this gap.

    rolling_window: int = 5
    # Frames per rolling window for dispersion / stability (5 @ 5 fps = 1 s).

    blink_ear_ratio: float = 0.75
    # A frame is a blink candidate when its eye-aspect-ratio falls below this
    # fraction of the clip's own median EAR. A per-clip baseline is used rather
    # than a global constant because EAR varies strongly with face shape,
    # eyewear and camera angle.
    blink_min_separation: int = 1
    # Minimum frames between two counted blinks, to avoid double-counting one
    # blink spread over consecutive samples.

    off_screen_yaw_deg: float = 25.0
    off_screen_pitch_deg: float = 20.0
    # Gaze beyond either bound is treated as "not looking at the screen".

    min_frames: int = 4
    # Clips with fewer landmarked frames than this are skipped and reported.


# --------------------------------------------------------------------------- #
# Step 2 -- affect features
# --------------------------------------------------------------------------- #
@dataclass
class AffectFeatureConfig:
    model_name: str = "enet_b0_8_best_afew"
    # HSEmotion EfficientNet-B0 trained on AffectNet/AFEW. Exposes both an
    # 8-class emotion distribution and the 1280-d penultimate embedding.
    device: str = "cpu"
    batch_size: int = 32
    save_embeddings: bool = True
    # Store the 1280-d embedding alongside the 8 probabilities. Both are written;
    # which one the model consumes is decided by DataConfig.affect_feature_set.


# --------------------------------------------------------------------------- #
# Step 3 -- dataset assembly
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    max_sequence_length: int = 50
    # Clips are ~10 s at 5 fps, so 50 covers a full clip. Longer sequences are
    # truncated, shorter ones padded with an attention mask.
    affect_feature_set: str = "probs"
    # 'probs' (8-d, low variance, appropriate for a small sample) or
    # 'embedding' (1280-d) or 'both'.
    target: str = "Engagement"
    auxiliary_targets: Tuple[str, ...] = ("Boredom", "Confusion", "Frustration")
    num_classes: int = 4
    standardize: bool = True
    # Z-score features using TRAIN-split statistics only; the fitted scaler is
    # saved so validation/test never leak into it.
    batch_size: int = 8
    num_workers: int = 0
    # 0 on Windows: worker processes would re-import and re-pickle everything for
    # a dataset that fits comfortably in RAM.


# --------------------------------------------------------------------------- #
# Step 4 -- fusion model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.3
    max_len: int = 512
    pooling: str = "masked_mean"
    loss: str = "coral"
    # 'coral' = CORAL ordinal loss (K-1 cumulative binary heads with a shared
    # backbone weight and independent biases; Cao et al. 2020). Chosen over a
    # cumulative-link/ordinal-logit model because CORAL guarantees monotone
    # cumulative probabilities by construction, so it cannot produce the
    # inverted thresholds that make ordinal-logit fits unstable on small data.
    # 'ce' = plain categorical cross-entropy, kept for ablation.
    class_weighting: bool = True


# --------------------------------------------------------------------------- #
# Step 6 -- training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    monitor: str = "val_macro_f1"
    seed: int = 42
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    gaze: GazeFeatureConfig = field(default_factory=GazeFeatureConfig)
    affect: AffectFeatureConfig = field(default_factory=AffectFeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    splits: Tuple[str, ...] = ("Train", "Validation", "Test")
    models_to_run: Tuple[str, ...] = (
        "majority",
        "logreg_meanpool",
        "gaze_only",
        "affect_only",
        "late_fusion",
        "cross_attention_fusion",
    )


# --------------------------------------------------------------------------- #
# helpers (mirrors pipeline/engagement_pipeline/config.py)
# --------------------------------------------------------------------------- #
def _coerce(value: Any, target_type: Any) -> Any:
    type_str = str(target_type)
    if "Tuple" in type_str or "tuple" in type_str:
        return tuple(value) if isinstance(value, list) else value
    if target_type is float and isinstance(value, int):
        return float(value)
    return value


def _apply_overrides(instance: Any, overrides: Dict[str, Any]) -> None:
    valid = {f.name: f for f in fields(instance)}
    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        if key not in valid:
            raise KeyError(
                "Unknown config key '{}' for {}. Valid: {}".format(
                    key, type(instance).__name__, sorted(valid)
                )
            )
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(instance, key, _coerce(value, valid[key].type))


def load_experiment_config(
    config_path: Optional[Path] = None, **overrides: Any
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if config_path is not None:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        _apply_overrides(cfg, raw)
    clean = {k: v for k, v in overrides.items() if v is not None}
    if clean:
        _apply_overrides(cfg, clean)
    return cfg


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def dump_config(cfg: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")
