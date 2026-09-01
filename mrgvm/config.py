"""Configuration for MRG-VM Phases 3-7.

Same precedence rule as the rest of the project: CLI > JSON file > dataclass
default. Every ablation switch in Phase 6 is a field here, so an ablation is a
config override rather than a code change.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class VisionMambaConfig:
    """Phase 3 -- MRG-VM backbone."""

    image_size: int = 112
    # Phase 1 stores 224x224 crops; they are downscaled on load. 112 with
    # patch 16 gives 49 spatial tokens, which keeps the pure-PyTorch scan
    # (a Python loop over the sequence axis) tractable on CPU.
    patch_size: int = 16
    d_model: int = 128
    spatial_depth: int = 2
    temporal_depth: int = 2
    d_state: int = 16
    dropout: float = 0.1
    max_frames: int = 64

    # --- the "motion reliability guided" mechanisms (Phase 6 ablates these) ---
    guide_delta: bool = True
    # Scale the SSM timestep dt by the frame's MRS, so unreliable frames move
    # the hidden state less.
    guide_pooling: bool = True
    # MRS-weighted temporal mean instead of a flat mean.
    min_delta_scale: float = 0.25
    # dt multiplier floor: the worst surviving frame is damped, not silenced.
    delta_map: str = "linear"
    # 'linear'         = floor + (1-floor)*MRS. Measured to be too weak: on this
    #                    data MRS is 0.93 +/- 0.049, so the multiplier is
    #                    near-constant and dt_proj's learned bias absorbs it.
    # 'clip_normalised'= standardise MRS within each clip first, so dt responds
    #                    to RELATIVE reliability. Try this first at full scale.


@dataclass
class ReliabilityConfig:
    """The v2 novelty: learned, multi-factor reliability conditioning."""

    learn_weights: bool = False
    # Replace the fixed equal-weight mean over the five MRS sub-scores with
    # learned softmax weights. Initialised uniform, so it starts identical to v1.
    adaptive_conditioning: bool = False
    # Feed the five-vector (not the scalar) into a learnable conditioner that
    # modulates the temporal stream. Supersedes vision_mamba.guide_delta.
    hidden_dim: int = 32
    min_dt_scale: float = 0.1
    dropout: float = 0.0
    # The three conditioning mechanisms, individually ablatable:
    use_film: bool = True      # feature-wise affine, conditioned on failure mode
    use_gate: bool = True      # blend toward a learned null token
    use_dt: bool = True        # scale the SSM timestep

    loss_weighting: bool = False
    # Weight each clip's loss by its mean reliability.
    loss_weight_strength: float = 1.0
    loss_weight_floor: float = 0.25


@dataclass
class FusionConfig:
    """Phase 4 -- feature fusion."""

    kind: str = "adaptive"          # 'adaptive' | 'concat' | 'cross_attention'
    num_heads: int = 4              # cross_attention only
    hidden_dim: int = 128
    dropout: float = 0.2
    use_mamba: bool = True          # Phase 6 ablation: drop the Vision Mamba stream
    use_geometric: bool = True      # Phase 6 ablation: drop the landmark stream
    geometric_pooling: str = "stats"
    # 'stats' = mean/std/min/max over frames (4x width, captures dynamics);
    # 'mean'  = plain mean.


@dataclass
class ClassifierConfig:
    """Phase 5 -- lightweight MLP head."""

    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.3
    num_classes: int = 4


@dataclass
class MRGVMDataConfig:
    max_frames: int = 50
    target: str = "Engagement"
    num_classes: int = 4
    batch_size: int = 4
    num_workers: int = 0
    standardize: bool = True
    use_mrs_gate: bool = True
    # Phase 6 ablation for the MRS itself: when False, every retained frame is
    # given MRS = 1.0, which disables BOTH guidance mechanisms at once and
    # isolates the contribution of the score.
    include_gaze_features: bool = True
    # Fold src/features.py gaze descriptors into the geometric stream.

    corruption_augment: int = 0
    # Corrupted variants generated per TRAINING clip (0 = off). Each variant gets
    # a random corruption at a random severity, with the MRS components AND the
    # geometric stream recomputed from the corrupted pixels.
    #
    # Why this exists: on clean DAiSEE, reliability is 0.93 +/- 0.049 for every
    # training frame. A conditioner cannot learn to use a signal that never
    # varies, which is why the v2 weights stayed at ~0.20 and why guided and
    # blind scored identically under corruption. Augmentation makes reliability
    # vary during training, so the mechanism has something to fit.
    #
    # Applied to the training split ONLY -- never to validation or test.
    corruption_severities: Tuple[int, ...] = (1, 2, 3, 4, 5)
    corruption_types: Tuple[str, ...] = (
        "gaussian_blur", "motion_blur", "occlusion", "darkness", "jpeg", "sensor_noise",
    )
    corruption_seed: int = 1234
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class MRGVMTrainConfig:
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    label_smoothing: float = 0.05
    class_weighting: bool = True
    loss: str = "ce"
    # 'ce'    = categorical cross-entropy (v1 default).
    # 'coral' = CORAL ordinal loss. Engagement is ordered 0-3, so treating the
    #           classes as unordered discards structure; CORAL's K-1 cumulative
    #           heads share one weight vector and differ only in bias, which
    #           makes the cumulative probabilities monotone by construction.
    seed: int = 42
    deterministic: bool = True


@dataclass
class MRGVMConfig:
    vision_mamba: VisionMambaConfig = field(default_factory=VisionMambaConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    data: MRGVMDataConfig = field(default_factory=MRGVMDataConfig)
    train: MRGVMTrainConfig = field(default_factory=MRGVMTrainConfig)
    splits: Tuple[str, ...] = ("Train", "Validation", "Test")


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


def load_mrgvm_config(config_path: Optional[Path] = None, **overrides: Any) -> MRGVMConfig:
    cfg = MRGVMConfig()
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
