"""Step 7b -- STUB. Explainability hooks for the fusion model.

NOT IMPLEMENTED YET. Signatures and the intended method for each technique are
fixed here so the XAI stage slots into the existing model without redesign.

The three planned techniques, and what each is for:

1. SHAP over modality-level features
   Which *features* drive the prediction. Run KernelSHAP (or GradientSHAP) over
   the clip-level mean-pooled feature vector -- the same representation the
   logistic-regression baseline consumes -- so SHAP values are directly
   comparable between the interpretable baseline and the transformer. Grouping
   the 21 gaze features and the 8 affect probabilities into named blocks
   (fixation / dispersion / blink / off-screen / head-stability / affect) gives
   modality-level attributions rather than an unreadable per-dimension bar chart.

2. Attention rollout
   Which *timesteps* drive the prediction. The cross-attention block in
   ``models.CrossAttentionFusion`` already returns its averaged attention
   weights under the keys ``gaze_to_affect`` and ``affect_to_gaze``, and
   ``TemporalEncoder`` can expose its self-attention with a forward hook.
   Rollout multiplies the per-layer attention matrices (adding an identity term
   for the residual path) to get an input-to-output attribution over time.

3. Deletion test -- the faithfulness check
   The one that keeps the other two honest. Ablate the top-k features (or
   timesteps) that SHAP/rollout claim matter, re-run the model, and measure how
   fast macro-F1 falls. A steep drop means the explanation identified genuinely
   load-bearing inputs; a flat curve means the explanation is decorative. Always
   report against a random-ablation control curve.

Nothing below is called by train.py yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Feature blocks used to aggregate per-dimension attributions into something
# a reader can interpret. Names must match features.GAZE_FEATURE_COLUMNS.
GAZE_FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "fixation": ("gaze_velocity", "is_fixation"),
    "dispersion": ("gaze_dispersion", "gaze_std_yaw", "gaze_std_pitch"),
    "blink": ("ear", "ear_normalised", "is_blink", "blink_rate_window"),
    "off_screen": ("off_screen",),
    "head_stability": ("head_std_yaw", "head_std_pitch", "head_std_roll"),
    "head_pose": ("yaw", "pitch", "roll"),
    "gaze_direction": ("gaze_yaw", "gaze_pitch", "iris_rel_x", "iris_rel_y"),
    "frame_quality": ("mrs",),
}


def compute_shap_values(
    model,
    background,
    samples,
    feature_names: Sequence[str],
    n_samples: int = 200,
):
    """Return per-feature SHAP attributions for a batch of clips.

    TODO: implement with shap.KernelExplainer over a wrapper that mean-pools the
    sequence, or shap.GradientExplainer directly on the torch model. Background
    must come from the TRAIN split only.
    """
    raise NotImplementedError("compute_shap_values is a stub; see module docstring.")


def aggregate_shap_by_group(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    groups: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Dict[str, float]:
    """Collapse per-feature SHAP values into named modality-level blocks.

    TODO: implement -- sum absolute SHAP values within each group in
    GAZE_FEATURE_GROUPS, plus one "affect" block over the 8 emotion columns,
    then normalise so the blocks sum to 1.
    """
    raise NotImplementedError("aggregate_shap_by_group is a stub.")


def attention_rollout(
    attention_weights: Sequence[np.ndarray],
    add_residual: bool = True,
) -> np.ndarray:
    """Multiply per-layer attention matrices into one input->output attribution.

    TODO: implement Abnar & Zuidema (2020): optionally add 0.5*(A + I) to model
    the residual stream, row-normalise, then take the matrix product across
    layers. Returns a (T,) importance vector over timesteps.
    """
    raise NotImplementedError("attention_rollout is a stub.")


def deletion_test(
    model,
    dataloader,
    importance: np.ndarray,
    fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    mode: str = "features",
    random_control: bool = True,
) -> Dict[str, List[float]]:
    """Ablate the most-important inputs and measure how fast macro-F1 falls.

    TODO: implement. For each fraction, zero out (post-standardisation, so zero
    == the training mean) the top-k inputs by ``importance``, re-run evaluation,
    and record macro-F1. When ``random_control`` is set, also produce a curve
    with randomly chosen inputs -- the gap between the two curves IS the
    faithfulness result. Returns {"fractions": [...], "guided": [...],
    "random": [...]}.
    """
    raise NotImplementedError("deletion_test is a stub.")


def explain_model(
    checkpoint_path: Path,
    output_root: Path,
    output_json: Optional[Path] = None,
) -> Dict[str, object]:
    """End-to-end XAI report: SHAP + rollout + deletion, written to JSON.

    TODO: implement once the three functions above exist. Intended to be
    callable as ``python src/explain.py --checkpoint outputs/checkpoints/
    cross_attention_fusion.pt --output-root outputs``.
    """
    raise NotImplementedError("explain_model is a stub; see module docstring.")
