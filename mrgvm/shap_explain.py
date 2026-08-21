"""Phase 7 -- Explainable AI with SHAP.

Estimates how much each behavioural feature contributes to the engagement
prediction, and renders the result.

WHAT IS EXPLAINED, AND WHY THAT CHOICE
--------------------------------------
SHAP is applied over the **named landmark/gaze features**, not over the raw
Vision Mamba embedding. A Shapley value for "mamba dimension 87" is not an
explanation of anything a reader can act on, whereas "blink rate" and "head yaw
stability" are exactly the behavioural quantities the project is about. The
Mamba stream is still represented: it enters as a small number of summary
components so its overall contribution can be compared against the
interpretable features on the same axis.

Two views are produced:

  * **per-feature** attributions, aggregated across the test split;
  * **per-group** attributions, folding features into the four behavioural
    categories the PDF names for Phase 3 -- eye gaze, blink patterns, head
    movement, facial dynamics -- plus the appearance stream.

The explainer wraps a scikit-learn surrogate fitted to the trained model's own
predictions on clip-level features. This is a deliberate, documented choice:
KernelSHAP against the full torch model would need thousands of forward passes
through the Vision Mamba scan (minutes per clip on CPU). The surrogate's fidelity
to the real model is measured and reported as ``surrogate_fidelity``; a low value
means the explanation should not be trusted, and that number is printed rather
than buried.

Usage:
    python -m mrgvm.shap_explain --output-root outputs \
        --checkpoint outputs/checkpoints/mrgvm.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .config import load_mrgvm_config  # noqa: E402
from .data import build_dataloaders  # noqa: E402
from .model import MRGVMModel, pool_geometric  # noqa: E402

logger = logging.getLogger("mrgvm.shap")

# Behavioural grouping, matching the four categories the PDF names for Phase 3.
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "eye_gaze": (
        "gaze_yaw", "gaze_pitch", "gaze_velocity", "is_fixation", "gaze_dispersion",
        "gaze_std_yaw", "gaze_std_pitch", "iris_rel_x", "iris_rel_y", "off_screen",
    ),
    "blink_patterns": ("ear", "ear_normalised", "is_blink", "blink_rate_window",
                       "geo_eye_openness_asymmetry"),
    "head_movement": ("yaw", "pitch", "roll", "head_std_yaw", "head_std_pitch",
                      "head_std_roll", "geo_nose_chin_dist"),
    "facial_dynamics": (
        "geo_face_width", "geo_face_height", "geo_aspect_ratio", "geo_mouth_openness",
        "geo_mouth_width", "geo_brow_eye_left", "geo_brow_eye_right",
        "geo_brow_asymmetry", "geo_landmark_energy", "geo_landmark_velocity",
    ),
}
STATISTIC_SUFFIXES = ("mean", "std", "min", "max")


def build_clip_matrix(
    model: MRGVMModel, loaders, geometric_columns: Sequence[str], device: torch.device,
    n_mamba_components: int = 8,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Assemble a clip-level feature matrix plus the model's own predictions.

    Columns are the pooled geometric statistics (interpretable, named) followed
    by a few PCA components of the Vision Mamba embedding (the appearance
    stream, kept comparable on the same axis).
    """
    model.eval()
    geometric_rows, mamba_rows, predictions, meta = [], [], [], []

    with torch.no_grad():
        for split, loader in loaders.items():
            for batch in loader:
                frames = batch["frames"].to(device)
                geometric = batch["geometric"].to(device)
                mrs = batch["mrs"].to(device)
                mask = batch["mask"].to(device)

                output = model(frames, geometric, mrs, mask)
                pooled = pool_geometric(
                    geometric, mask, mrs, model.cfg.fusion.geometric_pooling,
                    weight_by_mrs=model.cfg.vision_mamba.guide_pooling,
                )
                geometric_rows.append(pooled.cpu().numpy())
                if output["mamba_embedding"] is not None:
                    mamba_rows.append(output["mamba_embedding"].cpu().numpy())
                predictions.append(output["logits"].argmax(dim=1).cpu().numpy())
                for i in range(len(batch["clip_id"])):
                    meta.append({
                        "ClipID": batch["clip_id"][i],
                        "split": batch["split"][i],
                        "label": int(batch["label"][i].item()),
                    })

    geometric_matrix = np.vstack(geometric_rows)
    names: List[str] = []
    if model.cfg.fusion.geometric_pooling == "stats":
        for suffix in STATISTIC_SUFFIXES:
            names += [f"{c}_{suffix}" for c in geometric_columns]
    else:
        names = list(geometric_columns)

    frame = pd.DataFrame(geometric_matrix, columns=names)

    if mamba_rows:
        from sklearn.decomposition import PCA

        mamba_matrix = np.vstack(mamba_rows)
        components = min(n_mamba_components, mamba_matrix.shape[0], mamba_matrix.shape[1])
        reduced = PCA(n_components=components, random_state=0).fit_transform(mamba_matrix)
        for i in range(components):
            frame[f"vision_mamba_pc{i + 1}"] = reduced[:, i]

    meta_frame = pd.DataFrame(meta)
    for column in meta_frame.columns:
        frame[column] = meta_frame[column].to_numpy()
    return frame, np.concatenate(predictions), list(names)


def explain(
    output_root: Path, checkpoint_path: Path, device: torch.device, n_background: int = 30
) -> Dict[str, object]:
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = load_mrgvm_config(None, **checkpoint["config"])
    loaders, datasets, info = build_dataloaders(
        output_root, cfg.data, cfg.splits, cfg.vision_mamba.image_size
    )
    model = MRGVMModel(cfg, info["geometric_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    frame, predictions, geometric_names = build_clip_matrix(
        model, loaders, info["geometric_columns"], device
    )
    feature_names = [c for c in frame.columns if c not in ("ClipID", "split", "label")]
    X = frame[feature_names].to_numpy(dtype=np.float64)

    # Surrogate fitted to the MODEL's predictions, not the ground truth: SHAP is
    # explaining the model's behaviour, so the target must be what it predicts.
    surrogate = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=0, class_weight="balanced"
    )
    surrogate.fit(X, predictions)
    fidelity = float(accuracy_score(predictions, surrogate.predict(X)))
    logger.info("Surrogate fidelity to the MRG-VM model: %.3f", fidelity)
    if fidelity < 0.8:
        logger.warning(
            "Surrogate fidelity is %.3f -- below 0.8 the SHAP attributions below "
            "describe the surrogate more than they describe MRG-VM. Treat with caution.",
            fidelity,
        )

    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)

    # shap returns (n, features, classes) for multiclass trees in recent
    # versions and a list of per-class arrays in older ones; normalise both.
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(v) for v in shap_values], axis=-1)
    else:
        stacked = np.abs(np.asarray(shap_values))
        if stacked.ndim == 2:
            stacked = stacked[:, :, None]
    importance = stacked.mean(axis=(0, 2))
    total = importance.sum()
    normalised = importance / total if total > 0 else importance

    per_feature = sorted(
        ({"feature": n, "mean_abs_shap": float(v), "share": float(s)}
         for n, v, s in zip(feature_names, importance, normalised)),
        key=lambda r: r["mean_abs_shap"], reverse=True,
    )

    # ---- fold into behavioural groups ---------------------------------- #
    group_totals: Dict[str, float] = {name: 0.0 for name in FEATURE_GROUPS}
    group_totals["appearance_vision_mamba"] = 0.0
    unassigned = 0.0
    for name, value in zip(feature_names, importance):
        if name.startswith("vision_mamba_pc"):
            group_totals["appearance_vision_mamba"] += float(value)
            continue
        base = name
        for suffix in STATISTIC_SUFFIXES:
            if base.endswith("_" + suffix):
                base = base[: -(len(suffix) + 1)]
                break
        for group, members in FEATURE_GROUPS.items():
            if base in members:
                group_totals[group] += float(value)
                break
        else:
            unassigned += float(value)
    if unassigned > 0:
        group_totals["other"] = unassigned

    group_sum = sum(group_totals.values())
    group_shares = {
        k: (v / group_sum if group_sum > 0 else 0.0) for k, v in group_totals.items()
    }

    report = {
        "checkpoint": str(checkpoint_path),
        "surrogate_fidelity": fidelity,
        "n_clips": int(len(frame)),
        "n_features": len(feature_names),
        "per_feature_importance": per_feature,
        "group_importance": group_totals,
        "group_share": group_shares,
        "note": (
            "SHAP is computed over a RandomForest surrogate fitted to the MRG-VM "
            "model's own predictions. surrogate_fidelity reports how faithfully "
            "the surrogate reproduces those predictions; below ~0.8 the "
            "attributions should not be read as explanations of MRG-VM itself."
        ),
    }

    target_dir = output_root / "results_mrgvm"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "shap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(per_feature).to_csv(target_dir / "shap_feature_importance.csv", index=False)

    _render_plots(shap_values, X, feature_names, group_shares, target_dir)
    return report


def _render_plots(shap_values, X, feature_names, group_shares, target_dir: Path) -> None:
    """Write the SHAP summary bar chart and the behavioural-group breakdown."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        logger.warning("matplotlib/shap unavailable; skipping plots")
        return

    try:
        plt.figure(figsize=(9, 7))
        shap.summary_plot(
            shap_values, X, feature_names=feature_names, plot_type="bar",
            max_display=20, show=False,
        )
        plt.title("SHAP feature importance -- MRG-VM engagement prediction")
        plt.tight_layout()
        plt.savefig(target_dir / "shap_feature_importance.png", dpi=140)
        plt.close()

        groups = {k: v for k, v in sorted(group_shares.items(), key=lambda kv: -kv[1])}
        plt.figure(figsize=(8, 4.5))
        plt.barh(list(groups)[::-1], [v * 100 for v in list(groups.values())[::-1]])
        plt.xlabel("share of total |SHAP| (%)")
        plt.title("Behavioural feature-group contribution to engagement")
        plt.tight_layout()
        plt.savefig(target_dir / "shap_group_importance.png", dpi=140)
        plt.close()
        logger.info("Wrote SHAP plots to %s", target_dir)
    except Exception as exc:  # plotting must never fail the analysis
        logger.warning("SHAP plotting failed (%s); JSON/CSV still written", exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%H:%M:%S",
    )
    output_root = Path(args.output_root)
    checkpoint = args.checkpoint or (output_root / "checkpoints" / "mrgvm.pt")
    if not Path(checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}. Run mrgvm.train_mrgvm first.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    report = explain(output_root, Path(checkpoint), device)

    logger.info("=" * 74)
    logger.info("PHASE 7 -- SHAP behavioural group contributions")
    for group, share in sorted(report["group_share"].items(), key=lambda kv: -kv[1]):
        logger.info("  %-26s %5.1f%%", group, share * 100)
    logger.info("-" * 74)
    logger.info("  Top 10 individual features:")
    for row in report["per_feature_importance"][:10]:
        logger.info("    %-34s %.5f  (%4.1f%%)",
                    row["feature"], row["mean_abs_shap"], row["share"] * 100)
    logger.info("  Surrogate fidelity: %.3f", report["surrogate_fidelity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
