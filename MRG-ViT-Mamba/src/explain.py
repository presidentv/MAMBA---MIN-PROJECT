"""SHAP attribution over the fusion representation (spec section 32).

Where the attribution is taken matters. The fused 512-d vector is
[deep_projected || landmark_projected], and both halves are *learned
projections* whose individual dimensions have no names -- attributing there can
only ever say "deep vs. landmark".

So the attribution input here is one step earlier:

    [ Mamba temporal embedding (d_model) || landmark clip features (3L, named) ]
                  |
        deep_branch / landmark_branch -> fusion -> MLP -> logits

Every landmark dimension carries a real name (mean_ear_left, delta_blendshape_
jawOpen, ...), so contributions can be grouped into eye openness, iris/gaze,
head pose, brow, mouth/jaw and face scale exactly as the spec asks, while the
deep branch is reported as one block.

SHAP is attribution, not causation: it says which inputs this model leaned on,
not which facial behaviours cause engagement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .landmarks import CLIP_FEATURE_NAMES, FEATURE_GROUPS
from .metrics import DEFAULT_CLASS_NAMES
from .utils import ensure_dir, save_json


class FusionHead(nn.Module):
    """The part of the model SHAP attributes over: branch projections, fusion, MLP."""

    def __init__(self, model, deep_dim: int):
        super().__init__()
        self.model = model
        self.deep_dim = int(deep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep_in = x[:, :self.deep_dim]
        land_in = x[:, self.deep_dim:]
        deep = self.model.deep_branch(deep_in)
        land = self.model.landmark_branch(land_in) if self.model.use_landmarks else None
        return self.model.classifier(self.model.fusion(deep, land))


def build_group_index(deep_dim: int, landmark_names: tuple[str, ...]) -> dict[str, list[int]]:
    """Map each reported group to its column indices in the attribution input."""
    groups: dict[str, list[int]] = {"deep_temporal": list(range(deep_dim))}

    # A landmark clip feature is <stat>_<frame feature name>; a frame feature
    # belongs to a group, so the clip feature inherits it across all three stats.
    name_to_group: dict[str, str] = {}
    for group, members in FEATURE_GROUPS.items():
        for member in members:
            name_to_group[member] = group

    for i, clip_name in enumerate(landmark_names):
        col = deep_dim + i
        base = clip_name.split("_", 1)[1] if "_" in clip_name else clip_name
        group = name_to_group.get(base, "other_landmark")
        groups.setdefault(group, []).append(col)
    return groups


@torch.no_grad()
def collect_inputs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the temporal half and return [N, d_model + 3L] plus the labels."""
    xs, ys = [], []
    model.eval()
    for batch in loader:
        feats = batch["vit_features"].to(device)
        mrs = batch["mrs"].to(device)
        land = batch["landmark_features"].to(device)
        if model.learnable_mrs is not None:
            mrs = model.learnable_mrs(batch["mrs_components"].to(device))
        weighted = model.mrs_weighting(feats, mrs)
        _, pooled = model.temporal(weighted, mrs)
        xs.append(torch.cat([pooled, land], dim=-1).float().cpu().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(xs), np.concatenate(ys)


def run_shap(model, train_loader, eval_loader, device, cfg,
             out_prefix: str = "main") -> dict:
    import shap

    background, _ = collect_inputs(model, train_loader, device)
    values_in, labels = collect_inputs(model, eval_loader, device)

    n_bg = min(int(cfg["explain"]["shap_background_size"]), len(background))
    n_ev = min(int(cfg["explain"]["shap_eval_size"]), len(values_in))
    rng = np.random.default_rng(int(cfg["seed"]))
    bg = background[rng.choice(len(background), n_bg, replace=False)]
    ev = values_in[rng.choice(len(values_in), n_ev, replace=False)]

    deep_dim = int(cfg["mamba"]["d_model"])
    head = FusionHead(model, deep_dim).to(device).eval()

    bg_t = torch.tensor(bg, dtype=torch.float32, device=device)
    ev_t = torch.tensor(ev, dtype=torch.float32, device=device)

    # GradientExplainer (expected gradients) rather than DeepExplainer: it does
    # not need a hand-written rule for every activation, so a GELU MLP works.
    explainer = shap.GradientExplainer(head, bg_t)
    shap_values = explainer.shap_values(ev_t)

    # Normalise the return shape across shap versions: want [N, F, C].
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(v) for v in shap_values], axis=-1)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            arr = arr[..., None]
    num_classes = arr.shape[-1]

    groups = build_group_index(deep_dim, CLIP_FEATURE_NAMES)
    mean_abs = np.abs(arr).mean(axis=0)                 # [F, C]
    overall = mean_abs.mean(axis=-1)                    # [F]

    group_totals = {g: float(overall[idx].sum()) for g, idx in groups.items()}
    total = sum(group_totals.values()) or 1.0
    group_share = {g: v / total for g, v in group_totals.items()}

    per_class_groups = {
        DEFAULT_CLASS_NAMES[c]: {g: float(mean_abs[idx, c].sum()) for g, idx in groups.items()}
        for c in range(min(num_classes, len(DEFAULT_CLASS_NAMES)))
    }

    landmark_scores = {
        name: float(overall[deep_dim + i]) for i, name in enumerate(CLIP_FEATURE_NAMES)
    }
    top_landmark = sorted(landmark_scores.items(), key=lambda kv: kv[1], reverse=True)[:30]

    art = ensure_dir(cfg["paths"]["artifacts_dir"])
    plot_path = art / f"shap_group_contributions_{out_prefix}.png"
    _plot_groups(group_share, plot_path, out_prefix)
    top_path = art / f"shap_top_landmark_features_{out_prefix}.png"
    _plot_top(top_landmark, top_path, out_prefix)

    result = {
        "method": "shap.GradientExplainer (expected gradients)",
        "attribution_input": "[Mamba temporal embedding (d_model) || landmark clip features (3L)]",
        "attribution_model": "deep/landmark branch projections -> fusion -> MLP -> logits",
        "background_samples": int(n_bg),
        "evaluated_samples": int(n_ev),
        "num_classes_in_output": int(num_classes),
        "deep_dim": deep_dim,
        "landmark_dim": len(CLIP_FEATURE_NAMES),
        "group_sizes": {g: len(idx) for g, idx in groups.items()},
        "group_total_mean_abs_shap": group_totals,
        "group_share_of_total": group_share,
        "group_share_per_class": {
            cls: {g: v / (sum(d.values()) or 1.0) for g, v in d.items()}
            for cls, d in per_class_groups.items()
        },
        "top_landmark_features": [{"feature": k, "mean_abs_shap": v} for k, v in top_landmark],
        "plots": {"groups": str(plot_path), "top_features": str(top_path)},
        "caveat": "SHAP reports what this model attended to, not causal structure. "
                  "Group totals are sums over differently sized groups: the deep block "
                  "has d_model dimensions and the landmark groups have far fewer, so a "
                  "larger total partly reflects a larger group. group_share_per_dimension "
                  "is given for a size-normalised view.",
        "group_share_per_dimension": {
            g: float(overall[idx].mean()) for g, idx in groups.items()
        },
    }
    save_json(result, art / f"shap_analysis_{out_prefix}.json")
    return result


def _plot_groups(group_share: dict, path: Path, prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(group_share.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(names) + 1.6), dpi=140)
    ax.barh(range(len(names)), vals, color="#3b6ea5")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("share of total mean |SHAP|")
    ax.set_title(f"Feature-group attribution ({prefix})")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v*100:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_top(top: list, path: Path, prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [k for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.32 * len(names) + 1.6), dpi=140)
    ax.barh(range(len(names)), vals, color="#7a9e3b")
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"Top landmark clip features ({prefix})")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
