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


# --------------------------------------------------------------------------- #
# Per-clip explanations for the fine-tuned model (scripts/explain_finetuned.py)
# --------------------------------------------------------------------------- #
def choose_examples(predictions: list[dict], n: int = 3) -> list[dict]:
    """Pick `n` test clips from `n` different people, chosen to be informative.

    In order: the most confident correct prediction; the most confident mistake
    (on a class not yet shown, if there is one); then the rarest class not yet
    shown. Every pick is from a new subject. Deterministic: ties break on clip id.
    Each prediction needs clip_id, subject_id, true, pred and probs.
    """
    ranked = sorted(predictions, key=lambda p: (-p["probs"][p["pred"]], p["clip_id"]))
    chosen: list[dict] = []
    subjects: set = set()

    def take(cond) -> bool:
        for p in ranked:
            if p["subject_id"] not in subjects and cond(p):
                chosen.append(p)
                subjects.add(p["subject_id"])
                return True
        return False

    def shown() -> set:
        return {p["true"] for p in chosen}

    counts: dict[int, int] = {}
    for p in predictions:
        counts[p["true"]] = counts.get(p["true"], 0) + 1

    wants = [
        [lambda p: p["pred"] == p["true"]],
        [lambda p: p["pred"] != p["true"] and p["true"] not in shown(),
         lambda p: p["pred"] != p["true"]],
        [lambda p, c=c: p["true"] == c and c not in shown()
         for c in sorted(counts, key=lambda c: (counts[c], c))],
    ]
    for alternatives in wants:
        if len(chosen) >= n:
            break
        for cond in alternatives:
            if take(cond):
                break
    while len(chosen) < n and take(lambda p: True):
        pass
    return chosen[:n]


def pick_frame(weights) -> int:
    """The frame the model weighted most; ties go to the one nearest the middle."""
    w = np.asarray(weights, dtype=np.float64)
    candidates = np.flatnonzero(np.isclose(w, w.max()))
    middle = (len(w) - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - middle))])


def integrated_gradients(f, x: torch.Tensor, baseline: torch.Tensor,
                         steps: int = 64) -> tuple[torch.Tensor, float]:
    """Integrated gradients of scalar-per-row `f` from `baseline` to `x` (both [D]).

    Returns (attribution [D], completeness error). The attributions sum to
    f(x) - f(baseline) up to the returned error, which shrinks with `steps`.
    """
    x = x.detach().float()
    baseline = baseline.detach().float().to(x.device)
    alphas = ((torch.arange(steps, device=x.device, dtype=torch.float32) + 0.5) / steps)[:, None]
    path = (baseline + alphas * (x - baseline)).requires_grad_(True)
    with torch.enable_grad():
        grads = torch.autograd.grad(f(path).sum(), path)[0]
    attr = (x - baseline) * grads.mean(dim=0)
    with torch.no_grad():
        gap = float(f(x[None])[0] - f(baseline[None])[0])
    return attr.detach(), float(attr.sum()) - gap


class ViTGradCAM:
    """Grad-CAM for a timm Vision Transformer.

    Hooks the input to the last block's attention (blocks[-1].norm1), the usual
    target layer for ViT Grad-CAM, and turns the patch tokens back into the
    patch grid. Works whether the encoder runs the frames in one chunk or many.
    """

    def __init__(self, backbone: nn.Module):
        blocks = getattr(backbone, "blocks", None)
        if blocks is None or not hasattr(backbone, "patch_embed"):
            raise ValueError("Grad-CAM needs a timm VisionTransformer backbone")
        self.prefix = int(getattr(backbone, "num_prefix_tokens", 1))
        self.grid = tuple(int(g) for g in backbone.patch_embed.grid_size)
        self.acts: list[torch.Tensor] = []
        self._handle = blocks[-1].norm1.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if output.requires_grad:
            output.retain_grad()
        self.acts.append(output)

    def maps(self) -> np.ndarray:
        """[frames, grid_h, grid_w], each map scaled to [0, 1]."""
        if not self.acts or self.acts[0].grad is None:
            raise RuntimeError("run a forward and a backward pass before reading Grad-CAM")
        a = torch.cat([t.detach() for t in self.acts]).float()[:, self.prefix:]
        g = torch.cat([t.grad for t in self.acts]).float()[:, self.prefix:]
        h, w = self.grid
        a = a.reshape(a.shape[0], h, w, -1)
        g = g.reshape(g.shape[0], h, w, -1)
        cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(dim=-1))
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        self._handle.remove()


def explain_clip(model, item: dict, device, ig_steps: int = 64) -> dict:
    """Everything needed to show how the fine-tuned model reached one prediction.

    item is one FineTuneClipDataset item. Returns the prediction, the per-frame
    reliability and pooling weights, a Grad-CAM map per frame, and two
    integrated-gradients decompositions of the predicted-class logit: over the
    fused vector (deep branch vs landmark branch, baseline all-zero), and over
    the named landmark clip features (baseline the training-set mean clip).
    """
    from .fusion import StandardScaler

    model.eval()
    enc = model.vit_encoder
    frames = item["frames"][None].to(device).requires_grad_(True)
    mrs = item["mrs"][None].to(device)
    comps = item["mrs_components"][None].to(device)
    land = item["landmark_features"][None].to(device)

    was_frozen = enc.frozen
    enc.frozen = False                  # the encoder only builds a graph when not frozen
    cam = ViTGradCAM(enc.backbone)
    try:
        # Full precision, like the final evaluation in run_finetune.py, so the
        # prediction explained here is the one that was reported.
        with torch.enable_grad():
            feats = enc(frames)
            logits, mid = model(feats, mrs, land, return_intermediates=True,
                                mrs_components=comps)
        logits = logits.float()
        pred = int(logits.argmax(-1)[0])
        logits[0, pred].backward()
        cams = cam.maps()
    finally:
        cam.close()
        enc.frozen = was_frozen
        model.zero_grad(set_to_none=True)

    probs = torch.softmax(logits.detach(), -1)[0].cpu().numpy()
    r = mid["mrs_used"][0].detach().float().cpu().numpy()
    pooling = model.temporal.pooling
    if pooling == "mrs_weighted":
        pool_w = r / (r.sum() + 1e-6)
    elif pooling == "mean":
        pool_w = np.full_like(r, 1.0 / len(r))
    else:
        pool_w = None                   # "last" / "attention": not a fixed per-frame weight

    fused = mid["fused"].detach().float()[0]
    deep = mid["deep_projected"].detach().float()
    deep_dim = deep.shape[-1]

    def head(z):
        return model.classifier(z)[:, pred]

    fused_attr, fused_err = integrated_gradients(head, fused, torch.zeros_like(fused), ig_steps)
    with torch.no_grad():
        zero_logit = float(head(torch.zeros_like(fused)[None])[0])

    landmark = None
    if model.landmark_branch is not None:
        norm = model.landmark_branch.norm
        x = land[0].float()
        base = norm.mean.float() if isinstance(norm, StandardScaler) else torch.zeros_like(x)

        def via_landmarks(v):
            return model.classifier(model.fusion(deep.expand(v.shape[0], -1),
                                                 model.landmark_branch(v)))[:, pred]

        attr, err = integrated_gradients(via_landmarks, x, base, ig_steps)
        names = (list(CLIP_FEATURE_NAMES) if len(CLIP_FEATURE_NAMES) == x.shape[0]
                 else [f"feature_{i}" for i in range(x.shape[0])])
        groups = {g: cols for g, cols in build_group_index(0, tuple(names)).items()
                  if g != "deep_temporal"}
        attr_np = attr.cpu().numpy()
        landmark = {
            "names": names,
            "attribution": attr_np,
            "value": x.cpu().numpy(),
            "train_mean": base.cpu().numpy(),
            "group_attribution": {g: float(attr_np[c].sum()) for g, c in groups.items()},
            "completeness_error": err,
        }

    return {
        "pred": pred,
        "true": int(item["label"]),
        "probs": probs,
        "logits": logits.detach()[0].cpu().numpy(),
        "mrs": r,
        "mrs_components": item["mrs_components"].float().cpu().numpy(),
        "pool_weights": pool_w,
        "pooling": pooling,
        "gradcam": cams,
        "logit_decomposition": {
            "zero_input_logit": zero_logit,
            "deep_branch": float(fused_attr[:deep_dim].sum()),
            "landmark_branch": float(fused_attr[deep_dim:].sum()),
            "predicted_logit": float(logits.detach()[0, pred]),
            "completeness_error": fused_err,
        },
        "landmark_attribution": landmark,
    }
