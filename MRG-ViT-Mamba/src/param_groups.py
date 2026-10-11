"""Optimiser parameter groups for fine-tuning a pretrained ViT.

Two standard practices that the first fine-tuning run did not apply, both of
which matter most exactly in this regime -- a large pretrained backbone, a small
dataset:

*   **Layer-wise learning-rate decay.** Early transformer blocks hold generic
    low-level features that transfer; late blocks hold task-specific ones that
    need to move. One rate for all twelve blocks either disturbs the early
    layers or under-trains the late ones. Block *i* gets
    ``backbone_lr * decay**(n_layers - i)``, so the rate falls geometrically
    towards the patch embedding.

*   **No weight decay on norms, biases, the class token or position
    embeddings.** Decaying a LayerNorm gain pulls it towards zero, which is not
    regularisation but destruction of the pretrained scale; the same applies to
    the class token and position embeddings. Only matrix weights are decayed.

Returns plain dicts so the caller stays in charge of the optimiser.
"""

from __future__ import annotations

import torch.nn as nn


def _timm_layer_id(name: str, n_blocks: int) -> int:
    """0 = stem (patch embed / tokens), 1..n_blocks = blocks, n_blocks+1 = head side."""
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
        return 0
    if name.startswith("blocks."):
        try:
            return int(name.split(".")[1]) + 1
        except (IndexError, ValueError):
            return n_blocks + 1
    return n_blocks + 1          # final norm and anything else at the top


def _no_decay(module: nn.Module, name: str, param) -> bool:
    """True for parameters that must not be weight-decayed."""
    if param.ndim <= 1:          # every bias and every norm gain/shift
        return True
    skip = set()
    if hasattr(module, "no_weight_decay"):
        try:
            skip = set(module.no_weight_decay())
        except Exception:
            skip = set()
    return name in skip or name.split(".")[0] in skip


def build_param_groups(model, *, head_lr: float, backbone_lr: float,
                       weight_decay: float, mrs_lr: float,
                       layer_decay: float | None = None) -> tuple[list[dict], dict]:
    """Group the model's trainable parameters for AdamW.

    ``layer_decay`` of None or 1.0 gives every backbone parameter the same
    ``backbone_lr`` -- the original behaviour, kept so the two recipes can be
    compared.
    """
    encoder = getattr(model, "vit_encoder", None)
    backbone = getattr(encoder, "backbone", None) if encoder is not None else None
    blocks = getattr(backbone, "blocks", None) if backbone is not None else None
    n_blocks = len(blocks) if blocks is not None else 0
    use_llrd = bool(layer_decay) and layer_decay != 1.0 and n_blocks > 0

    groups: dict[tuple, dict] = {}
    summary = {"layer_decay": layer_decay if use_llrd else None,
               "n_backbone_blocks": n_blocks, "groups": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("learnable_mrs."):
            key, lr, wd = ("mrs", 0), mrs_lr, 0.0
        elif name.startswith("vit_encoder."):
            inner = name[len("vit_encoder.backbone."):] if ".backbone." in name else name
            decay = not _no_decay(backbone, inner, param)
            layer = _timm_layer_id(inner, n_blocks) if use_llrd else 0
            scale = (layer_decay ** (n_blocks + 1 - layer)) if use_llrd else 1.0
            key = ("backbone", layer, decay)
            lr, wd = backbone_lr * scale, (weight_decay if decay else 0.0)
        else:
            decay = param.ndim > 1
            key = ("head", decay)
            lr, wd = head_lr, (weight_decay if decay else 0.0)

        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd,
                                    "name": "/".join(str(k) for k in key)})
        g["params"].append(param)

    ordered = [groups[k] for k in sorted(groups, key=lambda k: (str(k[0]), k[1:]))]
    for g in ordered:
        summary["groups"].append({
            "name": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"],
            "n_params": int(sum(p.numel() for p in g["params"])),
        })
    summary["n_groups"] = len(ordered)
    return ordered, summary
