"""ViT frame encoder: one face crop in, one spatial embedding out.

The ViT is responsible for spatial information *within* a frame; all temporal
modelling happens later in Mamba (spec section 13).

The embedding dimension D is **discovered from the loaded model**, never assumed
to be 768. So is the expected input resolution and the normalisation statistics
-- a CLIP backbone and an ImageNet backbone disagree on both, and using the
wrong mean/std quietly costs accuracy without raising anything.

Two backbone families are supported:
  ``timm:<model_name>``  -- anything in the timm hub (default when unprefixed)
  ``hf:<repo_id>``       -- a transformers vision model, e.g. a ViT fine-tuned
                            for facial expression recognition

The point of supporting both is that ImageNet-supervised ViT-B/16 is a weak
default for faces. Which backbone is actually better here is an empirical
question, answered by scripts/compare_backbones.py, not by assertion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Curated candidates. Each entry is *a hypothesis to test*, with the reason it is
# worth testing. None of these is claimed to be best for engagement recognition
# until scripts/compare_backbones.py has been run and its artifact exists.
CANDIDATE_BACKBONES: dict[str, str] = {
    "timm:vit_base_patch16_224.augreg2_in21k_ft_in1k":
        "ImageNet-21k -> 1k supervised ViT-B/16. The conventional default; "
        "included as the reference point, not because faces are an ImageNet class.",
    "timm:vit_base_patch16_clip_224.openai":
        "CLIP ViT-B/16 image tower. Trained on web image-text pairs, so it has "
        "seen enormous numbers of people and expressions.",
    "timm:vit_small_patch14_dinov2.lvd142m":
        "DINOv2 ViT-S/14, self-supervised. Strong dense/part-aware features; "
        "small variant chosen so the 518px input stays affordable.",
    "hf:trpakov/vit-face-expression":
        "ViT-B/16 fine-tuned for facial expression recognition. Closest in "
        "domain to engagement, which is an affective read-out of the face.",
    "hf:dima806/facial_emotions_image_detection":
        "ViT fine-tuned on facial emotion data; a second in-domain candidate so "
        "the comparison does not rest on one fine-tune.",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class BackboneSpec:
    name: str
    family: str            # "timm" | "hf"
    embed_dim: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    num_params: int

    def as_dict(self) -> dict:
        return {
            "name": self.name, "family": self.family, "embed_dim": self.embed_dim,
            "input_size": self.input_size, "mean": list(self.mean), "std": list(self.std),
            "num_params": self.num_params,
        }


def _split_name(model_name: str) -> tuple[str, str]:
    if ":" in model_name:
        family, rest = model_name.split(":", 1)
        if family in ("timm", "hf"):
            return family, rest
    return "timm", model_name


class ViTFrameEncoder(nn.Module):
    """Wraps a pretrained backbone and exposes ``[B, T, C, H, W] -> [B, T, D]``.

    The [B,T,...] -> [B*T,...] flatten/restore is done once here rather than in a
    Python loop over frames (spec section 13).
    """

    def __init__(self, model_name: str, pretrained: bool = True, freeze: bool = True,
                 chunk_size: int = 32):
        super().__init__()
        self.model_name = model_name
        self.family, self.backbone_id = _split_name(model_name)
        self.chunk_size = int(chunk_size)
        self.frozen = bool(freeze)

        if self.family == "timm":
            self.backbone, spec = self._build_timm(self.backbone_id, pretrained)
        else:
            self.backbone, spec = self._build_hf(self.backbone_id, pretrained)
        self.spec = spec

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    # ------------------------------------------------------------------ build
    def _build_timm(self, name: str, pretrained: bool):
        import timm

        # num_classes=0 makes timm return the pooled pre-logits feature.
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        model.eval()

        embed_dim = int(getattr(model, "num_features", 0))
        if not embed_dim:
            raise RuntimeError(f"timm model {name} did not report num_features")

        # Resolve the model's own preprocessing. timm moved this API between
        # 0.9 and 1.x, so try the new location first and fall back.
        cfg = None
        try:
            from timm.data import resolve_model_data_config

            cfg = resolve_model_data_config(model)
        except Exception:
            try:
                from timm.data import resolve_data_config

                cfg = resolve_data_config({}, model=model)
            except Exception:
                cfg = None
        if cfg:
            input_size = int(cfg["input_size"][-1])
            mean = tuple(float(v) for v in cfg["mean"])
            std = tuple(float(v) for v in cfg["std"])
        else:
            input_size, mean, std = 224, IMAGENET_MEAN, IMAGENET_STD

        spec = BackboneSpec(f"timm:{name}", "timm", embed_dim, input_size, mean, std,
                            sum(p.numel() for p in model.parameters()))
        return model, spec

    def _build_hf(self, repo_id: str, pretrained: bool):
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "backbone name starts with 'hf:' but the transformers package is not "
                "installed. Install it, or use a 'timm:' backbone."
            ) from exc

        # add_pooling_layer=False matters. A HF image-classification checkpoint
        # stores no pooler weights, so AutoModel would *randomly initialise* one
        # and report it as MISSING; reading pooler_output would then return the
        # output of an untrained layer and quietly throw away everything the
        # fine-tune learned. ViTForImageClassification classifies from the CLS
        # token anyway, so that is what this encoder reads.
        kwargs = {"add_pooling_layer": False}
        if not pretrained:
            config = AutoConfig.from_pretrained(repo_id)
            try:
                model = AutoModel.from_config(config, **kwargs)
            except TypeError:
                model = AutoModel.from_config(config)
        else:
            try:
                model = AutoModel.from_pretrained(repo_id, **kwargs)
            except TypeError:
                model = AutoModel.from_pretrained(repo_id)
        model.eval()
        self._hf_has_pooler = getattr(model, "pooler", None) is not None

        processor = AutoImageProcessor.from_pretrained(repo_id)
        size = getattr(processor, "size", None) or {}
        input_size = int(size.get("height") or size.get("shortest_edge") or 224)
        mean = tuple(float(v) for v in getattr(processor, "image_mean", IMAGENET_MEAN))
        std = tuple(float(v) for v in getattr(processor, "image_std", IMAGENET_STD))

        hidden = getattr(model.config, "hidden_size", None)
        if hidden is None:
            raise RuntimeError(f"could not read hidden_size from {repo_id} config")

        spec = BackboneSpec(f"hf:{repo_id}", "hf", int(hidden), input_size, mean, std,
                            sum(p.numel() for p in model.parameters()))
        return model, spec

    # --------------------------------------------------------------- forward
    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.family == "timm":
            return self.backbone(x)
        out = self.backbone(pixel_values=x)
        # CLS token, deliberately in preference to pooler_output -- see _build_hf.
        hidden = out.last_hidden_state
        return hidden[:, 0] if hidden.dim() == 3 else hidden.mean(dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] -> [B, T, D]."""
        if frames.dim() != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        flat = frames.reshape(b * t, *frames.shape[2:])

        outputs = []
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            for i in range(0, flat.shape[0], self.chunk_size):
                outputs.append(self._forward_backbone(flat[i:i + self.chunk_size]))
        feats = torch.cat(outputs, dim=0)

        if feats.shape[-1] != self.spec.embed_dim:
            raise RuntimeError(
                f"backbone reported embed_dim={self.spec.embed_dim} but produced "
                f"{feats.shape[-1]}; refusing to continue with an unverified dimension"
            )
        return feats.reshape(b, t, self.spec.embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self


def preprocess_crops(crops_bgr: np.ndarray, spec: BackboneSpec) -> torch.Tensor:
    """uint8 BGR [T, H, W, 3] -> normalised float tensor [T, 3, S, S].

    Uses the backbone's *own* mean/std and input size, resolved at load time.
    """
    import cv2

    size = spec.input_size
    out = np.empty((crops_bgr.shape[0], size, size, 3), dtype=np.float32)
    for i, crop in enumerate(crops_bgr):
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    tensor = torch.from_numpy(out).permute(0, 3, 1, 2)
    mean = torch.tensor(spec.mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std
