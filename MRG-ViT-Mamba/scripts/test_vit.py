"""TEST 6 + TEST 7 + CHECKPOINT 6 - ViT shapes and MRS weighting.

Prints the *measured* embedding dimension rather than assuming 768, verifies the
[B,T,C,H,W] -> [B*T,...] -> [B,T,D] batching, and checks that MRS weighting
preserves the shape and the temporal ordering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal_mamba import MRSWeighting  # noqa: E402
from src.utils import Timer, get_device, load_config, save_json  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    backbone = args.backbone or str(cfg["vit"]["model_name"])
    T = int(cfg["video"]["num_frames"])
    B = args.batch

    print("=" * 72)
    print("TEST 6 / CHECKPOINT 6 - VIT SHAPES")
    print("=" * 72)
    print(f"backbone requested : {backbone}")

    with Timer() as t:
        encoder = ViTFrameEncoder(model_name=backbone,
                                  pretrained=bool(cfg["vit"]["pretrained"]),
                                  freeze=bool(cfg["vit"]["freeze"]),
                                  chunk_size=int(cfg["vit"]["batch_size"])).to(device.device)
    spec = encoder.spec
    print(f"load time          : {t.elapsed:.1f}s")
    print(f"resolved name      : {spec.name}")
    print(f"family             : {spec.family}")
    print(f"parameters         : {spec.num_params:,}")
    print(f"input size         : {spec.input_size}")
    print(f"normalisation mean : {spec.mean}")
    print(f"normalisation std  : {spec.std}")
    print(f"reported embed_dim : {spec.embed_dim}")

    S = spec.input_size
    frames = torch.randn(B, T, 3, S, S, device=device.device)
    print(f"\ninput shape        : {tuple(frames.shape)}")
    print(f"flattened shape    : {(B * T, 3, S, S)}")

    with Timer() as t:
        feats = encoder(frames)
    print(f"forward time       : {t.elapsed:.2f}s")
    print(f"ACTUAL output dim D: {feats.shape[-1]}")
    print(f"reshaped shape     : {tuple(feats.shape)}")
    print(f"device             : {feats.device}")
    print(f"dtype              : {feats.dtype}")

    problems = []
    if feats.shape != (B, T, spec.embed_dim):
        problems.append(f"expected {(B, T, spec.embed_dim)}, got {tuple(feats.shape)}")
    if not torch.isfinite(feats).all():
        problems.append("ViT output contains NaN/Inf")

    # ---------------------------------------------------- TEST 7: MRS weighting
    print("\n" + "=" * 72)
    print("TEST 7 - MRS WEIGHTING PRESERVES [B, T, D]")
    print("=" * 72)
    mrs = torch.rand(B, T, device=device.device)
    weighting = MRSWeighting(mode="multiply")
    weighted = weighting(feats, mrs)
    print(f"features.shape        : {tuple(feats.shape)}")
    print(f"mrs.shape             : {tuple(mrs.shape)}")
    print(f"weighted.shape        : {tuple(weighted.shape)}")
    print(f"shapes equal          : {weighted.shape == feats.shape}")

    if weighted.shape != feats.shape:
        problems.append("MRS weighting changed the tensor shape")

    # Temporal ordering must be untouched: position t must still be a scalar
    # multiple of the original position t, for every t.
    ratios = []
    for t_i in range(T):
        denom = feats[0, t_i]
        mask = denom.abs() > 1e-4
        if mask.any():
            ratios.append(float((weighted[0, t_i][mask] / denom[mask]).median()))
    expected = mrs[0].tolist()
    max_dev = max(abs(r - e) for r, e in zip(ratios, expected))
    print(f"per-position scale == MRS (max deviation): {max_dev:.2e}")
    if max_dev > 1e-3:
        problems.append(f"weighted features are not r_t * f_t at every position "
                        f"(max deviation {max_dev:.3e}) - ordering may have shifted")

    none_mode = MRSWeighting(mode="none")(feats, mrs)
    print(f"mode='none' is identity: {torch.equal(none_mode, feats)}")

    save_json({
        "backbone": spec.as_dict(),
        "batch": B, "num_frames": T,
        "input_shape": [B, T, 3, S, S],
        "flattened_shape": [B * T, 3, S, S],
        "output_shape": list(feats.shape),
        "actual_embed_dim": int(feats.shape[-1]),
        "device": str(feats.device),
        "dtype": str(feats.dtype),
        "mrs_weighting_preserves_shape": bool(weighted.shape == feats.shape),
        "mrs_scale_max_deviation": float(max_dev),
        "problems": problems,
    }, "artifacts/vit_shapes.json")
    print("\nwrote artifacts/vit_shapes.json")

    print("\n" + "=" * 72)
    if problems:
        print("CHECKPOINT 6 FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("=" * 72)
        return 1
    print("CHECKPOINT 6 + TEST 7 PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
