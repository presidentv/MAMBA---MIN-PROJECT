"""TEST 9-12 - fusion shapes, a real full forward pass, one training step, and
the tiny overfit test.

TEST 12 is the gate: a model that cannot drive the loss down on 16 clips it sees
repeatedly has a wiring or optimisation bug, and full training would only produce
an expensive-looking null result (spec section 26).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import CachedClipDataset, collate, load_calibration  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.model import MRGViTMamba  # noqa: E402
from src.preprocess import stage1_key, stage2_key  # noqa: E402
from src.train import fit_landmark_scaler  # noqa: E402
from src.utils import get_device, load_config, save_json, set_seed  # noqa: E402
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit-clips", type=int, default=16)
    ap.add_argument("--overfit-steps", type=int, default=250)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    dev_info = get_device()
    device = torch.device(dev_info.device)
    index = build_index(cfg)

    backbone = str(cfg["vit"]["model_name"])
    s1 = stage1_key(cfg)
    encoder_spec = ViTFrameEncoder(backbone, pretrained=True, freeze=True).spec
    s2 = stage2_key(cfg, s1, encoder_spec.name, encoder_spec.input_size)

    calibration = load_calibration(cfg)
    train_ds = CachedClipDataset(cfg, index, "train", s1, s2, calibration)
    model = MRGViTMamba(vit_dim=train_ds.vit_dim, landmark_dim=train_ds.landmark_dim,
                        cfg=cfg).to(device)
    fit_landmark_scaler(model, train_ds)
    model.to(device)

    problems = []
    num_classes = int(cfg["dataset"]["num_classes"])

    # ------------------------------------------------------- TEST 9: fusion
    print("=" * 72)
    print("TEST 9 - FUSION SHAPES")
    print("=" * 72)
    loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    feats = batch["vit_features"].to(device)
    mrs = batch["mrs"].to(device)
    land = batch["landmark_features"].to(device)

    model.eval()
    with torch.no_grad():
        logits, mid = model(feats, mrs, land, return_intermediates=True)

    print(f"vit_features         : {tuple(feats.shape)}")
    print(f"mrs                  : {tuple(mrs.shape)}   range [{mrs.min():.3f}, {mrs.max():.3f}]")
    print(f"weighted features    : {tuple(mid['weighted_features'].shape)}")
    print(f"mamba sequence       : {tuple(mid['temporal_sequence'].shape)}")
    print(f"mamba pooled         : {tuple(mid['temporal_pooled'].shape)}")
    print(f"deep projected       : {tuple(mid['deep_projected'].shape)}")
    print(f"landmark input       : {tuple(land.shape)}")
    print(f"landmark projected   : {tuple(mid['landmark_projected'].shape)}")
    print(f"fused                : {tuple(mid['fused'].shape)}")
    print(f"logits               : {tuple(logits.shape)}")

    proj = int(cfg["fusion"]["projected_dim"])
    checks = [
        (mid["weighted_features"].shape, feats.shape, "MRS weighting changed the shape"),
        (mid["deep_projected"].shape[-1], proj, "deep projection width"),
        (mid["landmark_projected"].shape[-1], proj, "landmark projection width"),
        (mid["fused"].shape[-1], proj * 2, "fused width"),
        (logits.shape, (feats.shape[0], num_classes), "logits shape"),
    ]
    for got, want, label in checks:
        if got != want:
            problems.append(f"{label}: got {got}, expected {want}")

    # ------------------------------------------- TEST 10: full forward pass
    print("\n" + "=" * 72)
    print("TEST 10 - FULL FORWARD PASS ON A REAL BATCH")
    print("=" * 72)
    print(f"clips in batch       : {batch['clip_id']}")
    print(f"labels               : {batch['label'].tolist()}")
    print(f"logits.shape         : {tuple(logits.shape)}   expected "
          f"[{feats.shape[0]}, {num_classes}]")
    print(f"logits finite        : {bool(torch.isfinite(logits).all())}")
    print(f"predictions          : {logits.argmax(-1).tolist()}")
    if not torch.isfinite(logits).all():
        problems.append("logits contain NaN/Inf")

    counts = model.parameter_counts()
    print(f"\nparameters total     : {counts['total']:,}")
    print(f"parameters trainable : {counts['trainable']:,}")
    for name, n in counts["per_module"].items():
        print(f"  {name:<18}: {n:,}")

    # --------------------------------------------- TEST 11: one training step
    print("\n" + "=" * 72)
    print("TEST 11 - ONE TRAINING STEP")
    print("=" * 72)
    model.train()
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    labels = batch["label"].to(device)

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    logits = model(feats, mrs, land)
    loss = loss_fn(logits, labels)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    optimiser.step()
    after_logits = model(feats, mrs, land)
    loss_after = loss_fn(after_logits, labels)

    changed = sum(1 for n, p in model.named_parameters()
                  if p.requires_grad and not torch.equal(p.detach(), before[n]))
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"loss before step     : {loss.item():.6f}")
    print(f"loss after step      : {loss_after.item():.6f}")
    print(f"global grad norm     : {grad_norm:.4f}")
    print(f"tensors updated      : {changed}/{total_trainable}")
    print(f"loss finite          : {bool(torch.isfinite(loss))}")

    if not torch.isfinite(loss) or not torch.isfinite(loss_after):
        problems.append("loss is NaN/Inf after one step")
    if not np.isfinite(grad_norm):
        problems.append("gradient norm is NaN/Inf")
    if changed == 0:
        problems.append("no parameters changed after optimiser.step()")

    # ------------------------------------------------ TEST 12: tiny overfit
    print("\n" + "=" * 72)
    print(f"TEST 12 - TINY OVERFIT ({args.overfit_clips} clips, {args.overfit_steps} steps)")
    print("=" * 72)
    set_seed(int(cfg["seed"]))
    small = Subset(train_ds, list(range(min(args.overfit_clips, len(train_ds)))))
    small_loader = DataLoader(small, batch_size=len(small), shuffle=False, collate_fn=collate)
    fixed = next(iter(small_loader))
    f2 = fixed["vit_features"].to(device)
    m2 = fixed["mrs"].to(device)
    l2 = fixed["landmark_features"].to(device)
    y2 = fixed["label"].to(device)
    print(f"labels in subset     : {y2.tolist()}")

    model2 = MRGViTMamba(vit_dim=train_ds.vit_dim, landmark_dim=train_ds.landmark_dim,
                         cfg=cfg).to(device)
    fit_landmark_scaler(model2, train_ds)
    model2.to(device)
    # Dropout is disabled for this diagnostic: the question is whether the model
    # *can* fit the data, not whether it generalises.
    for module in model2.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model2.train()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-3)
    losses = []
    for step in range(args.overfit_steps):
        out = model2(f2, m2, l2)
        loss2 = torch.nn.functional.cross_entropy(out, y2)
        opt2.zero_grad(set_to_none=True)
        loss2.backward()
        opt2.step()
        losses.append(float(loss2.item()))
        if step % max(1, args.overfit_steps // 10) == 0 or step == args.overfit_steps - 1:
            acc = (out.argmax(-1) == y2).float().mean().item()
            print(f"  step {step:>4}  loss {loss2.item():.6f}  train acc {acc:.3f}")

    model2.eval()
    with torch.no_grad():
        final = model2(f2, m2, l2)
        final_acc = (final.argmax(-1) == y2).float().mean().item()
    reduction = losses[0] - losses[-1]
    print(f"\ninitial loss         : {losses[0]:.6f}")
    print(f"final loss           : {losses[-1]:.6f}")
    print(f"reduction            : {reduction:.6f} ({100*reduction/max(losses[0],1e-9):.1f}%)")
    print(f"final train accuracy : {final_acc:.3f}")

    if not np.isfinite(losses).all():
        problems.append("overfit loss became NaN/Inf")
    if losses[-1] > 0.1 * losses[0]:
        problems.append(f"tiny overfit failed: loss only fell from {losses[0]:.4f} to "
                        f"{losses[-1]:.4f}. Do not start full training.")
    if final_acc < 0.95:
        problems.append(f"tiny overfit reached only {final_acc:.2f} accuracy on "
                        f"{len(small)} clips it saw {args.overfit_steps} times")

    save_json({
        "shapes": {
            "vit_features": list(feats.shape), "mrs": list(mrs.shape),
            "weighted_features": list(mid["weighted_features"].shape),
            "mamba_sequence": list(mid["temporal_sequence"].shape),
            "mamba_pooled": list(mid["temporal_pooled"].shape),
            "deep_projected": list(mid["deep_projected"].shape),
            "landmark_input": list(land.shape),
            "landmark_projected": list(mid["landmark_projected"].shape),
            "fused": list(mid["fused"].shape),
            "logits": list(logits.shape),
        },
        "parameters": counts,
        "training_step": {"loss_before": float(loss.item()),
                          "loss_after": float(loss_after.item()),
                          "grad_norm": float(grad_norm),
                          "tensors_updated": changed, "tensors_trainable": total_trainable},
        "tiny_overfit": {"clips": len(small), "steps": args.overfit_steps,
                         "initial_loss": losses[0], "final_loss": losses[-1],
                         "final_train_accuracy": final_acc,
                         "loss_curve": losses[::max(1, args.overfit_steps // 50)]},
        "problems": problems,
    }, "artifacts/full_pipeline_check.json")
    print("\nwrote artifacts/full_pipeline_check.json")

    print("\n" + "=" * 72)
    if problems:
        print("TESTS 9-12 FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("=" * 72)
        return 1
    print("TESTS 9, 10, 11, 12 PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
