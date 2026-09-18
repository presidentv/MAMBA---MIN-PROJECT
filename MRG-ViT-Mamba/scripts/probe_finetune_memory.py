"""Measure what a fine-tuning step of the ViT actually costs on this GPU.

Fine-tuning at T=32 puts batch_size x 32 crops through the backbone per step with
activations retained for backward, which is a very different memory profile from
the frozen cached-feature path. Rather than guessing a batch size and discovering
an out-of-memory error 40 minutes into a run, this probes the real peak
allocation for a forward+backward over a range of settings and prints a table.

Run:  python scripts/probe_finetune_memory.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, resolve_path  # noqa: E402


def probe(model_name: str, n_crops: int, size: int, checkpointing: bool,
          unfreeze_blocks: int | None, amp: bool) -> dict:
    """One forward+backward; returns peak MiB or the OOM message."""
    import timm

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dev = torch.device("cuda")

    model = timm.create_model(model_name, pretrained=False, num_classes=0).to(dev)
    if checkpointing:
        # timm exposes this uniformly; it recomputes block activations in
        # backward instead of storing them.
        model.set_grad_checkpointing(True)

    blocks = getattr(model, "blocks", None)
    total_blocks = len(blocks) if blocks is not None else 0
    if unfreeze_blocks is not None and blocks is not None:
        for p in model.parameters():
            p.requires_grad = False
        for blk in blocks[len(blocks) - unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(model, "norm"):
            for p in model.norm.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    x = torch.randn(n_crops, 3, size, size, device=dev)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    try:
        with torch.amp.autocast("cuda", enabled=amp):
            feats = model(x)
            loss = feats.float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        peak = torch.cuda.max_memory_allocated() / 2**20
        result = {"peak_mib": round(peak, 1), "ok": True}
    except torch.cuda.OutOfMemoryError as exc:
        result = {"peak_mib": None, "ok": False, "error": str(exc).split("\n")[0][:90]}

    result.update({"n_crops": n_crops, "checkpointing": checkpointing,
                   "unfreeze_blocks": unfreeze_blocks, "amp": amp,
                   "trainable_params": trainable, "total_blocks": total_blocks})
    del model, opt, x
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; fine-tuning probe is meaningless on CPU")
        return 1

    cfg = load_config(resolve_path(args.config))
    model_name = cfg["vit"]["model_name"].split(":", 1)[-1]
    size = int(cfg["face"]["image_size"])
    budget = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"device: {torch.cuda.get_device_name(0)}  ({budget:.0f} MiB)")
    print(f"backbone: {model_name}  input {size}px\n")

    rows = []
    # (crops per step, grad checkpointing, blocks left trainable)
    grid = [
        (256, False, None), (256, True, None),
        (128, True, None), (64, True, None), (32, True, None),
        (64, True, 4), (128, True, 4), (256, True, 4),
        (64, True, 6), (128, True, 6),
    ]
    for n, ckpt, unf in grid:
        r = probe(model_name, n, size, ckpt, unf, amp=True)
        rows.append(r)
        unf_s = "all" if unf is None else f"last {unf}"
        peak = f"{r['peak_mib']:>8.1f}" if r["ok"] else "     OOM"
        head = "  <= fits" if (r["ok"] and r["peak_mib"] < budget * 0.80) else ""
        print(f"  crops={n:>4}  ckpt={str(ckpt):<5}  trainable={unf_s:<8} "
              f"params={r['trainable_params']/1e6:>6.1f}M  peak={peak} MiB{head}")

    out = resolve_path("artifacts/finetune_memory_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"device": torch.cuda.get_device_name(0), "budget_mib": round(budget, 1),
         "backbone": model_name, "input_size": size, "rows": rows},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
