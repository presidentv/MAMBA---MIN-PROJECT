"""End-to-end fine-tuning of the ViT backbone together with the temporal head.

This is a different training regime from src/train.py, not a flag on it:

  * src/train.py reads the Stage 2 cache, which is a frozen backbone's output.
    An epoch is ~1 second because the ViT never runs.
  * here the ViT runs inside the graph on every step, so the Stage 2 cache is
    meaningless and the crops are re-encoded each time. An epoch costs minutes.

Three things make this fit on a 6 GB card at T=32 (measured in
artifacts/finetune_memory_probe.json):

  * gradient checkpointing on the backbone -- 256 crops/step costs 17.5 GB
    without it and 4.2 GB with it;
  * a small per-step batch with gradient accumulation, so the *effective* batch
    still matches the frozen run and the two remain comparable;
  * mixed precision.

The backbone gets its own much smaller learning rate. Driving pretrained
weights at the head's rate destroys the representation in the first few steps,
which shows up as training accuracy that never leaves chance.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import FineTuneClipDataset, collate_finetune, load_calibration
from .fusion import StandardScaler
from .losses import build_loss
from .metrics import compute_metrics
from .model import MRGViTMamba
from .mrs import COMPONENTS
from .preprocess import stage1_key
from .utils import ensure_dir, environment_record, gpu_memory_mb, save_json, set_seed
from .vit_encoder import ViTFrameEncoder

HISTORY_FIELDS = [
    "epoch", "train_loss", "train_accuracy", "train_macro_f1", "train_weighted_f1",
    "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1",
    "head_lr", "backbone_lr", "gpu_memory_mb", "seconds",
] + [f"w_{c}" for c in COMPONENTS]


@dataclass
class FineTuneResult:
    best_epoch: int
    best_val_macro_f1: float
    best_checkpoint: str
    history: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def build_finetune_datasets(cfg, index, s1_key: str, spec, splits=("train", "val", "test")):
    calibration = load_calibration(cfg)
    return {
        split: FineTuneClipDataset(cfg, index, split, s1_key, spec, calibration)
        for split in splits
    }


def fit_landmark_scaler(model: MRGViTMamba, train_ds: FineTuneClipDataset) -> dict:
    """Fit the landmark branch scaler on the training split only."""
    if model.landmark_branch is None:
        return {"fitted": False, "reason": "landmark branch disabled"}
    norm = model.landmark_branch.norm
    if not isinstance(norm, StandardScaler):
        return {"fitted": False, "reason": f"landmark norm is {type(norm).__name__}"}
    matrix = train_ds.landmark_matrix()
    norm.fit(matrix)
    return {"fitted": True, "n_train_clips": int(matrix.shape[0]),
            "dim": int(matrix.shape[1])}


def configure_backbone(encoder: ViTFrameEncoder, cfg_vit: dict, logger=None) -> dict:
    """Unfreeze the backbone, optionally only its last N blocks, and checkpoint it."""
    backbone = encoder.backbone
    encoder.frozen = False

    for p in backbone.parameters():
        p.requires_grad = True

    unfreeze_n = cfg_vit.get("unfreeze_blocks")
    blocks = getattr(backbone, "blocks", None)
    if unfreeze_n is not None and blocks is not None:
        unfreeze_n = int(unfreeze_n)
        for p in backbone.parameters():
            p.requires_grad = False
        for blk in list(blocks)[len(blocks) - unfreeze_n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(backbone, "norm"):
            for p in backbone.norm.parameters():
                p.requires_grad = True

    if bool(cfg_vit.get("grad_checkpointing", False)):
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(True)
        elif hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        else:
            raise RuntimeError(
                "grad_checkpointing was requested but this backbone exposes no way "
                "to enable it; fine-tuning at T=32 will not fit without it")

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    info = {
        "frozen": False,
        "total_blocks": len(blocks) if blocks is not None else None,
        "unfreeze_blocks": unfreeze_n,
        "grad_checkpointing": bool(cfg_vit.get("grad_checkpointing", False)),
        "backbone_params_total": int(total),
        "backbone_params_trainable": int(trainable),
    }
    if logger:
        logger.info("backbone: %.1fM/%.1fM params trainable, checkpointing=%s",
                    trainable / 1e6, total / 1e6, info["grad_checkpointing"])
    return info


def _run_epoch(model, loader, loss_fn, optimiser, device, scaler=None,
               grad_clip=0.0, accum_steps=1, collect_probs=False):
    """One pass. Gradients accumulate over ``accum_steps`` batches per update."""
    training = optimiser is not None
    model.train(training)

    losses, y_true, y_pred = [], [], []
    probs_out, clip_ids, subject_ids = [], [], []
    use_amp = scaler is not None

    if training:
        optimiser.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)
        mrs = batch["mrs"].to(device, non_blocking=True)
        comps = batch["mrs_components"].to(device, non_blocking=True)
        land = batch["landmark_features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=use_amp):
                feats = model.vit_encoder(frames)
                logits = model(feats, mrs, land, mrs_components=comps)
                loss = loss_fn(logits, labels)

        if training:
            # Scale so that accumulated gradients average rather than sum;
            # otherwise the effective learning rate silently multiplies.
            scaled = loss / accum_steps
            if use_amp:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            is_last = (step + 1) == len(loader)
            if (step + 1) % accum_steps == 0 or is_last:
                if grad_clip:
                    if use_amp:
                        scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimiser.param_groups for p in g["params"]], grad_clip)
                if use_amp:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                optimiser.zero_grad(set_to_none=True)

        if not torch.isfinite(loss):
            raise RuntimeError("loss became NaN/Inf - stopping rather than logging a "
                               "meaningless number")

        losses.append(float(loss.item()))
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(logits.detach().float().argmax(dim=-1).cpu().numpy())
        if collect_probs:
            probs_out.append(torch.softmax(logits.detach().float(), dim=-1).cpu().numpy())
            clip_ids.extend(batch["clip_id"])
            subject_ids.extend(batch["subject_id"])

    out = (float(np.mean(losses)), np.concatenate(y_true), np.concatenate(y_pred))
    if collect_probs:
        return out + (np.concatenate(probs_out), clip_ids, subject_ids)
    return out


def finetune_model(cfg, index, run_name: str = "ft32", logger=None,
                   overrides: dict | None = None) -> FineTuneResult:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    tcfg = dict(cfg["training"])
    if overrides:
        tcfg.update(overrides)

    set_seed(int(cfg.get("seed", 42)))

    # The backbone is built first: its spec determines the crop normalisation
    # the dataset must apply, so the dataset cannot be built before it exists.
    vcfg = dict(cfg["vit"])
    encoder = ViTFrameEncoder(
        model_name=vcfg["model_name"],
        pretrained=bool(vcfg.get("pretrained", True)),
        freeze=False,
        chunk_size=int(vcfg.get("batch_size", 32)),
    )
    backbone_info = configure_backbone(encoder, vcfg, logger)
    spec = encoder.spec

    s1_key = stage1_key(cfg)
    datasets = build_finetune_datasets(cfg, index, s1_key, spec, ("train", "val", "test"))
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    if logger:
        logger.info("clips: train=%d val=%d test=%d | T=%d  landmark_dim=%d  vit_dim=%d",
                    len(train_ds), len(val_ds), len(test_ds), train_ds.num_frames,
                    train_ds.landmark_dim, train_ds.vit_dim)
        logger.info("train label counts: %s", train_ds.label_counts())

    batch_size = int(tcfg["batch_size"])
    accum_steps = max(1, int(tcfg.get("grad_accum_steps", 1)))
    workers = int(tcfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_finetune, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_finetune, num_workers=workers)

    model = MRGViTMamba(vit_dim=spec.embed_dim, landmark_dim=train_ds.landmark_dim,
                        cfg=cfg, vit_encoder=encoder).to(device)
    scaler_info = fit_landmark_scaler(model, train_ds)
    model.to(device)

    loss_fn, loss_info = build_loss(tcfg, train_ds.label_counts(),
                                    int(cfg["dataset"]["num_classes"]))
    loss_fn = loss_fn.to(device)

    # Three parameter groups: the pretrained backbone at a small rate, the
    # freshly initialised head at the normal rate, and the five MRS weights
    # faster still with no decay (see the note in src/train.py).
    backbone_params, mrs_params, head_params = [], [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if name.startswith("vit_encoder."):
            backbone_params.append(prm)
        elif name.startswith("learnable_mrs."):
            mrs_params.append(prm)
        else:
            head_params.append(prm)

    head_lr = float(tcfg["learning_rate"])
    backbone_lr = float(vcfg.get("finetune_lr", head_lr / 30.0))
    groups = [{"params": head_params, "lr": head_lr,
               "weight_decay": float(tcfg["weight_decay"]), "name": "head"}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr,
                       "weight_decay": float(tcfg["weight_decay"]), "name": "backbone"})
    if mrs_params:
        groups.append({"params": mrs_params,
                       "lr": float(tcfg.get("mrs_weight_lr", 0.02)),
                       "weight_decay": 0.0, "name": "mrs"})
    optimiser = torch.optim.AdamW(groups)

    epochs = int(tcfg["epochs"])
    warmup = int(tcfg.get("warmup_epochs", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if str(tcfg.get("scheduler", "cosine")) != "cosine":
            return 1.0
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    use_amp = bool(tcfg.get("mixed_precision", False)) and device_str == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = ensure_dir(Path(cfg["paths"]["checkpoints_dir"]) / run_name)
    log_dir = ensure_dir(cfg["paths"]["logs_dir"])
    history_path = log_dir / f"training_history_{run_name}.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=HISTORY_FIELDS).writeheader()

    def checkpoint_payload(epoch, val_metrics):
        return {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "config": dict(cfg),
            "vit_dim": int(spec.embed_dim),
            "landmark_dim": train_ds.landmark_dim,
            "s1_key": s1_key,
            "s2_key": None,
            "num_frames": train_ds.num_frames,
            "mrs_mode": train_ds.mrs_mode,
            "loss_info": loss_info,
            "scaler_info": scaler_info,
            "backbone_info": backbone_info,
            "backbone_spec": spec.as_dict(),
            "finetuned": True,
        }

    best = {"macro_f1": -1.0, "epoch": -1}
    patience = int(tcfg.get("early_stopping_patience", epochs))
    since_improved = 0
    history: list[dict] = []
    num_classes = int(cfg["dataset"]["num_classes"])

    for epoch in range(epochs):
        t0 = time.perf_counter()
        if device_str == "cuda":
            torch.cuda.reset_peak_memory_stats()

        tr_loss, tr_true, tr_pred = _run_epoch(
            model, train_loader, loss_fn, optimiser, device, amp_scaler,
            float(tcfg.get("grad_clip", 0.0)), accum_steps)
        va_loss, va_true, va_pred = _run_epoch(model, val_loader, loss_fn, None, device)
        scheduler.step()

        m_tr = compute_metrics(tr_true, tr_pred, num_classes)
        m_va = compute_metrics(va_true, va_pred, num_classes)
        lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimiser.param_groups)}
        row = {
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "train_accuracy": round(m_tr["accuracy"], 6),
            "train_macro_f1": round(m_tr["macro_f1"], 6),
            "train_weighted_f1": round(m_tr["weighted_f1"], 6),
            "val_loss": round(va_loss, 6),
            "val_accuracy": round(m_va["accuracy"], 6),
            "val_macro_f1": round(m_va["macro_f1"], 6),
            "val_weighted_f1": round(m_va["weighted_f1"], 6),
            "head_lr": lrs.get("head"), "backbone_lr": lrs.get("backbone"),
            "gpu_memory_mb": round(gpu_memory_mb() or 0.0, 1),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        if model.learnable_mrs is not None:
            w = model.learnable_mrs.weights().detach().cpu().numpy()
            row.update({f"w_{c}": round(float(w[i]), 6) for i, c in enumerate(COMPONENTS)})
        else:
            row.update({f"w_{c}": "" for c in COMPONENTS})
        history.append(row)
        with open(history_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=HISTORY_FIELDS).writerow(row)

        if logger:
            logger.info(
                "ep %3d | train loss %.4f acc %.3f mF1 %.3f | val loss %.4f acc %.3f "
                "mF1 %.3f | head lr %.2e bb lr %.2e | %.1fs %.0fMB",
                epoch, tr_loss, m_tr["accuracy"], m_tr["macro_f1"], va_loss,
                m_va["accuracy"], m_va["macro_f1"], row["head_lr"] or 0.0,
                row["backbone_lr"] or 0.0, row["seconds"], row["gpu_memory_mb"])

        # Every epoch is stored, as with the earlier run, plus a separate best.
        torch.save(checkpoint_payload(epoch, m_va), ckpt_dir / f"epoch_{epoch:03d}.pt")

        if m_va["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m_va["macro_f1"], "epoch": epoch,
                    "accuracy": m_va["accuracy"], "weighted_f1": m_va["weighted_f1"],
                    "loss": va_loss}
            torch.save(checkpoint_payload(epoch, m_va), ckpt_dir / "best.pt")
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                if logger:
                    logger.info("early stopping at epoch %d", epoch)
                break

    info = {
        "run_name": run_name,
        "regime": "end-to-end fine-tuning (ViT trained, Stage 2 cache unused)",
        "device": device_str,
        "mixed_precision": use_amp,
        "epochs_run": len(history),
        "epochs_configured": epochs,
        "num_frames": train_ds.num_frames,
        "batch_size": batch_size,
        "grad_accum_steps": accum_steps,
        "effective_batch": batch_size * accum_steps,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "backbone": backbone_info,
        "backbone_spec": spec.as_dict(),
        "model": model.describe(),
        "loss": loss_info,
        "landmark_scaler": scaler_info,
        "s1_key": s1_key,
        "clips": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "label_counts": {
            "train": {str(k): v for k, v in train_ds.label_counts().items()},
            "val": {str(k): v for k, v in val_ds.label_counts().items()},
            "test": {str(k): v for k, v in test_ds.label_counts().items()},
        },
        "subjects": {
            "train": sorted(set(train_ds.subjects())),
            "val": sorted(set(val_ds.subjects())),
            "test": sorted(set(test_ds.subjects())),
        },
        "learnable_mrs_weights_final": (
            {c: round(float(v), 6) for c, v in
             zip(COMPONENTS, model.learnable_mrs.weights().detach().cpu().numpy())}
            if model.learnable_mrs is not None else None),
        "best": best,
        "selection_metric": "val_macro_f1",
        "history_csv": str(history_path),
        "checkpoint": str(ckpt_dir / "best.pt"),
        "checkpoint_dir": str(ckpt_dir),
        "environment": environment_record(),
    }
    save_json(info, Path(cfg["paths"]["artifacts_dir"]) / f"training_{run_name}.json")

    return FineTuneResult(best_epoch=best["epoch"], best_val_macro_f1=best["macro_f1"],
                          best_checkpoint=str(ckpt_dir / "best.pt"),
                          history=history, info=info)
