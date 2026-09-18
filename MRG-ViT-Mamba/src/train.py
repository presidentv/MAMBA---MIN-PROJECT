"""Training loop for the cached-feature (Stage B) pipeline.

Rules this file enforces rather than merely documents:

* the test split is never loaded here (Rule 7);
* the landmark scaler and the class weights are fitted on the training split
  only, and the scaler lives in the model's buffers so it is saved with the
  checkpoint (Rule 8);
* checkpoint selection uses validation macro-F1, never test performance
  (spec section 29);
* every epoch is written to logs/training_history.csv with the full metric set.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CachedClipDataset, collate, load_calibration
from .fusion import StandardScaler
from .losses import build_loss
from .metrics import compute_metrics
from .model import MRGViTMamba
from .utils import ensure_dir, environment_record, gpu_memory_mb, resolve_path, save_json

from .mrs import COMPONENTS

HISTORY_FIELDS = [
    "epoch", "train_loss", "train_accuracy", "train_macro_f1", "train_weighted_f1",
    "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1",
    "learning_rate", "gpu_memory_mb", "seconds",
] + [f"w_{c}" for c in COMPONENTS]


@dataclass
class TrainResult:
    best_epoch: int
    best_val_macro_f1: float
    best_checkpoint: str
    history: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def build_datasets(cfg, index, s1_key: str, s2_key: str, mrs_mode: str | None = None,
                   splits=("train", "val")) -> dict[str, CachedClipDataset]:
    calibration = load_calibration(cfg)
    return {
        split: CachedClipDataset(cfg, index, split, s1_key, s2_key, calibration,
                                 mrs_mode=mrs_mode)
        for split in splits
    }


def fit_landmark_scaler(model: MRGViTMamba, train_ds: CachedClipDataset) -> dict:
    """Fit the landmark branch's StandardScaler on the training split only."""
    if model.landmark_branch is None:
        return {"fitted": False, "reason": "landmark branch disabled"}
    norm = model.landmark_branch.norm
    if not isinstance(norm, StandardScaler):
        return {"fitted": False, "reason": f"landmark norm is {type(norm).__name__}"}
    matrix = train_ds.landmark_matrix()
    norm.fit(matrix)
    return {"fitted": True, "n_train_clips": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "mean_abs_mean": float(np.abs(norm.mean.cpu().numpy()).mean()),
            "mean_scale": float(norm.scale.cpu().numpy().mean())}


def _run_epoch(model, loader, loss_fn, optimiser, device, scaler=None, grad_clip=0.0):
    training = optimiser is not None
    model.train(training)
    losses, y_true, y_pred = [], [], []

    for batch in loader:
        feats = batch["vit_features"].to(device, non_blocking=True)
        mrs = batch["mrs"].to(device, non_blocking=True)
        comps = batch["mrs_components"].to(device, non_blocking=True)
        land = batch["landmark_features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            if scaler is not None and training:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(feats, mrs, land, mrs_components=comps)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(feats, mrs, land, mrs_components=comps)
                loss = loss_fn(logits, labels)

        if training:
            optimiser.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()

        if not torch.isfinite(loss):
            raise RuntimeError("loss became NaN/Inf - stopping rather than logging a "
                               "meaningless number")
        losses.append(float(loss.item()))
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(logits.detach().float().argmax(dim=-1).cpu().numpy())

    return (float(np.mean(losses)), np.concatenate(y_true), np.concatenate(y_pred))


def train_model(cfg, index, s1_key: str, s2_key: str, run_name: str = "main",
                mrs_mode: str | None = None, logger=None,
                overrides: dict | None = None) -> TrainResult:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    tcfg = dict(cfg["training"])
    if overrides:
        tcfg.update(overrides)

    datasets = build_datasets(cfg, index, s1_key, s2_key, mrs_mode=mrs_mode,
                              splits=("train", "val"))
    train_ds, val_ds = datasets["train"], datasets["val"]

    if logger:
        logger.info("train clips=%d  val clips=%d", len(train_ds), len(val_ds))
        logger.info("vit_dim=%d  landmark_dim=%d  T=%d",
                    train_ds.vit_dim, train_ds.landmark_dim, train_ds.num_frames)
        logger.info("train label counts: %s", train_ds.label_counts())

    batch_size = int(tcfg["batch_size"])
    workers = int(tcfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=workers)

    model = MRGViTMamba(vit_dim=train_ds.vit_dim, landmark_dim=train_ds.landmark_dim,
                        cfg=cfg).to(device)
    scaler_info = fit_landmark_scaler(model, train_ds)
    # Buffers were filled on the CPU copy of the matrix; make sure they live on
    # the training device.
    model.to(device)

    loss_fn, loss_info = build_loss(tcfg, train_ds.label_counts(),
                                    int(cfg["dataset"]["num_classes"]))
    loss_fn = loss_fn.to(device)
    if logger:
        logger.info("class weights (%s): %s", loss_info["scheme"],
                    [round(w, 3) for w in loss_info["weights"]])

    # The five MRS weights get their own, larger learning rate and no weight
    # decay. Five parameters among 1.37M would otherwise barely move, and decay
    # would pull theta toward zero, i.e. silently back toward uniform 0.2.
    mrs_params, main_params = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (mrs_params if name.startswith("learnable_mrs.") else main_params).append(prm)
    groups = [{"params": main_params, "lr": float(tcfg["learning_rate"]),
               "weight_decay": float(tcfg["weight_decay"])}]
    if mrs_params:
        groups.append({"params": mrs_params,
                       "lr": float(tcfg.get("mrs_weight_lr", 0.02)),
                       "weight_decay": 0.0})
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

    best = {"macro_f1": -1.0, "epoch": -1}
    patience = int(tcfg.get("early_stopping_patience", epochs))
    since_improved = 0
    history: list[dict] = []
    num_classes = int(cfg["dataset"]["num_classes"])

    for epoch in range(epochs):
        t0 = time.perf_counter()
        if device_str == "cuda":
            torch.cuda.reset_peak_memory_stats()

        tr_loss, tr_true, tr_pred = _run_epoch(model, train_loader, loss_fn, optimiser,
                                               device, amp_scaler,
                                               float(tcfg.get("grad_clip", 0.0)))
        va_loss, va_true, va_pred = _run_epoch(model, val_loader, loss_fn, None, device)
        scheduler.step()

        m_tr = compute_metrics(tr_true, tr_pred, num_classes)
        m_va = compute_metrics(va_true, va_pred, num_classes)
        row = {
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "train_accuracy": round(m_tr["accuracy"], 6),
            "train_macro_f1": round(m_tr["macro_f1"], 6),
            "train_weighted_f1": round(m_tr["weighted_f1"], 6),
            "val_loss": round(va_loss, 6),
            "val_accuracy": round(m_va["accuracy"], 6),
            "val_macro_f1": round(m_va["macro_f1"], 6),
            "val_weighted_f1": round(m_va["weighted_f1"], 6),
            "learning_rate": optimiser.param_groups[0]["lr"],
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
                "mF1 %.3f wF1 %.3f | lr %.2e",
                epoch, tr_loss, m_tr["accuracy"], m_tr["macro_f1"],
                va_loss, m_va["accuracy"], m_va["macro_f1"], m_va["weighted_f1"],
                optimiser.param_groups[0]["lr"])

        # Selection is on validation macro-F1. Test data is not loaded in this
        # function at all, so it cannot influence this decision.
        if m_va["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m_va["macro_f1"], "epoch": epoch,
                    "accuracy": m_va["accuracy"], "weighted_f1": m_va["weighted_f1"],
                    "loss": va_loss}
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_metrics": m_va,
                "config": dict(cfg),
                "vit_dim": train_ds.vit_dim,
                "landmark_dim": train_ds.landmark_dim,
                "s1_key": s1_key, "s2_key": s2_key,
                "mrs_mode": train_ds.mrs_mode,
                "loss_info": loss_info,
                "scaler_info": scaler_info,
            }, ckpt_dir / "best.pt")
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                if logger:
                    logger.info("early stopping at epoch %d (no val macro-F1 improvement "
                                "for %d epochs)", epoch, patience)
                break

    info = {
        "run_name": run_name,
        "device": device_str,
        "mixed_precision": use_amp,
        "epochs_run": len(history),
        "epochs_configured": epochs,
        "batch_size": batch_size,
        "learning_rate": float(tcfg["learning_rate"]),
        "mrs_mode": train_ds.mrs_mode,
        "model": model.describe(),
        "loss": loss_info,
        "landmark_scaler": scaler_info,
        "train_clips": len(train_ds), "val_clips": len(val_ds),
        "train_label_counts": {str(k): v for k, v in train_ds.label_counts().items()},
        "val_label_counts": {str(k): v for k, v in val_ds.label_counts().items()},
        "train_subjects": sorted(set(train_ds.subjects())),
        "val_subjects": sorted(set(val_ds.subjects())),
        "learnable_mrs_weights_final": (
            {c: round(float(v), 6) for c, v in
             zip(COMPONENTS, model.learnable_mrs.weights().detach().cpu().numpy())}
            if model.learnable_mrs is not None else None),
        "best": best,
        "selection_metric": "val_macro_f1",
        "history_csv": str(history_path),
        "checkpoint": str(ckpt_dir / "best.pt"),
        "environment": environment_record(),
    }
    save_json(info, Path(cfg["paths"]["artifacts_dir"]) / f"training_{run_name}.json")

    return TrainResult(best_epoch=best["epoch"], best_val_macro_f1=best["macro_f1"],
                       best_checkpoint=str(ckpt_dir / "best.pt"),
                       history=history, info=info)
