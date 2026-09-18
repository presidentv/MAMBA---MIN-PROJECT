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
import os
import random
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


def _atomic_torch_save(obj, path: Path) -> None:
    """torch.save to a temporary name, then rename into place.

    A checkpoint is overwritten every epoch; dying half-way through writing it
    would otherwise destroy the only copy - for last.pt, the only way back in.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])
        except RuntimeError:
            # Resumed on a machine with a different GPU count; the run still
            # continues correctly, only bit-exact reproducibility is lost.
            pass


# Settings that must not change between a run and its resumption: resuming a
# T=32 checkpoint with a T=16 dataset, or onto a different backbone, would
# silently train a different model.
_RESUME_MUST_MATCH = (
    ("video", "num_frames"), ("vit", "model_name"), ("vit", "unfreeze_blocks"),
    ("dataset", "num_classes"), ("mamba", "d_model"), ("mamba", "n_layers"),
)


def periodic_checkpoint_name(epoch: int, every: int) -> str | None:
    """File name for the periodic checkpoint after 0-based `epoch`, or None.

    Named by the number of epochs completed, so after_epoch_010.pt is the
    model after 10 epochs -- row epoch=9 of the 0-based history CSV.
    """
    if every <= 0 or (epoch + 1) % every != 0:
        return None
    return f"after_epoch_{epoch + 1:03d}.pt"


def should_extend(val_macro_f1s: list[float], window: int, min_delta: float) -> bool:
    """Is validation macro-F1 still rising at the end of the first cycle?

    True when the best score in the last `window` epochs beats the best score
    before them by at least `min_delta`. A peak earlier in the cycle, or a
    plateau, returns False. The margin is there because the final epochs of a
    cosine cycle run at a near-zero learning rate and routinely add a sliver of
    macro-F1 that says nothing about whether more training would help.
    """
    if window <= 0 or len(val_macro_f1s) <= window:
        return False
    recent = max(val_macro_f1s[-window:])
    earlier = max(val_macro_f1s[:-window])
    return recent >= earlier + min_delta


def make_lr_lambda(epochs: int, warmup: int, scheduler: str,
                   decision_epoch: int | None, restart_factor: float):
    """Per-epoch LR multiplier.

    Without a decision epoch: linear warmup, then one cosine over all epochs.
    With one: the first cosine ends at the decision epoch, so a run that stops
    there has a fully annealed model. If training continues, a second cosine
    runs from the decision epoch to the end, restarting at `restart_factor` of
    the peak LR (a warm restart, as in SGDR).
    """
    cycle1 = decision_epoch if decision_epoch else epochs

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if scheduler != "cosine":
            return 1.0
        if epoch < cycle1:
            progress = (epoch - warmup) / max(1, cycle1 - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
        progress = (epoch - cycle1) / max(1, epochs - cycle1)
        return restart_factor * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    return lr_lambda


def finetune_model(cfg, index, run_name: str = "ft32", logger=None,
                   overrides: dict | None = None, resume: bool = False,
                   num_workers: int | None = None,
                   resume_from: str | None = None) -> FineTuneResult:
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
    workers = int(num_workers if num_workers is not None else tcfg.get("num_workers", 0))
    # Each step decodes batch_size x T JPEG crops; on the full release that is
    # ~170k decodes per epoch, which starves a fast GPU if done in the main
    # process. Workers keep it fed.
    loader_kw = {"collate_fn": collate_finetune, "num_workers": workers,
                 "pin_memory": device_str == "cuda"}
    if workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)

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

    # Optional decision point: train `extend_decision_epoch` epochs as a
    # complete cycle, then continue to `epochs` only if validation macro-F1 is
    # still rising. Ignored when --epochs makes the run no longer than that.
    decision_epoch = tcfg.get("extend_decision_epoch")
    decision_epoch = int(decision_epoch) if decision_epoch else None
    if decision_epoch is not None and not (warmup < decision_epoch < epochs):
        decision_epoch = None
    extend_window = int(tcfg.get("extend_window", 5))
    extend_min_delta = float(tcfg.get("extend_min_delta", 0.005))
    restart_factor = float(tcfg.get("restart_lr_factor", 0.5))
    extend_decision: dict | None = None

    lr_lambda = make_lr_lambda(epochs, warmup, str(tcfg.get("scheduler", "cosine")),
                               decision_epoch, restart_factor)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    use_amp = bool(tcfg.get("mixed_precision", False)) and device_str == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = ensure_dir(Path(cfg["paths"]["checkpoints_dir"]) / run_name)
    log_dir = ensure_dir(cfg["paths"]["logs_dir"])
    history_path = log_dir / f"training_history_{run_name}.csv"

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
    save_every_epoch = bool(tcfg.get("save_every_epoch", True))
    # A full model checkpoint every N epochs, alongside best.pt and last.pt.
    # save_every_epoch keeps its original meaning and takes precedence.
    save_every_n = 1 if save_every_epoch else int(tcfg.get("save_every_n_epochs", 0) or 0)
    start_epoch = 0
    resumed_from = None
    last_path = ckpt_dir / "last.pt"

    # --resume continues from last.pt (the latest epoch). --resume-from continues
    # from a chosen snapshot, e.g. checkpoints/<run>/after_epoch_020.pt.
    resume_path = Path(resume_from) if resume_from else last_path
    if resume_from and not resume_path.is_file():
        raise FileNotFoundError(f"--resume-from {resume_path}: no such checkpoint")

    if (resume or resume_from) and resume_path.is_file():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if "optimiser_state" not in state:
            raise RuntimeError(
                f"{resume_path} holds model weights only (no optimiser/scheduler state), "
                f"so training cannot continue from it. Use last.pt or an "
                f"after_epoch_*.pt snapshot.")
        saved = state["config"]
        for section, key in _RESUME_MUST_MATCH:
            a = saved.get(section, {}).get(key)
            b = cfg[section].get(key) if section in cfg else None
            if a != b:
                raise RuntimeError(
                    f"cannot resume {resume_path}: {section}.{key} was {a!r} in the "
                    f"interrupted run but is {b!r} now")
        model.load_state_dict(state["model_state"])
        optimiser.load_state_dict(state["optimiser_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if amp_scaler is not None and state.get("amp_scaler_state"):
            amp_scaler.load_state_dict(state["amp_scaler_state"])
        best = state["best"]
        extend_decision = state.get("extend_decision")
        since_improved = int(state["since_improved"])
        history = list(state["history"])
        _restore_rng(state["rng"])
        start_epoch = int(state["epoch"]) + 1
        resumed_from = int(state["epoch"])
        del state
        # Rolling back to an earlier snapshot: a best.pt written after that
        # snapshot belongs to the abandoned continuation, and keeping it would
        # let the report quote a model the resumed run never produced. Move it
        # aside and restart selection from the snapshot onward.
        best_path = ckpt_dir / "best.pt"
        if resume_from and best_path.is_file():
            best_on_disk = int(torch.load(best_path, map_location="cpu",
                                          weights_only=False)["epoch"])
            if best_on_disk > resumed_from:
                aside = ckpt_dir / f"best_abandoned_epoch_{best_on_disk:03d}.pt"
                os.replace(best_path, aside)
                best = {"macro_f1": -1.0, "epoch": -1}
                since_improved = 0
                if logger:
                    logger.warning(
                        "best.pt (epoch %d) is later than the snapshot (epoch %d); moved "
                        "to %s and restarted best-model selection from the snapshot",
                        best_on_disk, resumed_from, aside.name)
        if logger:
            logger.info("resumed from %s: epoch %d done, best so far epoch %d "
                        "(val macro-F1 %.4f)", resume_path, resumed_from,
                        best["epoch"], best["macro_f1"])
    elif resume and logger:
        logger.info("--resume given but %s does not exist; starting from epoch 0", last_path)

    # The CSV is rebuilt from the restored history, so an epoch that was
    # logged but not yet captured in last.pt when the process died is not
    # duplicated.
    with open(history_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    if start_epoch > 0 and since_improved >= patience:
        start_epoch = epochs          # the interrupted run had already early-stopped
    if extend_decision is not None and not extend_decision["extend"]:
        start_epoch = epochs          # the run had already stopped at the decision point

    for epoch in range(start_epoch, epochs):
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

        if save_every_epoch:
            _atomic_torch_save(checkpoint_payload(epoch, m_va),
                               ckpt_dir / f"epoch_{epoch:03d}.pt")

        if m_va["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m_va["macro_f1"], "epoch": epoch,
                    "accuracy": m_va["accuracy"], "weighted_f1": m_va["weighted_f1"],
                    "loss": va_loss}
            _atomic_torch_save(checkpoint_payload(epoch, m_va), ckpt_dir / "best.pt")
            since_improved = 0
        else:
            since_improved += 1

        if decision_epoch is not None and epoch == decision_epoch - 1:
            f1s = [float(r["val_macro_f1"]) for r in history]
            extend = should_extend(f1s, extend_window, extend_min_delta)
            extend_decision = {
                "after_epochs": decision_epoch,
                "extend": extend,
                "best_last_window": round(max(f1s[-extend_window:]), 6),
                "best_before_window": round(max(f1s[:-extend_window]), 6),
                "window": extend_window,
                "min_delta": extend_min_delta,
            }
            if logger:
                logger.info(
                    "decision after %d epochs: best val macro-F1 in last %d epochs %.4f "
                    "vs %.4f before -> %s", decision_epoch, extend_window,
                    extend_decision["best_last_window"],
                    extend_decision["best_before_window"],
                    f"still rising, continuing to {epochs} epochs" if extend
                    else "peaked earlier, stopping here")

        # Everything needed to continue from the next epoch as if uninterrupted.
        resume_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "amp_scaler_state": amp_scaler.state_dict() if amp_scaler is not None else None,
            "best": best,
            "since_improved": since_improved,
            "extend_decision": extend_decision,
            "history": list(history),
            "rng": _rng_state(),
            "config": dict(cfg),
        }
        _atomic_torch_save(resume_state, last_path)

        # Periodic snapshot: resumable with --resume-from (it carries the same
        # state as last.pt) and loadable for evaluation like best.pt.
        periodic = periodic_checkpoint_name(epoch, 0 if save_every_epoch else save_every_n)
        if periodic is not None:
            _atomic_torch_save({**checkpoint_payload(epoch, m_va), **resume_state},
                               ckpt_dir / periodic)
            if logger:
                logger.info("saved snapshot %s", ckpt_dir / periodic)

        if since_improved >= patience:
            if logger:
                logger.info("early stopping at epoch %d (no val macro-F1 improvement for "
                            "%d epochs)", epoch, patience)
            break
        if extend_decision is not None and not extend_decision["extend"]:
            break

    info = {
        "run_name": run_name,
        "regime": "end-to-end fine-tuning (ViT trained, Stage 2 cache unused)",
        "device": device_str,
        "mixed_precision": use_amp,
        "epochs_run": len(history),
        "epochs_configured": epochs,
        "early_stopping_patience": patience,
        "extend_decision_epoch": decision_epoch,
        "extend_decision": extend_decision,
        "resumed_from_epoch": resumed_from,
        "num_workers": workers,
        "save_every_epoch": save_every_epoch,
        "save_every_n_epochs": save_every_n,
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
        # Clips left out because Stage 1 could not process them (see
        # dataset.missing_clip_policy). Clip ids only - no labels.
        "missing_clips": {"train": train_ds.missing, "val": val_ds.missing,
                          "test": test_ds.missing},
        "dataset_root": str(cfg["paths"]["dataset_root"]),
        "path_overrides": dict(cfg.get("_path_overrides", {})),
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
