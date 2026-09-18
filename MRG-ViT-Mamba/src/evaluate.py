"""Final evaluation and efficiency measurement.

This is the only module that reads the test split. It loads a checkpoint that was
selected on validation macro-F1, runs it once, and writes the metrics. Nothing
here tunes anything (Rule 7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CachedClipDataset, collate, load_calibration
from .mamba_ref import mamba_backend_info
from .metrics import DEFAULT_CLASS_NAMES, compute_metrics, format_report, plot_confusion_matrix
from .model import MRGViTMamba
from .utils import ensure_dir, environment_record, get_device, resolve_path, save_json


def load_checkpoint(cfg, path, device) -> tuple[MRGViTMamba, dict]:
    """Rebuild a model from a checkpoint using the config it was TRAINED with.

    The architecture must come from ``ckpt["config"]``, not from whatever config
    the caller happens to be holding. An ablation checkpoint trained with mean
    pooling and no MRS weighting has a state_dict that loads cleanly into a model
    built with reliability-weighted pooling -- the parameter shapes are identical
    -- and would then be evaluated with a mechanism it was never trained with.
    Nothing would raise; the numbers would just be wrong.
    """
    ckpt = torch.load(resolve_path(path), map_location=device, weights_only=False)
    model_cfg = ckpt.get("config") or cfg
    model = MRGViTMamba(vit_dim=int(ckpt["vit_dim"]),
                        landmark_dim=int(ckpt["landmark_dim"]),
                        cfg=model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    y_true, y_pred, logits_all, clip_ids, subjects = [], [], [], [], []
    for batch in loader:
        feats = batch["vit_features"].to(device)
        mrs = batch["mrs"].to(device)
        comps = batch["mrs_components"].to(device)
        land = batch["landmark_features"].to(device)
        logits = model(feats, mrs, land, mrs_components=comps)
        logits_all.append(logits.float().cpu().numpy())
        y_pred.append(logits.float().argmax(dim=-1).cpu().numpy())
        y_true.append(batch["label"].numpy())
        clip_ids += batch["clip_id"]
        subjects += batch["subject_id"]
    return (np.concatenate(y_true), np.concatenate(y_pred),
            np.concatenate(logits_all), clip_ids, subjects)


@torch.no_grad()
def measure_efficiency(model, dataset, device, repeats: int = 20, warmup: int = 5,
                       batch_size: int = 1) -> dict:
    """Timing with the conditions stated, as spec section 31 requires.

    Note this times the *cached-feature* path (Mamba + fusion + MLP). The frozen
    ViT and the MediaPipe preprocessing are timed separately and reported
    alongside, because quoting one number for 'inference' when most of the cost
    sits in preprocessing would be misleading.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    batches = [b for _, b in zip(range(max(repeats, warmup) + warmup), loader)]
    if not batches:
        return {"error": "no batches available"}

    def run_once(batch):
        feats = batch["vit_features"].to(device)
        mrs = batch["mrs"].to(device)
        comps = batch["mrs_components"].to(device)
        land = batch["landmark_features"].to(device)
        model(feats, mrs, land, mrs_components=comps)

    for i in range(warmup):
        run_once(batches[i % len(batches)])
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    import time

    times = []
    for i in range(repeats):
        batch = batches[i % len(batches)]
        t0 = time.perf_counter()
        run_once(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    per_video = times / batch_size
    return {
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "device": str(device),
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "repetitions": repeats,
        "scope": "cached-feature head only (MRS weighting + Mamba + fusion + MLP); "
                 "excludes video decode, MediaPipe and the frozen ViT",
        "seconds_per_video_mean": float(per_video.mean()),
        "seconds_per_video_std": float(per_video.std()),
        "seconds_per_video_median": float(np.median(per_video)),
        "videos_per_second": float(1.0 / per_video.mean()) if per_video.mean() > 0 else None,
        "peak_gpu_memory_mb": (float(torch.cuda.max_memory_allocated() / 1024**2)
                               if device.type == "cuda" else None),
    }


def evaluate_split(cfg, index, s1_key: str, s2_key: str, checkpoint: str,
                   split: str = "test", run_name: str = "main",
                   mrs_mode: str | None = None, logger=None,
                   artifact_prefix: str | None = None,
                   plots: bool = True, per_clip: bool = True) -> dict:
    dev_info = get_device()
    device = torch.device(dev_info.device)
    model, ckpt = load_checkpoint(cfg, checkpoint, device)

    calibration = load_calibration(cfg)
    # Reliability switch follows the checkpoint unless explicitly overridden, so a
    # model trained without MRS is never evaluated with it (or vice versa).
    dataset = CachedClipDataset(cfg, index, split, s1_key, s2_key, calibration,
                                mrs_mode=mrs_mode or ckpt.get("mrs_mode") or "real")
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]),
                        shuffle=False, collate_fn=collate)

    y_true, y_pred, logits, clip_ids, subjects = predict(model, loader, device)
    num_classes = int(cfg["dataset"]["num_classes"])
    metrics = compute_metrics(y_true, y_pred, num_classes)

    eff = measure_efficiency(model, dataset, device,
                             repeats=int(cfg["evaluation"]["timing_repeats"]),
                             warmup=int(cfg["evaluation"]["timing_warmup"]))

    prefix = artifact_prefix or f"{run_name}_{split}"
    art = ensure_dir(cfg["paths"]["artifacts_dir"])

    report_text = format_report(y_true, y_pred, num_classes)
    (art / f"classification_report_{prefix}.txt").write_text(
        f"Split: {split}\nCheckpoint: {checkpoint}\n"
        f"Selected on: validation macro-F1 (epoch {ckpt.get('epoch')})\n"
        f"Clips: {len(y_true)}   Unique subjects: {len(set(subjects))}\n"
        f"MRS weighting: {dataset.mrs_mode}\n"
        f"Mamba backend: {mamba_backend_info()['backend']}\n\n{report_text}\n",
        encoding="utf-8")

    # Sweeps (ablation, robustness, sampling) call this dozens of times; writing a
    # pair of near-identical plots per run buries the real artifacts. The numbers
    # are always in the JSON, so only the plots are optional.
    if plots:
        plot_confusion_matrix(metrics["confusion_matrix"], DEFAULT_CLASS_NAMES,
                              art / f"confusion_matrix_{prefix}.png",
                              title=f"{split} confusion matrix ({len(y_true)} clips)")
        plot_confusion_matrix(metrics["confusion_matrix"], DEFAULT_CLASS_NAMES,
                              art / f"confusion_matrix_{prefix}_normalised.png",
                              title=f"{split} confusion matrix, row-normalised",
                              normalize=True)

    result = {
        "split": split,
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "selection_metric": "val_macro_f1",
        "dataset_mrs_mode": dataset.mrs_mode,
        "model_mrs_weighting_mode": model.mrs_weighting.mode,
        "model_pooling": model.temporal.pooling,
        "num_videos": int(len(y_true)),
        "num_unique_subjects": len(set(subjects)),
        "subjects": sorted(set(subjects)),
        "label_counts": {str(k): int(v) for k, v in sorted(dataset.label_counts().items())},
        "metrics": metrics,
        "efficiency": eff,
        "parameters": model.parameter_counts(),
        "model": model.describe(),
        "num_sampled_frames": dataset.num_frames,
        "vit_backbone": str(cfg["vit"]["model_name"]),
        "vit_input_resolution": None,
        "cache_keys": {"stage1": s1_key, "stage2": s2_key},
        "calibration": calibration.to_dict(),
        # Per-clip rows pair a DAiSEE clip id with its ground-truth engagement
        # label, i.e. they redistribute dataset labels. They are written to a
        # separate, git-ignored file so the committed metrics stay label-free at
        # the per-clip level while error analysis is still possible locally.
        "per_clip_predictions_file": None,
        "environment": environment_record(),
    }
    if per_clip:
        pc_path = art / f"per_clip_predictions_{prefix}.json"
        save_json({"split": split, "checkpoint": str(checkpoint),
                   "note": "Contains DAiSEE ground-truth labels per clip; git-ignored.",
                   "rows": [{"clip_id": c, "subject_id": s, "true": int(t), "pred": int(p)}
                            for c, s, t, p in zip(clip_ids, subjects, y_true, y_pred)]},
                  pc_path)
        result["per_clip_predictions_file"] = str(pc_path.name)

    save_json(result, art / f"{'test_metrics' if split == 'test' else split + '_metrics'}"
                            f"_{prefix}.json")

    if logger:
        logger.info("%s: acc=%.4f macroF1=%.4f weightedF1=%.4f (n=%d, %d subjects)",
                    split, metrics["accuracy"], metrics["macro_f1"],
                    metrics["weighted_f1"], len(y_true), len(set(subjects)))
    return result
