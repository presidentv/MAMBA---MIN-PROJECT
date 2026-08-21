"""Phases 3 and 4 deliverables -- export the learned feature representations.

Phase 3's deliverable is "deep behavioural feature embeddings" and Phase 4's is
the "optimized feature representation". Both are produced by the trained MRG-VM
backbone, so this script loads a checkpoint and writes, per clip:

    mamba_embedding    (D,)   Phase 3: reliability-guided Vision Mamba embedding
    geometric_vector   (4*Dg,) pooled landmark geometry (mean/std/min/max)
    fused              (F,)   Phase 4: the adaptive-fusion output
    gate               (H,)   per-channel mixing weights, for interpretation

An untrained backbone emits noise, so this must run *after* train_mrgvm.py.

Output:
    outputs/embeddings/mrgvm_embeddings.parquet   one row per clip
    outputs/embeddings/manifest.json

Usage:
    python -m mrgvm.extract_embeddings --output-root outputs \
        --checkpoint outputs/checkpoints/mrgvm.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .config import load_mrgvm_config  # noqa: E402
from .data import build_dataloaders  # noqa: E402
from .model import MRGVMModel  # noqa: E402

logger = logging.getLogger("mrgvm.embed")


def extract(
    output_root: Path, checkpoint_path: Path, device: torch.device
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = load_mrgvm_config(None, **checkpoint["config"])
    logger.info("Loaded checkpoint from epoch %s (val macro-F1 %.3f)",
                checkpoint.get("epoch"), checkpoint.get("val_macro_f1", float("nan")))

    loaders, datasets, info = build_dataloaders(
        output_root, cfg.data, cfg.splits, cfg.vision_mamba.image_size
    )
    if info["geometric_dim"] != checkpoint["geometric_dim"]:
        raise SystemExit(
            f"Geometric dim mismatch: checkpoint {checkpoint['geometric_dim']} vs "
            f"data {info['geometric_dim']}. Re-run Phases 1-2 features or retrain."
        )

    model = MRGVMModel(cfg, info["geometric_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for split, loader in loaders.items():
            for batch in loader:
                output = model.embed(
                    batch["frames"].to(device), batch["geometric"].to(device),
                    batch["mrs"].to(device), batch["mask"].to(device),
                )
                size = len(batch["clip_id"])
                for i in range(size):
                    row: Dict[str, object] = {
                        "ClipID": batch["clip_id"][i],
                        "SubjectID": batch["subject_id"][i],
                        "split": batch["split"][i],
                        "label": int(batch["label"][i].item()),
                        "n_frames": int(batch["mask"][i].sum().item()),
                        "mean_mrs": float(
                            batch["mrs"][i][batch["mask"][i]].mean().item()
                        ),
                    }
                    for key in ("mamba_embedding", "geometric_vector", "fused", "gate"):
                        tensor = output.get(key)
                        if tensor is None:
                            continue
                        values = tensor[i].cpu().numpy()
                        prefix = {"mamba_embedding": "mamba", "geometric_vector": "geo",
                                  "fused": "fused", "gate": "gate"}[key]
                        for j, value in enumerate(values):
                            row[f"{prefix}_{j:04d}"] = float(value)
                    rows.append(row)

    frame = pd.DataFrame(rows)
    target_dir = output_root / "embeddings"
    target_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_dir / "mrgvm_embeddings.parquet", index=False)

    counts = {
        prefix: len([c for c in frame.columns if c.startswith(prefix + "_")])
        for prefix in ("mamba", "geo", "fused", "gate")
    }
    manifest = {
        "checkpoint": str(checkpoint_path),
        "n_clips": len(frame),
        "dimensions": counts,
        "geometric_columns": checkpoint.get("geometric_columns", []),
        "phase3_deliverable": "mamba_* columns (deep behavioural feature embeddings)",
        "phase4_deliverable": "fused_* columns (optimized feature representation)",
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %d clip embeddings -> %s | dims %s",
                len(frame), target_dir / "mrgvm_embeddings.parquet", counts)
    return frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%H:%M:%S",
    )
    output_root = Path(args.output_root)
    checkpoint = args.checkpoint or (output_root / "checkpoints" / "mrgvm.pt")
    if not Path(checkpoint).is_file():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}. Run mrgvm.train_mrgvm first."
        )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    extract(output_root, Path(checkpoint), device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
