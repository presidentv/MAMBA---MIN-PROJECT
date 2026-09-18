"""Run Stage A preprocessing.

    Stage 1  video -> face crops + landmarks + raw MRS   (CPU, backbone-agnostic)
    Stage 2  cached crops -> ViT features                (GPU, per backbone)

Usage:
    python scripts/run_preprocessing.py --stage all
    python scripts/run_preprocessing.py --stage 2 --backbone timm:vit_base_patch16_clip_224.openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_index import build_index  # noqa: E402
from src.preprocess import run_stage1, run_stage2, stage1_key  # noqa: E402
from src.utils import Timer, get_device, get_logger, load_config, save_json, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    ap.add_argument("--backbone", default=None, help="override cfg.vit.model_name")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="clips per split (smoke tests)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("preprocess", "logs/run_preprocessing.log")
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    index = build_index(cfg)
    if not index.is_usable():
        log.error("dataset index has fatal problems; run scripts/audit_dataset.py")
        return 1

    device = get_device()
    log.info("device: %s (%s)", device.device, device.name)

    s1_key = stage1_key(cfg)
    results = {}

    if args.stage in ("1", "all"):
        log.info("Stage 1 -> cache key %s", s1_key)
        with Timer() as t:
            s1 = run_stage1(cfg, index, splits=args.splits, limit=args.limit,
                            force=args.force, logger=log)
        s1["elapsed_seconds"] = round(t.elapsed, 2)
        results["stage1"] = s1
        c = s1["counts"]
        log.info("Stage 1 done in %.1fs: processed=%d cached=%d failed=%d (%.2f%% failure)",
                 t.elapsed, c["processed"], c["cached"], c["failed"], s1["failure_percentage"])
        for f in s1["failures"][:10]:
            log.error("  %s", f)

    if args.stage in ("2", "all"):
        from src.vit_encoder import ViTFrameEncoder

        backbone = args.backbone or str(cfg["vit"]["model_name"])
        log.info("Stage 2 backbone: %s", backbone)
        encoder = ViTFrameEncoder(model_name=backbone,
                                  pretrained=bool(cfg["vit"]["pretrained"]),
                                  freeze=True,
                                  chunk_size=int(cfg["vit"]["batch_size"])).to(device.device)
        log.info("  embed_dim=%d  input_size=%d  params=%s",
                 encoder.spec.embed_dim, encoder.spec.input_size,
                 f"{encoder.spec.num_params:,}")
        with Timer() as t:
            s2 = run_stage2(cfg, index, encoder, s1_key, splits=args.splits,
                            limit=args.limit, force=args.force,
                            device=device.device, logger=log)
        s2["elapsed_seconds"] = round(t.elapsed, 2)
        results["stage2"] = s2
        c = s2["counts"]
        log.info("Stage 2 done in %.1fs: processed=%d cached=%d failed=%d missing_stage1=%d",
                 t.elapsed, c["processed"], c["cached"], c["failed"], c["missing_stage1"])
        for f in s2["failures"][:10]:
            log.error("  %s", f)

    save_json({"device": device.as_dict(), **results}, "artifacts/preprocessing_summary.json")
    log.info("wrote artifacts/preprocessing_summary.json")

    failed = sum(r["counts"]["failed"] for r in results.values())
    if failed:
        log.error("%d clips failed; see the manifest for video_id/stage/error", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
