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
from src.utils import (  # noqa: E402
    Timer, get_device, get_logger, load_config, missing_clip_policy, save_json, set_seed,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    ap.add_argument("--backbone", default=None, help="override cfg.vit.model_name")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="clips per split (smoke tests)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="Stage 1 only: process every N-th clip starting at I, so N "
                         "processes can split the corpus, e.g. --shard 0/8 ... --shard 7/8")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    shard = None
    if args.shard:
        try:
            i_s, n_s = (int(x) for x in args.shard.split("/"))
        except ValueError:
            ap.error(f"--shard must look like I/N, got {args.shard!r}")
        if not (n_s >= 1 and 0 <= i_s < n_s):
            ap.error(f"--shard {args.shard}: need 0 <= I < N")
        if args.stage != "1":
            ap.error("--shard applies to Stage 1 only; run Stage 2 once, unsharded")
        shard = (i_s, n_s)

    log_name = (f"logs/run_preprocessing_shard{shard[0]}of{shard[1]}.log" if shard
                else "logs/run_preprocessing.log")
    log = get_logger("preprocess", log_name)
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
                            force=args.force, logger=log, shard=shard)
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

    summary_path = (f"artifacts/preprocessing_summary_shard{shard[0]}of{shard[1]}.json"
                    if shard else "artifacts/preprocessing_summary.json")
    save_json({"device": device.as_dict(), **results}, summary_path)
    log.info("wrote %s", summary_path)

    allow_missing, max_frac = missing_clip_policy(cfg)
    exit_code = 0
    for stage, r in results.items():
        failed, total = r["counts"]["failed"], r["counts"]["total"]
        if not failed:
            continue
        frac = failed / total if total else 1.0
        if allow_missing and frac <= max_frac:
            log.warning("%s: %d/%d clips failed (%.2f%%) - within the %.2f%% tolerance of "
                        "missing_clip_policy=skip; they will be left out of training",
                        stage, failed, total, 100 * frac, 100 * max_frac)
        else:
            log.error("%s: %d/%d clips failed (%.2f%%); see the manifest for "
                      "video_id/stage/error", stage, failed, total, 100 * frac)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
