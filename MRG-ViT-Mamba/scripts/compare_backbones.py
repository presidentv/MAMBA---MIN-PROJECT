"""Empirically compare candidate ViT backbones.

The spec fixes the *architecture* (a ViT spatial encoder feeding a Mamba temporal
encoder) but not which pretrained ViT. ImageNet-supervised ViT-B/16 is the
conventional default and a weak prior for faces, so this script measures several
candidates instead of asserting one.

Protocol -- deliberately cheap and deliberately honest:

  1. Load each candidate; record its *measured* embed_dim, input size and params.
  2. Run Stage 2 only (the Stage 1 face/landmark cache is shared, so MediaPipe
     is not repeated).
  3. Fit a linear probe -- multinomial logistic regression on the MRS-weighted,
     mean-pooled clip feature -- on TRAIN, and score it on VALIDATION.

A linear probe is used rather than the full model because the question here is
"how linearly separable is engagement in this feature space", which is what a
frozen backbone contributes. It is a *ranking* signal for choosing a backbone,
not a result: the reported numbers are validation-probe scores, not the model's
performance, and the test split is never touched.

Writes artifacts/backbone_comparison.json.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.dataset_index import build_index  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.mrs import mrs_from_arrays  # noqa: E402
from src.preprocess import run_stage2, stage1_key, stage1_path, stage2_key, stage2_path  # noqa: E402
from src.utils import Timer, get_device, get_logger, load_config, save_json, set_seed  # noqa: E402
from src.vit_encoder import CANDIDATE_BACKBONES, ViTFrameEncoder  # noqa: E402


def adaptive_chunk(base: int, input_size: int) -> int:
    """Keep peak activation memory roughly constant as the input resolution grows.

    A 518px DINOv2 input has ~5x the tokens of a 224px one, so a chunk size tuned
    for 224 will run a 6 GB card out of memory.
    """
    return max(1, int(base * (224.0 / max(input_size, 1)) ** 2))


def load_split(cfg, index, split, s1_key, s2_key, calibration, cfg_mrs):
    """Return (X [N, D] MRS-weighted mean-pooled features, y [N])."""
    X, y = [], []
    for rec in index.clips.get(split, []):
        p1 = stage1_path(cfg, s1_key, split, rec.stem)
        p2 = stage2_path(cfg, s2_key, split, rec.stem)
        if not (p1.exists() and p2.exists()):
            continue
        d = np.load(p1)
        feats = np.load(p2).astype(np.float64)          # [T, D]
        mrs = mrs_from_arrays(d["blur_raw"], d["face_area_fraction"], d["detector_confidence"],
                              d["head_pose_deg"], d["mrs_eye_visibility"],
                              d["mrs_motion_consistency"], cfg_mrs, calibration)
        X.append((feats * mrs[:, None]).mean(axis=0))
        y.append(int(rec.engagement))
    return np.stack(X), np.array(y, dtype=int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="override the candidate list")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    log = get_logger("backbones", "logs/compare_backbones.log")
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    index = build_index(cfg)
    device = get_device()
    s1_key = stage1_key(cfg)
    cfg_mrs = dict(cfg["mrs"])

    from src.dataset import load_calibration

    calibration = load_calibration(cfg)

    candidates = args.backbones or list(CANDIDATE_BACKBONES)
    print("=" * 78)
    print("BACKBONE COMPARISON  (linear probe, train -> validation; test never used)")
    print("=" * 78)
    print(f"device        : {device.device} ({device.name})")
    print(f"stage-1 cache : {s1_key}")
    print(f"calibrated MRS: {calibration.calibrated}")
    print(f"candidates    : {len(candidates)}\n")

    results = []
    for name in candidates:
        entry: dict = {"backbone": name, "rationale": CANDIDATE_BACKBONES.get(name)}
        print("-" * 78)
        print(f"{name}")
        try:
            with Timer() as t_load:
                encoder = ViTFrameEncoder(
                    model_name=name, pretrained=True, freeze=True,
                    chunk_size=int(cfg["vit"]["batch_size"])).to(device.device)
            spec = encoder.spec
            encoder.chunk_size = adaptive_chunk(int(cfg["vit"]["batch_size"]), spec.input_size)
            entry |= {"loaded": True, "load_seconds": round(t_load.elapsed, 1),
                      **spec.as_dict(), "chunk_size": encoder.chunk_size}
            print(f"  embed_dim={spec.embed_dim}  input={spec.input_size}  "
                  f"params={spec.num_params:,}  chunk={encoder.chunk_size}")

            s2_key = stage2_key(cfg, s1_key, spec.name, spec.input_size)
            with Timer() as t_feat:
                summary = run_stage2(cfg, index, encoder, s1_key, device=device.device, logger=None)
            entry["stage2"] = {"cache_key": summary["cache_key"],
                               "counts": summary["counts"],
                               "seconds": round(t_feat.elapsed, 1)}
            if summary["counts"]["failed"]:
                entry["stage2_failures"] = summary["failures"][:5]
            print(f"  stage2: processed={summary['counts']['processed']} "
                  f"cached={summary['counts']['cached']} failed={summary['counts']['failed']} "
                  f"in {t_feat.elapsed:.1f}s")

            Xtr, ytr = load_split(cfg, index, "train", s1_key, s2_key, calibration, cfg_mrs)
            Xva, yva = load_split(cfg, index, "val", s1_key, s2_key, calibration, cfg_mrs)

            scaler = StandardScaler().fit(Xtr)     # train statistics only
            probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0,
                                       random_state=int(cfg["seed"]))
            probe.fit(scaler.transform(Xtr), ytr)
            pred_tr = probe.predict(scaler.transform(Xtr))
            pred_va = probe.predict(scaler.transform(Xva))

            m_tr = compute_metrics(ytr, pred_tr, int(cfg["dataset"]["num_classes"]))
            m_va = compute_metrics(yva, pred_va, int(cfg["dataset"]["num_classes"]))
            entry["probe"] = {
                "train": {k: m_tr[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
                "val": {k: m_va[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
                "val_confusion_matrix": m_va["confusion_matrix"],
                "n_train": int(len(ytr)), "n_val": int(len(yva)),
            }
            print(f"  probe  train: acc={m_tr['accuracy']:.3f} macroF1={m_tr['macro_f1']:.3f}")
            print(f"  probe  VAL  : acc={m_va['accuracy']:.3f} macroF1={m_va['macro_f1']:.3f} "
                  f"weightedF1={m_va['weighted_f1']:.3f}")

            del encoder
            import torch
            if device.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            entry |= {"loaded": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(limit=5)}
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        results.append(entry)

    ok = [r for r in results if r.get("probe")]
    ok.sort(key=lambda r: r["probe"]["val"]["macro_f1"], reverse=True)

    print("\n" + "=" * 78)
    print("RANKING BY VALIDATION MACRO-F1 (linear probe)")
    print("=" * 78)
    print(f"{'backbone':<48}{'D':>6}{'val acc':>10}{'val mF1':>10}")
    for r in ok:
        print(f"{r['backbone'][:47]:<48}{r['embed_dim']:>6}"
              f"{r['probe']['val']['accuracy']:>10.3f}{r['probe']['val']['macro_f1']:>10.3f}")
    failed = [r for r in results if not r.get("loaded")]
    if failed:
        print(f"\n{len(failed)} candidate(s) could not be evaluated:")
        for r in failed:
            print(f"  {r['backbone']}: {r['error']}")

    save_json({
        "protocol": "Linear probe (multinomial logistic regression, class_weight=balanced) on "
                    "MRS-weighted mean-pooled frozen ViT features. Scaler and probe fitted on "
                    "TRAIN only; scored on VALIDATION. The test split is not read by this "
                    "script. These are probe scores for backbone selection, NOT model results.",
        "device": device.as_dict(),
        "stage1_key": s1_key,
        "calibration": calibration.to_dict(),
        "num_candidates": len(candidates),
        "results": results,
        "ranking_by_val_macro_f1": [
            {"backbone": r["backbone"], "embed_dim": r["embed_dim"],
             "val_macro_f1": r["probe"]["val"]["macro_f1"],
             "val_accuracy": r["probe"]["val"]["accuracy"]} for r in ok
        ],
    }, "artifacts/backbone_comparison.json")
    print("\nwrote artifacts/backbone_comparison.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
