"""Phase 6 -- evaluations and ablation studies.

The PDF asks for the contribution of four things to be measured. Each maps to
one config switch, so every row below is the full model minus exactly one
component -- which is what makes the deltas attributable:

    Motion Reliability Score   vision_mamba.guide_delta / guide_pooling = False,
                               plus a decomposition into each mechanism alone
    Vision Mamba               fusion.use_mamba = False
    Landmark features          fusion.use_geometric = False
    Adaptive Feature Fusion    fusion.kind = 'concat'

Every variant shares one seed, one data cache and one training schedule, so the
only thing that differs between rows is the ablated component.

TWO THINGS THE FIRST VERSION OF THIS FILE GOT WRONG, RECORDED SO THEY ARE NOT
REINTRODUCED
------------------------------------------------------------------------------
1. ``data.use_mrs_gate = False`` (setting every frame to MRS = 1.0) was included
   as a separate row from ``no_mrs_guidance``. They are **provably the same
   experiment**: the dt map is ``scale = floor + (1-floor)*mrs``, so mrs = 1
   gives scale = 1, i.e. no modulation, and MRS-weighted pooling with mrs = 1 is
   a uniform mean. Both variants therefore reduce to "MRS influences nothing",
   and on a shared seed they produced byte-identical metrics. The redundant row
   is gone; ``use_mrs_gate`` remains in the config because it is still the right
   switch for building an MRS-free data cache.

2. Neither variant ablates the **frame gate**. By the time this code runs,
   Phase 1 has already discarded sub-threshold frames, so the gate cannot be
   undone from here -- testing it requires re-running Phase 1 with
   ``--mrs-threshold 0``. On the DAiSEE sample that experiment is vacuous
   anyway: the gate rejected 1 frame out of 5,403, so there is nothing for it to
   have changed. The gate becomes measurable only on data with genuinely bad
   frames.

Usage:
    python -m mrgvm.ablation --output-root outputs --config mrgvm/configs/default.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import format_confusion_matrix  # noqa: E402

from .config import MRGVMConfig, config_to_dict, load_mrgvm_config  # noqa: E402
from .data import build_dataloaders  # noqa: E402
from .train_mrgvm import summarise_class_distribution, train  # noqa: E402

logger = logging.getLogger("mrgvm.ablation")


# name -> (human description, nested config override)
ABLATIONS: Dict[str, Dict[str, object]] = {
    "full": {
        "description": "Full MRG-VM: MRS guidance + Vision Mamba + landmarks + adaptive fusion",
        "override": {},
    },
    "no_mrs_guidance": {
        "description": "MRS no longer guides dt or pooling (score removed from the model)",
        "override": {"vision_mamba": {"guide_delta": False, "guide_pooling": False}},
    },
    "guide_delta_only": {
        "description": "MRS guides dt but pooling is unweighted",
        "override": {"vision_mamba": {"guide_delta": True, "guide_pooling": False}},
    },
    "guide_pooling_only": {
        "description": "MRS weights pooling but dt is unmodulated",
        "override": {"vision_mamba": {"guide_delta": False, "guide_pooling": True}},
    },
    "no_vision_mamba": {
        "description": "Vision Mamba branch removed (landmark geometry only)",
        "override": {"fusion": {"use_mamba": False}},
    },
    "no_landmarks": {
        "description": "Landmark geometric branch removed (Vision Mamba only)",
        "override": {"fusion": {"use_geometric": False}},
    },
    "concat_fusion": {
        "description": "Adaptive gated fusion replaced by plain concatenation",
        "override": {"fusion": {"kind": "concat"}},
    },
}


def _merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def run_ablation(
    output_root: Path,
    cfg: MRGVMConfig,
    device: torch.device,
    variants: Sequence[str],
    epochs: Optional[int] = None,
) -> pd.DataFrame:
    results_dir = output_root / "results_mrgvm"
    results_dir.mkdir(parents=True, exist_ok=True)
    base = config_to_dict(cfg)

    # Two data caches at most: MRS-on and MRS-off. Everything else is a model
    # change, so the decoded frames are shared across variants.
    caches: Dict[str, tuple] = {}

    def get_cache(variant_cfg: MRGVMConfig):
        # Keyed on everything that can change what gets loaded, so a variant can
        # never silently inherit another variant's data.
        key = json.dumps(
            {
                "data": config_to_dict(variant_cfg.data),
                "image_size": variant_cfg.vision_mamba.image_size,
                "splits": list(variant_cfg.splits),
            },
            sort_keys=True, default=str,
        )
        if key not in caches:
            logger.info("Building data cache (use_mrs_gate=%s)...", variant_cfg.data.use_mrs_gate)
            caches[key] = build_dataloaders(
                output_root, variant_cfg.data, variant_cfg.splits,
                variant_cfg.vision_mamba.image_size,
            )
        return caches[key]

    rows: List[Dict[str, object]] = []
    bundle: Dict[str, object] = {"variants": {}}

    for name in variants:
        if name not in ABLATIONS:
            logger.error("Unknown ablation %r; skipping", name)
            continue
        spec = ABLATIONS[name]
        merged = _merge(base, spec["override"])
        if epochs is not None:
            merged["train"]["epochs"] = epochs
        variant_cfg = load_mrgvm_config(None, **merged)

        logger.info("=" * 78)
        logger.info("ABLATION: %-18s %s", name, spec["description"])
        started = time.perf_counter()
        prebuilt = get_cache(variant_cfg)

        result = train(
            output_root, variant_cfg, device, run_name=f"ablation_{name}",
            save_checkpoint=(name == "full"), prebuilt=prebuilt,
        )
        datasets = result.pop("_datasets")
        result.pop("_model")
        result.pop("_info")
        result["description"] = spec["description"]
        result["override"] = spec["override"]
        result["seconds"] = round(time.perf_counter() - started, 1)
        bundle["variants"][name] = result

        for split, metrics in result["splits"].items():
            logger.info("  %-11s macro-F1 %.3f | acc %.3f | QWK %.3f",
                        split, metrics["macro_f1"], metrics["accuracy"],
                        metrics["quadratic_weighted_kappa"])
            rows.append({
                "variant": name,
                "description": spec["description"],
                "split": split,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
                "mean_absolute_error": metrics["mean_absolute_error"],
                "n_parameters": result["n_parameters"],
                "best_epoch": result["best_epoch"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            })

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "ablation_results.csv", index=False)

    # ---- contribution deltas, computed against the full model ------------- #
    test = table[table["split"] == "Test"].set_index("variant")
    if "full" in test.index:
        baseline = float(test.loc["full", "macro_f1"])
        deltas = []
        for name in test.index:
            if name == "full":
                continue
            deltas.append({
                "removed_component": ABLATIONS[name]["description"],
                "variant": name,
                "test_macro_f1": float(test.loc[name, "macro_f1"]),
                "delta_vs_full": float(test.loc[name, "macro_f1"]) - baseline,
            })
        bundle["contribution_deltas"] = deltas
        bundle["full_test_macro_f1"] = baseline

    (results_dir / "ablation_results.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for every variant.")
    parser.add_argument("--variants", nargs="+", default=list(ABLATIONS),
                        help=f"Subset of: {list(ABLATIONS)}")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    results_dir = output_root / "results_mrgvm"
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(results_dir / "ablation.log", mode="w", encoding="utf-8"),
        ],
    )

    cfg = load_mrgvm_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s | variants: %s", device, args.variants)

    table = run_ablation(output_root, cfg, device, args.variants, args.epochs)

    logger.info("=" * 78)
    logger.info("PHASE 6 ABLATION SUMMARY (Test split, macro-F1 is the headline)")
    test = table[table["split"] == "Test"].sort_values("macro_f1", ascending=False)
    logger.info("  %-18s %9s %9s %9s %10s", "variant", "macro-F1", "accuracy", "QWK", "params")
    for row in test.to_dict(orient="records"):
        logger.info("  %-18s %9.3f %9.3f %9.3f %10d",
                    row["variant"], row["macro_f1"], row["accuracy"],
                    row["quadratic_weighted_kappa"], row["n_parameters"])

    if "full" in set(test["variant"]):
        baseline = float(test[test["variant"] == "full"]["macro_f1"].iloc[0])
        logger.info("-" * 78)
        logger.info("  Contribution of each removed component (negative = removing it HURT):")
        for row in test.to_dict(orient="records"):
            if row["variant"] == "full":
                continue
            logger.info("    %-18s delta macro-F1 %+.3f", row["variant"], row["macro_f1"] - baseline)

    logger.info("Results -> %s", results_dir / "ablation_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
