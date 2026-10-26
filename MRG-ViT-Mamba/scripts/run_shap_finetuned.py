"""SHAP attribution for a FINE-TUNED checkpoint.

scripts/run_shap.py reads the Stage 2 feature cache, which holds the output of
a *frozen* ViT. The moment the backbone is fine-tuned those vectors describe a
model that no longer exists, so running that script against a fine-tuned
checkpoint would either fail (no such cache) or, if a stale cache happened to
be present, produce attributions for the wrong model -- numbers that look
entirely reasonable and are wrong.

This script closes that gap: it rebuilds the encoder from the checkpoint, runs
the Stage 1 face crops through the actual trained backbone, and feeds the
resulting features into the same attribution code in src/explain.py. Nothing
about the attribution maths differs; only the provenance of the features does.

What it answers: across many clips, how much of the prediction comes from the
ViT -> Mamba branch versus the named landmark features, and which landmark
features matter most. The per-clip worked examples in explain_finetuned.py
answer the same question for three clips; this is the population view.

Run:
    python scripts/run_shap_finetuned.py --config configs/config_binary_full.yaml \\
        --run fullbin --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.class_merge import class_names as merged_class_names  # noqa: E402
from src.dataset import collate_finetune  # noqa: E402
from src.dataset_index import build_index  # noqa: E402
from src.explain import run_shap  # noqa: E402
from src.finetune import build_finetune_datasets, fit_landmark_scaler  # noqa: E402
from src.model import MRGViTMamba  # noqa: E402
from src.preprocess import stage1_key  # noqa: E402
from src.utils import (  # noqa: E402
    Config, get_device, get_logger, load_config, resolve_path, set_seed,
)
from src.vit_encoder import ViTFrameEncoder  # noqa: E402


class FineTunedFeatureLoader:
    """Yields collate()-shaped batches whose features come from the trained ViT.

    ``collect_inputs`` in src/explain.py consumes batches containing
    ``vit_features``; the cached-feature path gets those from Stage 2. Here they
    are computed on the fly so the attribution describes the model that was
    actually evaluated.

    ``limit`` caps how many clips are encoded. SHAP samples its background and
    evaluation sets from what it is given, so encoding the entire training split
    to then sample 128 of it would waste minutes of GPU for no change in the
    result. The cap is recorded in the output.
    """

    def __init__(self, dataset, encoder, device, batch_size=8, workers=0, limit=None):
        self.dataset = dataset
        self.encoder = encoder
        self.device = device
        self.limit = limit
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_finetune, num_workers=workers,
                                 pin_memory=device.type == "cuda")

    def __iter__(self):
        seen = 0
        self.encoder.eval()
        for batch in self.loader:
            if self.limit is not None and seen >= self.limit:
                return
            with torch.no_grad():
                feats = self.encoder(batch["frames"].to(self.device, non_blocking=True))
            seen += batch["label"].shape[0]
            yield {
                "vit_features": feats.detach().cpu(),
                "mrs": batch["mrs"],
                "mrs_components": batch["mrs_components"],
                "landmark_features": batch["landmark_features"],
                "label": batch["label"],
                "clip_id": batch["clip_id"],
                "subject_id": batch["subject_id"],
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="run name; checkpoint is <ckpt_dir>/<run>/best.pt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--background-clips", type=int, default=512,
                    help="clips encoded to sample the SHAP background from")
    ap.add_argument("--eval-clips", type=int, default=512,
                    help="clips encoded to sample the explained set from")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    log = get_logger("shap_ft", f"logs/shap_{args.run}.log")
    cfg = load_config(resolve_path(args.config))
    set_seed(int(cfg["seed"]))
    device = torch.device(get_device().device)

    ckpt_path = (resolve_path(args.checkpoint) if args.checkpoint else
                 resolve_path(Path(cfg["paths"]["checkpoints_dir"]) / args.run / "best.pt"))
    if not Path(ckpt_path).is_file():
        log.error("checkpoint not found: %s", ckpt_path)
        return 1

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # The checkpoint stores a plain dict, and its paths are a snapshot of
    # wherever the data sat during training. Architecture and dataset settings
    # must come from the checkpoint -- evaluating a model with a mechanism it
    # was not trained with gives numbers that look fine and are wrong -- but the
    # paths should follow the current environment, so DAISEE_ROOT and
    # MRG_CACHE_DIR still work.
    ccfg = Config(dict(ckpt["config"]))
    ccfg["paths"] = dict(cfg["paths"])
    # How many clips SHAP samples is an analysis choice, not a property of the
    # trained model, so it comes from the config passed on the command line.
    # Leaving it baked into the checkpoint would mean re-running the analysis at
    # a different sample size required retraining, and would silently ignore the
    # caller's setting.
    ccfg["explain"] = dict(cfg["explain"])
    if bool(ccfg["vit"].get("freeze", True)):
        log.warning("this checkpoint has a FROZEN backbone; scripts/run_shap.py is "
                    "the cheaper path for it (it reuses the Stage 2 cache)")

    # The encoder must carry the trained weights, which arrive with the rest of
    # the model state below -- not a freshly pretrained backbone.
    encoder = ViTFrameEncoder(model_name=ccfg["vit"]["model_name"], pretrained=False,
                              freeze=True, chunk_size=int(ccfg["vit"]["batch_size"]))
    model = MRGViTMamba(vit_dim=int(ckpt["vit_dim"]), landmark_dim=int(ckpt["landmark_dim"]),
                        cfg=ccfg, vit_encoder=encoder).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    num_classes = int(ccfg["dataset"]["num_classes"])
    names = tuple(ckpt.get("class_names") or merged_class_names(ccfg, num_classes))

    index = build_index(ccfg)
    datasets = build_finetune_datasets(ccfg, index, stage1_key(ccfg), encoder.spec,
                                       ("train", args.split))
    # The landmark branch standardiser is part of the trained state, but fitting
    # it again on the same training split is harmless and guards against a
    # checkpoint saved before it was stored.
    fit_landmark_scaler(model, datasets["train"])

    train_loader = FineTunedFeatureLoader(datasets["train"], encoder, device,
                                          workers=args.workers, limit=args.background_clips)
    eval_loader = FineTunedFeatureLoader(datasets[args.split], encoder, device,
                                         workers=args.workers, limit=args.eval_clips)

    print("=" * 78)
    print("SHAP ANALYSIS (fine-tuned checkpoint)")
    print("=" * 78)
    print(f"checkpoint : {ckpt_path} (epoch {ckpt.get('epoch')})")
    print(f"classes    : {list(names)}")
    print(f"features   : computed through the FINE-TUNED backbone, not the Stage 2 cache")
    print(f"explaining : {args.split} split, up to {args.eval_clips} clips encoded")
    print(f"background : train split, up to {args.background_clips} clips encoded")
    print()

    result = run_shap(model, train_loader, eval_loader, device, ccfg,
                      out_prefix=f"{args.run}_{args.split}", class_names=names)
    result["features_from"] = "fine-tuned backbone (recomputed, Stage 2 cache not used)"
    result["clips_encoded"] = {"background": args.background_clips, "eval": args.eval_clips}

    print(f"attribution input : {result['attribution_input']}")
    print(f"background/eval   : {result['background_samples']} / {result['evaluated_samples']}")
    print(f"\n{'group':<22}{'dims':>7}{'share of total':>17}{'per-dimension':>16}")
    for group, share in sorted(result["group_share_of_total"].items(),
                               key=lambda kv: kv[1], reverse=True):
        print(f"{group:<22}{result['group_sizes'][group]:>7}{100 * share:>16.2f}%"
              f"{result['group_share_per_dimension'][group]:>16.5f}")

    print("\ntop 15 named landmark features by mean |SHAP|:")
    for item in result["top_landmark_features"][:15]:
        print(f"  {item['feature']:<40}{item['mean_abs_shap']:.5f}")

    print(f"\nCAVEAT: {result['caveat']}")
    print(f"\nwrote artifacts/shap_analysis_{args.run}_{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
