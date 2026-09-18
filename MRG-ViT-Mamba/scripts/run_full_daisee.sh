#!/usr/bin/env bash
# End-to-end run on the FULL DAiSEE release: fine-tuned ViT-B/16 + Mamba, T=32.
#
#   export DAISEE_ROOT=/path/to/DAiSEE        # contains DataSet/ and Labels/
#   bash scripts/run_full_daisee.sh
#
# Safe to re-run after an interruption. Stage 1 skips clips already cached,
# and training resumes from checkpoints/$RUN/last.pt. For a completely fresh
# run, delete checkpoints/$RUN first.
#
# Optional environment:
#   RUN=full_ft32          run name (checkpoints/, logs/, artifacts/ file names)
#   STAGE1_SHARDS=<nproc>  parallel Stage 1 processes (CPU-bound MediaPipe)
#   WORKERS=4              DataLoader workers during training/evaluation
#   MRG_CACHE_DIR=...      put the ~4.3 GB Stage 1 cache on a fast local disk
#   MRG_CHECKPOINT_DIR=... put checkpoints (best.pt + last.pt, ~1.4 GB) somewhere with space
#
# On Windows use scripts/run_full_daisee.ps1, which runs the same steps.

set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="configs/config_full.yaml"
RUN="${RUN:-full_ft32}"
WORKERS="${WORKERS:-4}"
NPROC="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
STAGE1_SHARDS="${STAGE1_SHARDS:-$NPROC}"
PY="${PYTHON:-python}"

step() { printf '\n==== [%s] %s ====\n' "$(date +%H:%M:%S)" "$*"; }

if [ -z "${DAISEE_ROOT:-}" ]; then
    echo "DAISEE_ROOT is not set; falling back to paths.dataset_root in $CONFIG"
fi
mkdir -p logs artifacts

step "environment"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name}  {p.total_memory / 2**30:.1f} GiB")
else:
    raise SystemExit("No CUDA device. Fine-tuning ViT-B/16 on ~5,400 clips x 32 frames "
                     "on CPU is not practical; run this on a GPU machine.")
EOF

step "1/6  fetch MediaPipe models"
"$PY" scripts/fetch_models.py

step "2/6  audit the dataset (layout, labels, subject disjointness)"
"$PY" scripts/audit_dataset.py --config "$CONFIG"

step "3/6  Stage 1 preprocessing across $STAGE1_SHARDS shards"
pids=()
for ((i = 0; i < STAGE1_SHARDS; i++)); do
    "$PY" scripts/run_preprocessing.py --config "$CONFIG" --stage 1 \
        --shard "$i/$STAGE1_SHARDS" > "logs/stage1_shard${i}.out" 2>&1 &
    pids+=("$!")
done
# Shard exit codes are not trusted individually: the consolidating pass below
# decides against the tolerance using the whole corpus.
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "shards finished; consolidating (retries any failed clip once more)"
"$PY" scripts/run_preprocessing.py --config "$CONFIG" --stage 1

step "4/6  fit MRS calibration on the training split"
"$PY" scripts/fit_mrs_stats.py --config "$CONFIG"

step "5/6  GPU memory probe (informational)"
"$PY" scripts/probe_finetune_memory.py --config "$CONFIG" || \
    echo "memory probe failed - continuing; training uses the configured batch size"

step "6/6  fine-tune and evaluate (run: $RUN)"
"$PY" scripts/run_finetune.py --config "$CONFIG" --run-name "$RUN" \
    --workers "$WORKERS" --resume

step "figures"
"$PY" scripts/make_finetune_figures.py --run "$RUN" --baseline ''

step "done"
echo "report : artifacts/finetune_report_${RUN}.json"
echo "figures: artifacts/report_${RUN}/"
echo "history: logs/training_history_${RUN}.csv"
