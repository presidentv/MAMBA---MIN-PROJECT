#!/usr/bin/env bash
# Binary disengagement detector on the FULL DAiSEE release, end to end.
#
#   bash scripts/run_fullbin.sh
#
# Waits for any Stage 1 shards already in flight, fits the MRS calibration on
# the full training split, then trains and evaluates. Safe to re-run: Stage 1
# skips cached clips and training resumes from last.pt.
#
# The dataset is read from a removable drive, so DAISEE_ROOT points at it while
# the cache and checkpoints go to internal disk -- reading 8,570 videos over USB
# once is fine, doing it every epoch would not be.

set -uo pipefail
cd "$(dirname "$0")/.."

export DAISEE_ROOT="${DAISEE_ROOT:-D:/DAiSEE}"
export MRG_CACHE_DIR="${MRG_CACHE_DIR:-C:/Users/ashok/mrg_run/cache}"
export MRG_CHECKPOINT_DIR="${MRG_CHECKPOINT_DIR:-C:/Users/ashok/mrg_run/checkpoints}"
CONFIG="configs/config_binary_full.yaml"
RUN="${RUN:-fullbin}"
PY="./.venv/Scripts/python.exe"

step() { printf '\n==== [%s] %s ====\n' "$(date +%H:%M:%S)" "$*"; }

step "waiting for any Stage 1 shards still running"
while tasklist 2>/dev/null | grep -q "python"; do
    n=$(find "$MRG_CACHE_DIR" -name "*.npz" 2>/dev/null | wc -l)
    printf '  %s  %s clips cached\n' "$(date +%H:%M:%S)" "$n"
    sleep 60
done

step "Stage 1 consolidation pass (retries anything a shard failed)"
"$PY" scripts/run_preprocessing.py --config "$CONFIG" --stage 1 2>&1 | tail -4

step "MRS calibration on the full training split"
"$PY" scripts/fit_mrs_stats.py --config "$CONFIG" 2>&1 | tail -3

# Two arms, identical in all but two settings (mrs.weighting_mode and
# mamba.pooling, verified: 2 differences out of 117). The full model alone
# would only show that the backbone works; the DIFFERENCE between the arms is
# what tests whether reliability weighting contributes anything.
#
# Each arm trains, selects its checkpoint on validation balanced accuracy, and
# then reads train / val / test once. Test never informs selection.
step "ARM 1 of 2: full model, reliability weighting ON (run: $RUN)"
"$PY" scripts/run_finetune.py --config "$CONFIG" --run-name "$RUN" \
    --workers 2 --resume --tta

step "ARM 2 of 2: control, reliability weighting REMOVED (run: ${RUN}_nomrs)"
"$PY" scripts/run_finetune.py --config configs/config_binary_full_nomrs.yaml \
    --run-name "${RUN}_nomrs" --workers 2 --resume --tta

step "done"
echo "full model : artifacts/finetune_report_${RUN}.json"
echo "control    : artifacts/finetune_report_${RUN}_nomrs.json"
echo "histories  : logs/training_history_${RUN}.csv, logs/training_history_${RUN}_nomrs.csv"
