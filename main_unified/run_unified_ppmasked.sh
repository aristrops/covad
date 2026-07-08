#!/usr/bin/env bash
# unified++masked ABLATION: unified++ but anomalous samples never update the student
# backbone (--mask_student). Concept net + main head still train on all samples;
# the STFPM loss is already normal-only. Isolates "does the student benefit from
# seeing anomalies via the concept path?".
# Same 4 cats as the unified++ spot-check, to compare directly.
# Results -> results/unified_ppmasked_results.{md,csv}
#
# Usage:  CUDA_VISIBLE_DEVICES=1 bash main_unified/run_unified_ppmasked.sh [DEVICE] [EPOCHS]
set -uo pipefail
export PYTHONUNBUFFERED=1

DEVICE="${1:-cuda}"
EPOCHS="${2:-50}"
PY="${PYTHON:-.venv/bin/python}"
TEACHER="cbm_models/backbones/fine-tuned-mobilenet.pth"
RESULTS="results/unified_ppmasked_results.csv"
mkdir -p results

for CAT in hazelnut leather capsule grid; do
    DF="cbm_data/mvtec/${CAT}_dataset_automated.csv"
    SAVE="cbm_models/mvtec/unified_ppmasked/${CAT}/mobilenet_v2.pth"
    echo "########## unified++masked ${CAT} ##########"

    "$PY" -m main_unified.unified --mode train \
        --dataframe_path "$DF" --category "$CAT" --backbone mobilenet_v2 \
        --device "$DEVICE" --teacher_path "$TEACHER" --save_path "$SAVE" \
        --epochs "$EPOCHS" --batch_size 8 --lambda_ 0.55 --inject_diffs --mask_student \
        || { echo "${CAT}: TRAIN FAILED"; continue; }

    "$PY" -m main_unified.unified --mode eval \
        --dataframe_path "$DF" --category "$CAT" --backbone mobilenet_v2 \
        --device "$DEVICE" --save_path "$SAVE" \
        --dataset mvtec --results_path "$RESULTS" --inject_diffs \
        || echo "${CAT}: EVAL FAILED"
done

echo "=== unified++masked DONE -> results/unified_ppmasked_results.md ==="
