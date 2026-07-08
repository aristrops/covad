#!/usr/bin/env bash
# Sanity experiments for the unified STFPM+concept branch (fully-supervised).
# Trains + evaluates on MVTec hazelnut and cable.
#
# Usage:  bash main_unified/run_unified_mvtec.sh [DEVICE]
set -euo pipefail

DEVICE="${1:-cuda}"
PY="${PYTHON:-.venv/bin/python}"
TEACHER="cbm_models/backbones/fine-tuned-mobilenet.pth"

RESULTS="results/unified_results.csv"

for CAT in hazelnut cable; do
    DF="cbm_data/mvtec/${CAT}_dataset_automated.csv"
    SAVE="cbm_models/mvtec/unified/${CAT}/mobilenet_v2.pth"

    echo "==================== TRAIN ${CAT} ===================="
    "$PY" -m main_unified.unified \
        --mode train \
        --dataframe_path "$DF" \
        --category "$CAT" \
        --backbone mobilenet_v2 \
        --device "$DEVICE" \
        --teacher_path "$TEACHER" \
        --save_path "$SAVE" \
        --epochs 50 \
        --batch_size 8 \
        --lambda_ 0.55

    echo "==================== EVAL ${CAT} ===================="
    "$PY" -m main_unified.unified \
        --mode eval \
        --dataframe_path "$DF" \
        --category "$CAT" \
        --backbone mobilenet_v2 \
        --device "$DEVICE" \
        --save_path "$SAVE" \
        --dataset mvtec \
        --results_path "$RESULTS"
done
