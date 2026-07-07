#!/usr/bin/env bash
# Full-scale unified runs: all MVTec + VisA categories (fully-supervised).
# Trains + evaluates each; checkpoints -> cbm_models/<dataset>/unified/<cat>/mobilenet_v2.pth
# Results summary appended to results/unified_summary.txt
#
# Usage:  CUDA_VISIBLE_DEVICES=1 bash main_scripts/run_unified_all.sh [DEVICE] [EPOCHS]
set -uo pipefail
export PYTHONUNBUFFERED=1

DEVICE="${1:-cuda}"
EPOCHS="${2:-50}"
PY="${PYTHON:-.venv/bin/python}"

MVTEC="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
VISA="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"

RESULTS="results/unified_results.csv"
mkdir -p results

run_one () {
    local DATASET="$1" CAT="$2" TEACHER="$3"
    local DF="cbm_data/${DATASET}/${CAT}_dataset_automated.csv"
    local SAVE="cbm_models/${DATASET}/unified/${CAT}/mobilenet_v2.pth"
    echo "########## ${DATASET}/${CAT} ##########"

    "$PY" -m main_scripts.unified --mode train \
        --dataframe_path "$DF" --category "$CAT" --backbone mobilenet_v2 \
        --device "$DEVICE" --teacher_path "$TEACHER" --save_path "$SAVE" \
        --epochs "$EPOCHS" --batch_size 8 --lambda_ 0.55 \
        || { echo "${DATASET}/${CAT}: TRAIN FAILED"; return; }

    "$PY" -m main_scripts.unified --mode eval \
        --dataframe_path "$DF" --category "$CAT" --backbone mobilenet_v2 \
        --device "$DEVICE" --save_path "$SAVE" \
        --dataset "$DATASET" --results_path "$RESULTS" \
        || echo "${DATASET}/${CAT}: EVAL FAILED"
}

for CAT in $MVTEC; do run_one mvtec "$CAT" "cbm_models/backbones/fine-tuned-mobilenet.pth"; done
for CAT in $VISA;  do run_one visa  "$CAT" "cbm_models/backbones/fine-tuned-mobilenet-visa.pth"; done

echo "=== ALL DONE @ $(date) -> results/unified_results.md ==="
