#!/usr/bin/env bash
# unified++ : same as unified, but ALSO add the deeper feature differences (f8, f14)
# into the matching internal activations of the truncated concept net (--inject_diffs).
# Check it doesn't break on easy cats (hazelnut, leather) and whether it helps the
# hard ones (capsule, grid). Results -> results/unified_pp_results.{md,csv}
#
# Usage:  CUDA_VISIBLE_DEVICES=2 bash main_unified/run_unified_pp.sh [DEVICE] [EPOCHS]
set -uo pipefail
export PYTHONUNBUFFERED=1

DEVICE="${1:-cuda}"
EPOCHS="${2:-50}"
PY="${PYTHON:-.venv/bin/python}"
TEACHER="cbm_models/backbones/fine-tuned-mobilenet.pth"
RESULTS="results/unified_pp_results.csv"
mkdir -p results

for CAT in hazelnut leather capsule grid; do
    DF="cbm_data/mvtec/${CAT}_dataset_automated.csv"
    SAVE="cbm_models/mvtec/unified_pp/${CAT}/mobilenet_v2.pth"
    echo "########## unified++ ${CAT} ##########"

    "$PY" -m main_unified.unified --mode train \
        --dataframe_path "$DF" --category "$CAT" --backbone mobilenet_v2 \
        --device "$DEVICE" --teacher_path "$TEACHER" --save_path "$SAVE" \
        --epochs "$EPOCHS" --batch_size 8 --lambda_ 0.55 --inject_diffs \
        || { echo "${CAT}: TRAIN FAILED"; continue; }

    "$PY" -m main_unified.unified --mode eval \
        --dataframe_path "$DF" --category "$CAT" --backbone mobilenet_v2 \
        --device "$DEVICE" --save_path "$SAVE" \
        --dataset mvtec --results_path "$RESULTS" --inject_diffs \
        || echo "${CAT}: EVAL FAILED"
done

echo "=== unified++ DONE -> results/unified_pp_results.md ==="
