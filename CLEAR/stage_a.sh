#!/usr/bin/env bash
# Stage A — measure r_i on the post-weakening KVW model, eval per-sample,
# then run the analysis. Assumes:
#   - kvw.sh has been run and checkpoints/KVW_05 exists
#   - kc/kc_r_retain_95.pt exists (from compute_kc_r.sh)
#
# Decision-locked config:
#   - Option B: measure r_i on the final (post-weakening) model
#   - Layer range: [start_layer, end_layer] = [1, 25] (match KVW intervention)
#   - "Correct" label: substring match from eval.py

set -euo pipefail

MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"
WEAKENED_DIR="checkpoints/KVW_05"
FORGET_RATIO=5
RETAIN_RATIO=$((100 - FORGET_RATIO))
FORGET_STR=$(printf '%02d' ${FORGET_RATIO})
RETAIN_STR=$(printf '%02d' ${RETAIN_RATIO})
START_LAYER=1
END_LAYER=25
OUTDIR="outputs/stage_a_forget${FORGET_STR}"

# Folder names align with eval.sh in this repo.
FORGET_CLS_FOLDER="forget${FORGET_STR}_perturbed"
FORGET_GEN_FOLDER="forget${FORGET_STR}+tofu"
RETAIN_CLS_FOLDER="retain_perturbed"
RETAIN_GEN_FOLDER="retain${RETAIN_STR}+tofu"
REALFACE_FOLDER="real_faces"
REALWORLD_FOLDER="real_world"

mkdir -p "${OUTDIR}"

echo "=== Step 1/3: compute per-sample r_i on weakened model ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -m baselines.KVW \
    --model_id "${MODEL_ID}" \
    --vanilla_dir "${WEAKENED_DIR}" \
    --forget_ratio "${FORGET_RATIO}" \
    --batch_size 1 \
    --num_epochs 1 \
    --phase stage_a_measure \
    --data_folder data/CLEAR \
    --save_dir "${OUTDIR}" \
    --start_layer "${START_LAYER}" \
    --end_layer "${END_LAYER}"

echo "=== Step 2/3: per-sample eval on weakened model ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 eval.py \
    --model_id "${MODEL_ID}" \
    --cache_path "${WEAKENED_DIR}" \
    --data_folder data/CLEAR \
    --forget_cls_folder "${FORGET_CLS_FOLDER}" \
    --forget_gen_folder "${FORGET_GEN_FOLDER}" \
    --retain_cls_folder "${RETAIN_CLS_FOLDER}" \
    --retain_gen_folder "${RETAIN_GEN_FOLDER}" \
    --realface_folder "${REALFACE_FOLDER}" \
    --realworld_folder "${REALWORLD_FOLDER}" \
    --output_folder "${OUTDIR}/eval" \
    --eval_list "forget" \
    --shot_num zero_shots

echo "=== Step 3/3: analysis ==="
python3 ../scripts/stage_a_analysis.py \
    --r-path "kc/r_stageA_forget${FORGET_STR}.pt" \
    --eval-path "${OUTDIR}/eval/forget_per_sample.json" \
    --output-dir "${OUTDIR}/analysis"

echo "Stage A done. See ${OUTDIR}/analysis/"
