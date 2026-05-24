#!/bin/bash
set -euo pipefail

PROJECT_PATH="./"
OUTPUT_ROOT="./eval_results/two_stage_zeroshot"

SPLIT_NAME="abo_v4"
SPLIT_PART="val"
TOP_K=10

DEVICE="cuda"
SEED=42

NAME="siglip2"
MODEL_ID="google/siglip2-so400m-patch14-384"

mkdir -p "$OUTPUT_ROOT"

IMG_BS=256
TXT_BS=256

echo "----------------------------------------------------"
echo "Evaluating model: $NAME ($MODEL_ID)"
echo "----------------------------------------------------"

python -m vkr.bench_model \
    --project-path "$PROJECT_PATH" \
    --output-dir "$OUTPUT_ROOT" \
    --split-name "$SPLIT_NAME" \
    --split-part "$SPLIT_PART" \
    --top-k $TOP_K \
    --seed $SEED \
    --device "$DEVICE" \
    --image-batch-size $IMG_BS \
    --text-batch-size $TXT_BS \
    --model "$NAME" \
    --model_id "$MODEL_ID" \
    --lora-checkpoint "./checkpoints/abo_v4_sigmoid_lora_r8_siglip2_heavy" \
    --second_model_id "Qwen/Qwen3.5-4B" \
    --rerank-chunk-size 16 \
    --rerank-mode "pointwise"

sleep 5

echo "All evaluations completed!"
