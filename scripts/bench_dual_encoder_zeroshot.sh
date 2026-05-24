#!/bin/bash
set -euo pipefail

SCRIPT_NAME="vkr.bench_model"
PROJECT_PATH="./"
OUTPUT_ROOT="./eval_results/dual_encoder_zeroshot"

SPLIT_NAME="abo_v4"
SPLIT_PART="val"
TOP_K=50

DEVICE="cuda"
SEED=42

MODELS=(
    "metaclip:facebook/metaclip-b16-400m"
    "metaclip_heavy:facebook/metaclip-l14-400m"
    "siglip:google/siglip-base-patch16-256"
    "siglip_heavy:google/siglip-so400m-patch14-384"
    "siglip2:google/siglip2-base-patch16-256"
    "siglip2_heavy:google/siglip2-so400m-patch14-384"
)

mkdir -p "$OUTPUT_ROOT"

echo "Starting evaluation of ${#MODELS[@]} models on split '$SPLIT_NAME' (part: $SPLIT_PART)..."

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    MODEL_ID="${entry#*:}"

    IMG_BS=256
    TXT_BS=256

    echo "----------------------------------------------------"
    echo "Evaluating model: $NAME ($MODEL_ID)"
    echo "----------------------------------------------------"

    python -m "$SCRIPT_NAME" \
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
        --model_id "$MODEL_ID" 

    sleep 5
done

echo "All evaluations completed!"
