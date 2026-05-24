#!/bin/bash
set -euo pipefail

SPLIT_NAME="abo_v4"

python -m vkr.finetune_dualencoder \
    --split-name ${SPLIT_NAME} \
    --model_id google/siglip2-base-patch16-256 \
    --use-lora --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --lora-target-modules q_proj,v_proj \
    --loss-function sigmoid \
    --main-image-only \
    --learning-rate 2e-4 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --scheduler cosine --warmup-ratio 0.03 \
    --train-batch-size 256 --infer-batch-size 256 \
    --epochs 20 --patience 3 --min-delta 0.001 \
    --seed 42 --device cuda \
    --checkpoint-dir ./checkpoints \
    --run-name siglip2_sigmoid


python -m vkr.finetune_dualencoder \
    --split-name ${SPLIT_NAME} \
    --model_id google/siglip2-base-patch16-256 \
    --use-lora \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --lora-target-modules q_proj,v_proj \
    --loss-function infonce \
    --main-image-only \
    --learning-rate 2e-4 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --scheduler cosine --warmup-ratio 0.05 \
    --train-batch-size 256 --infer-batch-size 256 \
    --epochs 20 --patience 3 --min-delta 0.001 \
    --seed 42 --device cuda \
    --checkpoint-dir ./checkpoints \
    --run-name siglip2_infonce


python -m vkr.finetune_dualencoder \
    --split-name ${SPLIT_NAME} \
    --model_id google/siglip2-base-patch16-256 \
    --use-lora \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --lora-target-modules q_proj,v_proj,k_proj,out_proj \
    --loss-function sigmoid \
    --main-image-only \
    --learning-rate 2e-4 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --scheduler cosine --warmup-ratio 0.03 \
    --train-batch-size 192 --infer-batch-size 256 \
    --epochs 20 --patience 3 --min-delta 0.001 \
    --seed 42 --device cuda \
    --checkpoint-dir ./checkpoints \
    --run-name siglip2_all_proj


python -m vkr.finetune_dualencoder \
    --split-name ${SPLIT_NAME} \
    --model_id google/siglip2-base-patch16-256 \
    --use-lora \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --lora-target-modules q_proj,v_proj \
    --loss-function sigmoid \
    --use-multi-positive \
    --learning-rate 2e-4 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --scheduler cosine --warmup-ratio 0.03 \
    --train-batch-size 256 --infer-batch-size 256 \
    --epochs 20 --patience 3 --min-delta 0.001 \
    --seed 42 --device cuda \
    --checkpoint-dir ./checkpoints \
    --run-name siglip2_all_images


python -m vkr.finetune_dualencoder \
    --split-name ${SPLIT_NAME} \
    --model_id google/siglip2-so400m-patch14-384 \
    --use-lora \
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,out_proj \
    --loss-function sigmoid \
    --use-multi-positive \
    --main-image-only \
    --learning-rate 2e-4 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --scheduler cosine --warmup-ratio 0.05 \
    --train-batch-size 20 --infer-batch-size 256 \
    --epochs 20 --patience 5 --min-delta 0.001 \
    --seed 42 --device cuda \
    --checkpoint-dir ./checkpoints \
    --run-name siglip2_heavy
