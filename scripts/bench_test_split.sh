#!/bin/bash
set -euo pipefail

python -m vkr.bench_model \
    --split-name abo_v4 --split-part test \
    --model siglip2 \
    --model_id google/siglip2-so400m-patch14-384 \
    --top-k 10 \
    --run-name retriever_only

python -m vkr.bench_model \
    --split-name abo_v4 --split-part test \
    --model siglip2 \
    --model_id google/siglip2-so400m-patch14-384 \
    --lora-checkpoint ./checkpoints/abo_v4_sigmoid_lora_r8_siglip2_heavy \
    --top-k 10 \
	--save-per-query-ranks \
    --run-name retriever_finetuned

python -m vkr.bench_model \
    --split-name abo_v4 --split-part test \
    --model siglip2 \
    --model_id google/siglip2-so400m-patch14-384 \
    --lora-checkpoint ./checkpoints/abo_v4_sigmoid_lora_r8_siglip2_heavy \
    --top-k 10 \
    --second_model_id Qwen/Qwen3.5-4B \
    --rerank-mode listwise \
    --run-name reranker_zeroshot

python -m vkr.bench_model \
    --split-name abo_v4 --split-part test \
    --model siglip2 \
    --model_id google/siglip2-so400m-patch14-384 \
    --lora-checkpoint ./checkpoints/abo_v4_sigmoid_lora_r8_siglip2_heavy \
    --top-k 10 \
    --second_model_id Qwen/Qwen3.5-4B \
    --rerank-mode listwise \
    --second-lora-checkpoint ./checkpoints/qwen_listwise_lora_v3 \
	--save-per-query-ranks \
    --run-name reranker_finetuned_test_v3
