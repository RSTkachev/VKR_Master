#!/bin/bash

python -m vkr.splits create \
    --name abo_v4 \
    --train-siglip 14500 \
    --train-reranker 14500 \
    --val 10000 \
    --test 10000 \
    --preferred-lang en \
    --strict-preferred-lang \
    --dedupe-by-main-image \
    --unique-other-images-only \
    --subsample-category "CELLULAR_PHONE_CASE:5000" \
    --seed 42
