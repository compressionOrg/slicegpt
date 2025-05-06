#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# "meta-llama/Llama-2-7b-hf"  "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3" 
MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf" 
SPARSITY=0.2
SAVE_DIR=pruned/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}
# LOG_PATH=logs/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}_pruned.log


python evaluate_ppl.py \
    --model ${MODEL_NAME_OR_PATH} \
    --sliced-model-path ${SAVE_DIR} \
    --sparsity ${SPARSITY} \
    --cal-datasets wikitext2 ptb c4 \
    --ppl-eval-batch-size 8 \
    --dtype fp16 \
    --no-wandb