#!/bin/bash
# conda activate slicegpt2

MODEL_NAME_OR_PATH=$1
SPARSITY=$2
SAVE_DIR=pruned/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}
# LOG_PATH=logs/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}_pruned.log
mkdir -p $SAVE_DIR
mkdir -p logs

python run_slicegpt.py \
        --model ${MODEL_NAME_OR_PATH} \
        --save-dir ${SAVE_DIR} \
        --sparsity ${SPARSITY} \
        --device cuda:0 \
        --eval-baseline \
        --no-wandb 
