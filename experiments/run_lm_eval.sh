#!/bin/bash
# conda activate slicegpt

#!/bin/bash
# conda activate slicegpt

MODEL_NAME_OR_PATH=$1
SPARSITY=$2
SAVE_DIR=pruned/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}
# LOG_PATH=logs/${MODEL_NAME_OR_PATH//\//-}_sparsity_${SPARSITY}_pruned.log


python run_lm_eval.py \
    --model ${MODEL_NAME_OR_PATH} \
    --sliced-model-path ${SAVE_DIR} \
    --sparsity ${SPARSITY} \
    --tasks  "piqa" "hellaswag" "winogrande" "arc_easy" "arc_challenge" "boolq" "openbookqa" "mathqa" \
    --no-wandb 
