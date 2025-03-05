PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 experiments/finetuning.py \
           --model facebook/opt-125m \
           --sliced-model-path data/sliced \
           --save-dir data/sliced/finetune \
           --sparsity 0.2 5 \
           --device cuda:0 \
           --ppl-eval-dataset alpaca \
           --finetune-dataset alpaca \
           --finetune-train-nsamples 8000 \
           --finetune-train-seqlen 1024 \
           --finetune-train-batch-size 1 \
           --lora-alpha 10 \
           --lora-r 32 \
           --lora-dropout 0.1 \
           --lora-target-option attn_head_and_mlp \
           --eval-steps 1 \
           --save-steps 1 \
           --no-wandb