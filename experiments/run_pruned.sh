#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# /bin/bash run_slicegpt.sh "meta-llama/Llama-2-7b-hf" 0.2
/bin/bash run_lm_eval.sh "meta-llama/Llama-2-7b-hf" 0.2

# /bin/bash run_slicegpt.sh "meta-llama/Llama-3.1-8B" 0.2
# /bin/bash run_lm_eval.sh "meta-llama/Llama-3.1-8B" 0.2


# /bin/bash run_slicegpt.sh "mistralai/Mistral-7B-v0.3" 0.2
# /bin/bash run_lm_eval.sh "mistralai/Mistral-7B-v0.3" 0.2

