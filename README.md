# Transformer Compression with SliceGPT
Customized repo forked from [TransformerCompression](https://github.com/microsoft/TransformerCompression/tree/main) of the paper [SliceGPT](https://arxiv.org/abs/2401.15024) (ICLR'24).

## Python Environment
```bash
$ virtualenv env
$ source env/bin/activate
$ pip install -r requirements.txt
```


## Configurations (aka `args`)
Set [argparse.Namespace](https://github.com/microsoft/TransformerCompression/blob/main/experiments/run_slicegpt.py#L18) with the local configuration file rather than the bash shell as the original repo did . 
To set the experiment arguments, edit the desired configurations (args) in `configs/*.yaml`. The configurations setting rules for different purposes are as follows: 

### Slicing
Leave `model-path` and `sliced-model-path` blank and fill `model` with the model name in huggingface.

**Additional models supports on the top of [Supported models](https://github.com/microsoft/TransformerCompression/tree/main?tab=readme-ov-file#supported-models):**

- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [microsoft/phi-4](https://huggingface.co/microsoft/phi-4)

### Evaluating language task performance
#### Perplexity
Supported datasets: wikitext2, ptb, alpaca.
##### 1. Unsliced
The same args as slicing, except for setting `model-path` the same as `model`. Then in `experiments/run_slicegpt.py`: 
```python3
kwargs = prepare_slicing(slicing_args)
slicing_main(slicing_args, kwargs)
```

##### 2. Sliced
Except for the args of unsliced setting, set `sliced-model-path` with the local path that stores the sliced model.
```python3
prepare_slicing(slicing_args)
```

#### Others
TBC.
