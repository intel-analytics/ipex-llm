# GaLore Finetuning with IPEX-LLM

This example ports [GaLore on Llama2 7B](https://github.com/geronimi73/3090_shorts/blob/main/nb_galore_llama2-7b.ipynb) to IPEX-LLM on [Intel GPU](../../../README.md).

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install galore-torch
pip install accelerate==0.28.0
pip install bitsandbytes==0.43.0
pip install datasets==2.18.0
pip install transformers==4.39.1
pip install trl==0.8.1
```

### 2. GaLore Finetune

Currently, GaLore only supports local fine-tuning, and here is how to fine-tune Llama2 7B on an Intel Max GPU server:

```bash
# Configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
python galore_finetuning.py # optional parameters as below
```

Optional parameters for `galore_finetuning.py`:

**--repo-id-or-model-path** : default to `meta-llama/Llama-2-7b-hf`, and you can also specify your local model path.

**--data-path** : default to `HuggingFaceH4/helpful_instructions`, and you can also specify your local datal path, while note that changing to the other datasets will introduce code modification effort for yourself.

**--output-dir** : default to `./ipex-llm-galore` to save fine-tuned model, and you can change if needed.

### 3. Sample Output
```log
......
{'loss': 0.7624, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.12}
{'loss': 0.7557, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.12}
{'loss': 0.7079, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.13}
{'loss': 1.4144, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.13}
{'loss': 0.7582, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.13}
{'loss': 0.4273, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.13}
{'loss': 0.7137, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.13}
{'loss': 0.9176, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.14}
{'loss': 0.7178, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.14}
{'loss': 0.8935, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.14}
  5%|████▏                                                                                      | 70/1500 [xx:xx<x:xx:xx, xx.xxs/it]
......
```
