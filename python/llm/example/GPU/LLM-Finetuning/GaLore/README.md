# GaLore Finetuning with IPEX-LLM

This is an example of IPEX-LLM GaLore fine-tuning on [Intel GPU](../../../README.md), which refers [Huggingface GaLore blog](https://huggingface.co/blog/galore) and changes model to [openlm-research/open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2) and dataset to [HuggingFaceH4/helpful_instructions](https://huggingface.co/datasets/HuggingFaceH4/helpful_instructions).

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install galore-torch
pip install transformers==4.45.0 "trl<0.12.0" datasets
pip install bitsandbytes==0.45.1
```

### 2. GaLore Finetune

Currently, GaLore only supports local fine-tuning, and here is how to fine-tune Llama2 7B on an Intel Max GPU server:

```bash
# Configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
python galore_finetuning.py # optional parameters as below
```

Optional parameters for `galore_finetuning.py`:

**--repo-id-or-model-path** : default to `openlm-research/open_llama_3b_v2`, and you can also specify your local model path.

**--data-path** : default to `HuggingFaceH4/helpful_instructions`, and you can also specify your local datal path, while note that changing to the other datasets will introduce code modification effort for yourself.

**--output-dir** : default to `./ipex-llm-galore` to save fine-tuned model, and you can change if needed.

### 3. Sample Output
```log
......
{'loss': 2.0989, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.0}
{'loss': 1.9064, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.0}
{'loss': 1.7483, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.9551, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.783, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.3328, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.4622, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.9094, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.02}
  5%|████▏                                                                                      | 70/1500 [xx:xx<x:xx:xx, xx.xxs/it]
......
```
