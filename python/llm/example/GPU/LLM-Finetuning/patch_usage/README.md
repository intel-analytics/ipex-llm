# Finetuning with BigDL-LLM through patch function

This example demonstrates how to easily run LLM finetuning application of PEFT use BigDL-LLM 4bit optimizations using [Intel GPUs](../../../README.md). By applying BigDL-LLM patch, you could run CUDA code on Intel GPUs using BigDL-LLM optimization without changing any other code.

Note, this example is just used for illustrating related usage and don't guarantee convergence of training.

### 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.34.0 datasets
pip install fire peft==0.5.0
pip install oneccl_bind_pt==2.1.100 -f https://developer.intel.com/ipex-whl-stable-xpu # necessary to run distributed finetuning
pip install accelerate==0.23.0
pip install bitsandbytes scipy
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune

This example shows how to apply bigdl-llm patch on [bnb-4bit-training](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing).

```
python ./finetune.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH
```
