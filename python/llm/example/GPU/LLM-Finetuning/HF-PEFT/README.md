# Finetuning on Intel GPU using Hugging Face PEFT code

This example demonstrates how to easily run LLM finetuning application of PEFT use IPEX-LLM 4bit optimizations using [Intel GPUs](../../../README.md). By applying IPEX-LLM patch, you could run Hugging Face PEFT code on Intel GPUs using IPEX-LLM optimization without modification.

Note, this example is just used for illustrating related usage and don't guarantee convergence of training.

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.45.0 "trl<0.12.0" datasets
pip install bitsandbytes==0.45.1 scipy
pip install fire peft==0.10.0
pip install oneccl_bind_pt==2.1.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ # necessary to run distributed finetuning
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune

This example shows how to run [Alpaca LoRA Training](https://github.com/tloen/alpaca-lora/tree/main) directly on Intel GPU.

```
cd alpaca-lora
python ./finetune.py --base_model "meta-llama/Llama-2-7b-hf" \
                     --data_path "yahma/alpaca-cleaned"
```
