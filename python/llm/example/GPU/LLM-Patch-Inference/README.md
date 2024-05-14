# Inference on Intel GPU using Hugging Face code

This example demonstrates how to easily run LLM inference application with IPEX-LLM 4bit optimizations using [Intel GPUs](../README.md). By applying IPEX-LLM patch, you could run Hugging Face code on Intel GPUs using IPEX-LLM optimization without modification.

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Inference

This example shows how to run [LLM Foundry Inference](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/hf_generate.py). You could still use the following command but directly run on Intel GPU.

```
python hf_generate.py --name_or_path "meta-llama/Llama-2-7b-chat-hf" --device 'cuda'
```
