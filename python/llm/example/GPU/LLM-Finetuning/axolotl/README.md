# Finetune LLM on Intel GPU using axolotl without writing code

This example demonstrates how to easily run LLM finetuning application using axolotl and IPEX-LLM 4bit optimizations with [Intel GPUs](../../../README.md). By applying IPEX-LLM patch, you could use axolotl on Intel GPUs using IPEX-LLM optimization without modification.

Note, this example is just used for illustrating related usage and don't guarantee convergence of training.

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.34.0 datasets
pip install fire peft==0.5.0
# install axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout v0.3.0
cp ../requirements.txt .
pip install -e .
```

### 2. Configures OneAPI environment variables and accelerate

```bash
source /opt/intel/oneapi/setvars.sh
```

Config `accelerate`, disable `use_cpu`.

```bash
accelerate config
```

### 3. Finetune

This example shows how to run [Alpaca LoRA Training](https://github.com/tloen/alpaca-lora/tree/main) directly on Intel GPU.


```
accelerate launch finetune.py qlora.yml
```

### 3. Other examples

Please refer to [axolotl examples](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.3.0/examples). Download `yml` and replace `qlora.yml` with new `yml`.

```
accelerate launch finetune.py xxx.yml
```
