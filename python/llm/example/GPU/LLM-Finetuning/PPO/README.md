# PPO Finetuning with IPEX-LLM

This example ports to [ppo finetuning](https://github.com/jasonvanf/llama-trl/blob/main/tuning_lm_with_rl.py), and adds trl PPO finetuning to IPEX-LLM on [Intel GPU](../../../GPU/README.md).

### 0. Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../GPU/README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install accelerate==0.28.0
pip install datasets==2.18.0
pip install transformers==4.37.0
pip install trl tqdm peft
```

### 2. PPO Finetune

```bash
# Configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
python ./lisa_finetuning.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --dataset_name "HuggingFaceH4/helpful_instructions" \
```
