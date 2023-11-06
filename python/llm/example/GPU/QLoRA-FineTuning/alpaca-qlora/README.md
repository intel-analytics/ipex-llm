# Alpaca QLoRA Finetuning (experimental support)

This example demonstrates how to finetune a llama2-7b model use BigDL-LLM 4bit optimizations using [Intel GPUs](../README.md).

## 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

## Example: Alpaca QLoRA Finetuning

This example is ported from [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py). Just a few code changes are needed to finetune a llama2-7b model using QLoRA with BigDL-LLM 4bit optimizations on Intel GPUs.

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.34.0
pip install datasets peft==0.5.0
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune model

Before running the example, please make sure you have prepared the [templates folder](https://github.com/tloen/alpaca-lora/tree/main/templates) in the current directory.

Example usage on single Arc:

```
python ./alpaca_qlora_finetuning.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./bigdl-qlora-alpaca"
```

#### Sample Output
```log
{'loss': 1.9231, 'learning_rate': 2.9999945367033285e-05, 'epoch': 0.0}                                                                                                                            
{'loss': 1.8622, 'learning_rate': 2.9999781468531096e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.9043, 'learning_rate': 2.9999508305687345e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.8967, 'learning_rate': 2.999912588049185e-05, 'epoch': 0.01}                                                                                                                            
{'loss': 1.9658, 'learning_rate': 2.9998634195730358e-05, 'epoch': 0.01}                                                                                                                           
{'loss': 1.8386, 'learning_rate': 2.9998033254984483e-05, 'epoch': 0.02}                                                                                                                           
{'loss': 1.809, 'learning_rate': 2.999732306263172e-05, 'epoch': 0.02}                                                                                                                             
{'loss': 1.8552, 'learning_rate': 2.9996503623845395e-05, 'epoch': 0.02}                                                                                                                           
  1%|█                                                                                                                                                         | 8/1164 [xx:xx<xx:xx:xx, xx s/it]
```
