# Alpaca QLoRA Finetuning (experimental support)

This example demonstrates how to finetune a llama2-7b model use BigDL-LLM 4bit optimizations using [Intel GPUs](../README.md).

## 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Alpaca QLoRA Finetuning

This example is ported from is ported from [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py).

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.34.0
pip install peft==0.5.0
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune model

```
python ./alpaca_qlora_finetuning.py
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 (e.g.`meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-hf'`.
- `--dataset DATASET`: argument defining the dataset to use. It is default to be `'yahma/alpaca-cleaned'`.
- `--output_dir OUTPUT_DIR`: argument defining the path to save the adapter. It is default to be `"./bigdl-qlora-alpaca"`.

#### Sample Output
```log
{'loss': 1.6134, 'learning_rate': 0.0002, 'epoch': 0.03}                                                                                 
{'loss': 1.3038, 'learning_rate': 0.00017777777777777779, 'epoch': 0.06}                                                                 
{'loss': 1.2634, 'learning_rate': 0.00015555555555555556, 'epoch': 0.1}                                                                  
{'loss': 1.2389, 'learning_rate': 0.00013333333333333334, 'epoch': 0.13}                                                                 
{'loss': 1.0399, 'learning_rate': 0.00011111111111111112, 'epoch': 0.16}                                                                 
{'loss': 1.0406, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.19}                                                                  
{'loss': 1.3114, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.22}                                                                  
{'loss': 0.9876, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.26}                                                                 
{'loss': 1.1406, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.29}                                                                 
{'loss': 1.1728, 'learning_rate': 0.0, 'epoch': 0.32}                                                                                    
{'train_runtime': 225.8005, 'train_samples_per_second': 3.543, 'train_steps_per_second': 0.886, 'train_loss': 1.211241865158081, 'epoch': 0.32}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:45<00:00,  1.13s/it]
TrainOutput(global_step=200, training_loss=1.211241865158081, metrics={'train_runtime': 225.8005, 'train_samples_per_second': 3.543, 'train_steps_per_second': 0.886, 'train_loss': 1.211241865158081, 'epoch': 0.32})
```
