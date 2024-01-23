# Simple Example of QLoRA Finetuning with BigDL-LLM

This simple example demonstrates how to finetune a llama2-7b model use BigDL-LLM 4bit optimizations using [Intel GPUs](../../../README.md).
Note, this example is just used for illustrating related usage and don't guarantee convergence of training.

## 0. Requirements
To run this example with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Finetune llama2-7b using qlora

This example is ported from [bnb-4bit-training](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing). The `export_merged_model.py` is ported from [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py).

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.34.0 datasets
pip install peft==0.5.0
pip install accelerate==0.23.0
pip install bitsandbytes scipy
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Finetune model

```
python ./qlora_finetuning.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH
```

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

### 4. Merge the adapter into the original model

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.
