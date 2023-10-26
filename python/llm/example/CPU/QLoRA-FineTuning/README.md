# Finetuning LLAMA Using Q-Lora (experimental support)

This example demonstrates how to finetune a llama2-7b model use Big-LLM 4bit optimizations using [Intel CPUs](../README.md).


## Example: Finetune llama2-7b using qlora

This example is ported from [bnb-4bit-training](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing). The `export_merged_model.py` is ported from [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py).

### 1. Install

```bash
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[all]
pip install transformers==4.34.0
pip install peft==0.5.0
pip install datasets
```

### 2. Finetune model

```
python ./qlora_finetuning_cpu.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --dataset DATASET
```

#### Sample Output
```log
{'loss': 2.2771, 'learning_rate': 0.0002, 'epoch': 0.03}
{'loss': 1.891, 'learning_rate': 0.00017777777777777779, 'epoch': 0.06}
{'loss': 1.5504, 'learning_rate': 0.00015555555555555556, 'epoch': 0.1}
{'loss': 1.497, 'learning_rate': 0.00013333333333333334, 'epoch': 0.13}
{'loss': 1.3256, 'learning_rate': 0.00011111111111111112, 'epoch': 0.16}
{'loss': 1.2642, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.19}
{'loss': 1.397, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.22}
{'loss': 1.2516, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.26}
{'loss': 1.3439, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.29}
{'loss': 1.3288, 'learning_rate': 0.0, 'epoch': 0.32}
{'train_runtime': 410.4375, 'train_samples_per_second': 1.949, 'train_steps_per_second': 0.487, 'train_loss': 1.5126623916625976, 'epoch': 0.32}
100%|██████████████████████████████████████████████████████████████████████████████████████| 200/200 [06:50<00:00,  2.05s/it]
TrainOutput(global_step=200, training_loss=1.5126623916625976, metrics={'train_runtime': 410.4375, 'train_samples_per_second': 1.949, 'train_steps_per_second': 0.487, 'train_loss': 1.5126623916625976, 'epoch': 0.32}
```

### 3. Merge the adapter into the original model

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.
