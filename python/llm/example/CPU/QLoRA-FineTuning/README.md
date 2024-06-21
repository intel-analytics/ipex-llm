# Finetuning LLAMA Using QLoRA (experimental support)

This example demonstrates how to finetune a llama2-7b model using Big-LLM 4bit optimizations on [Intel CPUs](../README.md).


## Distributed Training Guide
1. Single node with single socket: [simple example](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/QLoRA-FineTuning#example-finetune-llama2-7b-using-qlora)
or [alpaca example](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/QLoRA-FineTuning/alpaca-qlora)
2. [Single node with multiple sockets](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/QLoRA-FineTuning/alpaca-qlora#guide-to-finetuning-qlora-on-one-node-with-multiple-sockets)
3. [multiple nodes with multiple sockets](https://github.com/intel-analytics/ipex-llm/blob/main/docker/llm/finetune/qlora/cpu/kubernetes/README.md)

## Example: Finetune llama2-7b using QLoRA

This example is ported from [bnb-4bit-training](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k). 

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.36.0
pip install peft==0.10.0
pip install datasets
pip install bitsandbytes scipy
```

### 2. Finetune model
If the machine memory is not enough, you can try to set `use_gradient_checkpointing=True` in [here](https://github.com/intel-analytics/ipex-llm/blob/1747ffe60019567482b6976a24b05079274e7fc8/python/llm/example/CPU/QLoRA-FineTuning/qlora_finetuning_cpu.py#L53C6-L53C6). While gradient checkpointing may improve memory efficiency, it slows training by approximately 20%.
We Recommend using micro_batch_size of 8 for better performance using 48cores in this example. You can refer to [this guide](https://huggingface.co/docs/transformers/perf_train_gpu_one) for more details.
And remember to use `ipex-llm-init` before you start finetuning, which can accelerate the job.

```
source ipex-llm-init -t
python ./qlora_finetuning_cpu.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --dataset DATASET
```

#### Sample Output
```log
{'loss': 2.0251, 'learning_rate': 0.0002, 'epoch': 0.02}
{'loss': 1.2389, 'learning_rate': 0.00017777777777777779, 'epoch': 0.03}
{'loss': 1.032, 'learning_rate': 0.00015555555555555556, 'epoch': 0.05}
{'loss': 0.9141, 'learning_rate': 0.00013333333333333334, 'epoch': 0.06}
{'loss': 0.8505, 'learning_rate': 0.00011111111111111112, 'epoch': 0.08}
{'loss': 0.8713, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.09}
{'loss': 0.8635, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.11}
{'loss': 0.8853, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.12}
{'loss': 0.859, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.14}
{'loss': 0.8608, 'learning_rate': 0.0, 'epoch': 0.15}
{'train_runtime': xxxx, 'train_samples_per_second': xxxx, 'train_steps_per_second': xxxx, 'train_loss': 1.0400420665740966, 'epoch': 0.15}
100%|███████████████████████████████████████████████████████████████████████████████████| 200/200 [07:16<00:00,  2.18s/it]
TrainOutput(global_step=200, training_loss=1.0400420665740966, metrics={'train_runtime': xxxx, 'train_samples_per_second': xxxx, 'train_steps_per_second': xxxx, 'train_loss': 1.0400420665740966, 'epoch': 0.15})
```

### 3. Merge the adapter into the original model
Using the [export_merged_model.py](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/QLoRA/export_merged_model.py) to merge.
```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.
