# Example of QLoRA Finetuning with IPEX-LLM

This simple example demonstrates how to finetune a llama2-7b model use IPEX-LLM 4bit optimizations with TRL library on [Intel GPU](../../../README.md).
Note, this example is just used for illustrating related usage and don't guarantee convergence of training.

## 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Finetune llama2-7b using qlora

This example utilizes a subset of [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) for training. And the `export_merged_model.py` is ported from [alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py).

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==4.45.0 "trl<0.12.0" datasets
pip install peft==0.10.0
pip install bitsandbytes==0.45.1 scipy
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
{'loss': 3.1898, 'learning_rate': 2e-05, 'epoch': 0.02}
{'loss': 3.1854, 'learning_rate': 1.7777777777777777e-05, 'epoch': 0.03}
{'loss': 3.0359, 'learning_rate': 1.555555555555556e-05, 'epoch': 0.05}
{'loss': 2.9661, 'learning_rate': 1.3333333333333333e-05, 'epoch': 0.06}
{'loss': 2.7779, 'learning_rate': 1.1111111111111113e-05, 'epoch': 0.08}
{'loss': 2.7795, 'learning_rate': 8.888888888888888e-06, 'epoch': 0.09}
{'loss': 2.5149, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.11}
{'loss': 2.5759, 'learning_rate': 4.444444444444444e-06, 'epoch': 0.12}
{'loss': 2.5976, 'learning_rate': 2.222222222222222e-06, 'epoch': 0.14}
{'loss': 2.5744, 'learning_rate': 0.0, 'epoch': 0.15}
{'train_runtime': 116.1914, 'train_samples_per_second': 6.885, 'train_steps_per_second': 1.721, 'train_loss': 2.819730052947998, 'epoch': 0.15}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:56<00:00,  1.72it/s]
TrainOutput(global_step=200, training_loss=2.819730052947998, metrics={'train_runtime': 116.1914, 'train_samples_per_second': 6.885, 'train_steps_per_second': 1.721, 'train_loss': 2.819730052947998, 'epoch': 0.15})
```

### 4. Merge the adapter into the original model

```
python ./export_merged_model.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --adapter_path ./outputs/checkpoint-200 --output_path ./outputs/checkpoint-200-merged
```

Then you can use `./outputs/checkpoint-200-merged` as a normal huggingface transformer model to do inference.
