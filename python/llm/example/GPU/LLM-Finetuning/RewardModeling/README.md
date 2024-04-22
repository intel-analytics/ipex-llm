# Reward Modeling Finetuning with IPEX-LLM

This is an example of IPEX-LLM [reward modeling](https://huggingface.co/docs/trl/main/en/reward_trainer) (a kind of RLHF) on [Intel GPU](../../../README.md), which refers [TRL example](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py) to tune model [facebook/opt-350m](https://huggingface.co/facebook/opt-350m).

### 0. Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install accelerate==0.28.0
pip install bitsandbytes==0.43.0
pip install datasets==2.18.0
pip install transformers==4.39.1
pip install trl
pip install wandb
```

### 2. Reward Modeling Finetune

Here is how to fine-tune opt-350m on an Intel Max GPU server:

```bash
# arguments can be reset in the script e.g. model_name_or_path, per_device_train_batch_size and other hyperparameters
bash start-reward-modeling-finetuning.sh
```

### 3. Sample Output
```log
......
{'loss': 2.0989, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.0}
{'loss': 1.9064, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.0}
{'loss': 1.7483, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.9551, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.783, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.3328, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.4622, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.01}
{'loss': 1.9094, 'grad_norm': 0.0, 'learning_rate': 0.001, 'epoch': 0.02}
  5%|████▏                                                                                      | 70/1500 [xx:xx<x:xx:xx, xx.xxs/it]
......
```
