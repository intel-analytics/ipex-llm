# Reward Modeling Finetuning for Sequence Classfication with IPEX-LLM

This is an example of IPEX-LLM [reward modeling](https://huggingface.co/docs/trl/main/en/reward_trainer) (a kind of RLHF) on [Intel GPU](../../../README.md), which refers [TRL example](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py) to tune model [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) on a sequence classfication task.

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
After starting, you can configure where wanda report or directly disable it by select no-visualization option.

### 3. Sample Output
```log
......
{'loss': 0.8613, 'grad_norm': 2.837268590927124, 'learning_rate': 1.3854569190600522e-05, 'epoch': 0.02}
{'eval_loss': 0.8356835246086121, 'eval_accuracy': 0.4996802660186725, 'eval_runtime': xxxx, 'eval_samples_per_second': xxxx, 'eval_steps_per_second': xxxx, 'epoch': 0.03}
  4%|██▊                                                                          | 42/1149 [xx:xx<xx:xx:xx, xx.xx s/it]
......
```
