# Finetune LLM on Intel GPU using axolotl without writing code

This example demonstrates how to easily run LLM finetuning application using axolotl and IPEX-LLM 4bit optimizations with [Intel GPUs](../../../README.md). By applying IPEX-LLM patch, you could use axolotl on Intel GPUs using IPEX-LLM optimization without writing code.

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
# install axolotl v0.3.0
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout v0.3.0
# replace default requirements.txt in axolotl to avoid conflict
cp ../requirements.txt .
pip install -e .
# change to transformers 4.34.0
pip install transformers==4.34.0 datasets
```

### 2. Configures OneAPI environment variables and accelerate

```bash
source /opt/intel/oneapi/setvars.sh
```

Config `accelerate`

```bash
accelerate config
```

Ensure `use_cpu` is disable in config (`~/.cache/huggingface/accelerate/default_config.yaml`).

### 3. Finetune

This example shows how to run [Alpaca QLoRA finetune on Llama-2](https://github.com/artidoro/qlora) directly on Intel GPU, based on [axolotl Llama-2 qlora example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/v0.3.0/examples/llama-2/qlora.yml).

Modify parameters in `qlora.yml` based on your requirements.

```
accelerate launch finetune.py qlora.yml
```

Output in console

```
{'eval_loss': 0.9382301568984985, 'eval_runtime': 6.2513, 'eval_samples_per_second': 3.199, 'eval_steps_per_second': 3.199, 'epoch': 0.36}
{'loss': 0.944, 'learning_rate': 0.00019752490425051743, 'epoch': 0.38}
{'loss': 1.0179, 'learning_rate': 0.00019705675197106016, 'epoch': 0.4}
{'loss': 0.9346, 'learning_rate': 0.00019654872959986937, 'epoch': 0.41}
{'loss': 0.9747, 'learning_rate': 0.0001960010458282326, 'epoch': 0.43}
{'loss': 0.8928, 'learning_rate': 0.00019541392564000488, 'epoch': 0.45}
{'loss': 0.9317, 'learning_rate': 0.00019478761021918728, 'epoch': 0.47}
{'loss': 1.0534, 'learning_rate': 0.00019412235685085035, 'epoch': 0.49}
{'loss': 0.8777, 'learning_rate': 0.00019341843881544372, 'epoch': 0.5}
{'loss': 0.9447, 'learning_rate': 0.00019267614527653488, 'epoch': 0.52}
{'loss': 0.9651, 'learning_rate': 0.00019189578116202307, 'epoch': 0.54}
{'loss': 0.9067, 'learning_rate': 0.00019107766703887764, 'epoch': 0.56}
```

### 4. Other examples

Please refer to [axolotl examples](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.3.0/examples) for more models. Download `xxx.yml` and replace `qlora.yml` with new `xxx.yml`.

```
accelerate launch finetune.py xxx.yml
```
