# Finetune LLM on Intel GPU using axolotl v0.4.0 without writing code

This example demonstrates how to easily run LLM finetuning application using [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0) and IPEX-LLM 4bit optimizations with [Intel GPUs](../../../README.md). By applying IPEX-LLM patch, you could use axolotl on Intel GPUs using IPEX-LLM optimization without writing code.

Note, this example is just used for illustrating related usage and don't guarantee convergence of training.

### 0. Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../README.md#requirements) for more information.

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# install axolotl v0.4.0
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout v0.4.0
cp ../requirements-xpu.txt requirements.txt
pip install -e .
pip install transformers==4.36.0
# to avoid https://github.com/OpenAccess-AI-Collective/axolotl/issues/1544
pip install datasets==2.15.0
```

### 2. Configures OneAPI environment variables and accelerate

#### 2.1 Configures OneAPI environment variables 

```bash
source /opt/intel/oneapi/setvars.sh
```

#### 2.2 Configures `accelerate` in command line interactively. 

```bash
accelerate config
```

Please answer `NO` in option `Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:`.

After finish accelerate config, check if `use_cpu` is disable (i.e., ` use_cpu: false`) in accelerate config file (`~/.cache/huggingface/accelerate/default_config.yaml`).

#### 2.3 (Optional) Set ` HF_HUB_OFFLINE=1` to avoid huggingface hug signing.

```bash
export  HF_HUB_OFFLINE=1
```

For more details, please refer [hfhuboffline](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhuboffline).

### 3. Finetune Llama-2-7B

This example shows how to run [Alpaca LoRA training](https://github.com/tloen/alpaca-lora/tree/main) and [Alpaca QLoRA finetune](https://github.com/artidoro/qlora) directly on Intel GPU. Note that only Llama-2-7B LoRA and QLoRA examples are verified on Intel ARC 770 with 16GB memory.

#### 3.1 Alpaca LoRA

Based on [axolotl Llama-2 LoRA example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/v0.4.0/examples/llama-2/lora.yml).

```
accelerate launch finetune.py lora.yml
```

In v0.4.0, you can also use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```
accelerate launch train.py lora.yml
```

#### 3.2 Alpaca QLoRA

Based on [axolotl Llama-2 QLoRA example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/v0.4.0/examples/llama-2/qlora.yml).

Modify parameters in `qlora.yml` based on your requirements. Then, launch finetuning with the following command.

```
accelerate launch finetune.py qlora.yml
```

In v0.4.0, you can also use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```
accelerate launch train.py qlora.yml
```

#### 3.3 Expected Output

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
