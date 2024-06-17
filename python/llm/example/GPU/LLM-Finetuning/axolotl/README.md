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

You can download a default `default_config.yaml` with `use_cpu: false`.

```bash
mkdir -p  ~/.cache/huggingface/accelerate/
wget -O ~/.cache/huggingface/accelerate/default_config.yaml https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/default_config.yaml
```

As an alternative, you can config accelerate based on your requirements.

```bash
accelerate config
```

Please answer `NO` in option `Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:`.

After finish accelerate config, check if `use_cpu` is disable (i.e., `use_cpu: false`) in accelerate config file (`~/.cache/huggingface/accelerate/default_config.yaml`).

#### 2.3 (Optional) Set ` HF_HUB_OFFLINE=1` to avoid huggingface hug signing.

```bash
export  HF_HUB_OFFLINE=1
```

For more details, please refer [hfhuboffline](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhuboffline).

### 3. Finetune Llama-2-7B

This example shows how to run [Alpaca LoRA training](https://github.com/tloen/alpaca-lora/tree/main) and [Alpaca QLoRA finetune](https://github.com/artidoro/qlora) directly on Intel GPU. Note that only Llama-2-7B LoRA and QLoRA examples are verified on Intel ARC 770 with 16GB memory.

#### 3.1 Alpaca LoRA

Based on [axolotl Llama-2 LoRA example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/v0.4.0/examples/llama-2/lora.yml).

```bash
accelerate launch finetune.py lora.yml
```

In v0.4.0, you can also use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```bash
accelerate launch train.py lora.yml
```

#### 3.2 Alpaca QLoRA

Based on [axolotl Llama-2 QLoRA example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/v0.4.0/examples/llama-2/qlora.yml).

Modify parameters in `qlora.yml` based on your requirements. Then, launch finetuning with the following command.

```bash
accelerate launch finetune.py qlora.yml
```

In v0.4.0, you can also use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```bash
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

### 4. Finetune Llama-3-8B (Experimental)

Warning: this section will install axolotl main ([796a085](https://github.com/OpenAccess-AI-Collective/axolotl/tree/796a085b2f688f4a5efe249d95f53ff6833bf009)) for new features, e.g., Llama-3-8B.

#### 4.1 Install Axolotl main in conda

Axolotl main has lots of new dependencies. Please setup a new conda env for this version.

```bash
conda create -n llm python=3.11
conda activate llm
# install axolotl main
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl && git checkout 796a085
pip install -e .
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# install transformers etc
# to avoid https://github.com/OpenAccess-AI-Collective/axolotl/issues/1544
pip install datasets==2.15.0
pip install transformers==4.37.0
```

Config accelerate and oneAPIs, according to [Configures OneAPI environment variables and accelerate](#2-configures-oneapi-environment-variables-and-accelerate).

#### 4.2 Alpaca QLoRA

Based on [axolotl Llama-3 QLoRA example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora.yml).

Modify parameters in `llama3-qlora.yml` based on your requirements. Then, launch finetuning with the following command.

```bash
accelerate launch finetune.py llama3-qlora.yml
```

You can also use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```bash
accelerate launch train.py llama3-qlora.yml
```

Expected output

```bash
{'loss': 0.237, 'learning_rate': 1.2254711850265387e-06, 'epoch': 3.77}
{'loss': 0.6068, 'learning_rate': 1.1692453482951115e-06, 'epoch': 3.77}
{'loss': 0.2926, 'learning_rate': 1.1143322458989303e-06, 'epoch': 3.78}
{'loss': 0.2475, 'learning_rate': 1.0607326072295087e-06, 'epoch': 3.78}
{'loss': 0.1531, 'learning_rate': 1.008447144232094e-06, 'epoch': 3.79}
{'loss': 0.1799, 'learning_rate': 9.57476551396197e-07, 'epoch': 3.79}
{'loss': 0.2724, 'learning_rate': 9.078215057463868e-07, 'epoch': 3.79}
{'loss': 0.2534, 'learning_rate': 8.594826668332445e-07, 'epoch': 3.8}
{'loss': 0.3388, 'learning_rate': 8.124606767246579e-07, 'epoch': 3.8}
{'loss': 0.3867, 'learning_rate': 7.667561599972505e-07, 'epoch': 3.81}
{'loss': 0.2108, 'learning_rate': 7.223697237281668e-07, 'epoch': 3.81}
{'loss': 0.0792, 'learning_rate': 6.793019574868775e-07, 'epoch': 3.82}
```
