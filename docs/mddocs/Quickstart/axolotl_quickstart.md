# Finetune LLM with Axolotl on Intel GPU

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) is a popular tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures. You can now use [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `Axolotl` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of finetuning LLaMA2-7B on Intel Arc GPU below.

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/axolotl-qlora-linux-arc.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/axolotl-qlora-linux-arc.png"/></a></td>
  </tr>
  <tr>
    <td align="center">You could also click <a href="https://llm-assets.readthedocs.io/en/latest/_images/axolotl-qlora-linux-arc.mp4">here</a> to watch the demo video.</td>
  </tr>
</table>

## Table of Contents
- [Prerequisites](./axolotl_quickstart.md#0-prerequisites)
- [Install IPEX-LLM for Axolotl](./axolotl_quickstart.md#1-install-ipex-llm-for-axolotl)
- [Example: Finetune Llama-2-7B with Axolotl](./axolotl_quickstart.md#2-example-finetune-llama-2-7b-with-axolotl)
- [Finetune Llama-3-8B (Experimental)](./axolotl_quickstart.md#3-finetune-llama-3-8b-experimental)
- [Troubleshooting](./axolotl_quickstart.md#troubleshooting)



## Quickstart

### 0. Prerequisites

IPEX-LLM's support for [Axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0) is only available for Linux system. We recommend Ubuntu 20.04 or later (Ubuntu 22.04 is preferred).

Visit the [Install IPEX-LLM on Linux with Intel GPU](./install_linux_gpu.md), follow [Install Intel GPU Driver](./install_linux_gpu.md#install-gpu-driver) and [Install oneAPI](./install_linux_gpu.md#install-oneapi) to install GPU driver and Intel® oneAPI Base Toolkit 2024.0.

### 1. Install IPEX-LLM for Axolotl

Create a new conda env, and install `ipex-llm[xpu]`.

```bash
conda create -n axolotl python=3.11
conda activate axolotl
# install ipex-llm
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

Install [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0) from git.

```bash
# install axolotl v0.4.0
git clone https://github.com/OpenAccess-AI-Collective/axolotl -b v0.4.0
cd axolotl
# replace requirements.txt
rm requirements.txt
wget -O requirements.txt https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/requirements-xpu.txt
pip install -e .
pip install transformers==4.36.0
# to avoid https://github.com/OpenAccess-AI-Collective/axolotl/issues/1544
pip install datasets==2.15.0
# prepare axolotl entrypoints
wget https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/finetune.py
wget https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/train.py
```

**After the installation, you should have created a conda environment, named `axolotl` for instance, for running `Axolotl` commands with IPEX-LLM.**

### 2. Example: Finetune Llama-2-7B with Axolotl

The following example will introduce finetuning [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b) with [alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test) dataset using LoRA and QLoRA.

Note that you don't need to write any code in this example.

| Model | Dataset | Finetune method |
|-------|-------|-------|
| Llama-2-7B | alpaca_2k_test | LoRA (Low-Rank Adaptation)  |
| Llama-2-7B | alpaca_2k_test | QLoRA (Quantized Low-Rank Adaptation) |

For more technical details, please refer to [Llama 2](https://arxiv.org/abs/2307.09288), [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314).

#### 2.1 Download Llama-2-7B and alpaca_2k_test

By default, Axolotl will automatically download models and datasets from Huggingface. Please ensure you have login to Huggingface.

```bash
huggingface-cli login
```

If you prefer offline models and datasets, please download [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b) and [alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test). Then, set `HF_HUB_OFFLINE=1` to avoid connecting to Huggingface.

```bash
export HF_HUB_OFFLINE=1
```

#### 2.2 Set Environment Variables

> [!NOTE]
> This is a required step on for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

Configure oneAPI variables by running the following command:

```bash
source /opt/intel/oneapi/setvars.sh
```

Configure accelerate to avoid training with CPU. You can download a default `default_config.yaml` with `use_cpu: false`.

```bash
mkdir -p  ~/.cache/huggingface/accelerate/
wget -O ~/.cache/huggingface/accelerate/default_config.yaml https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/default_config.yaml
```

As an alternative, you can config accelerate based on your requirements.

```bash
accelerate config
```

Please answer `NO` in option `Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:`.

After finishing accelerate config, check if `use_cpu` is disabled (i.e., `use_cpu: false`) in accelerate config file (`~/.cache/huggingface/accelerate/default_config.yaml`).

#### 2.3 LoRA finetune

Prepare `lora.yml` for Axolotl LoRA finetune. You can download a template from github.

```bash
wget https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/lora.yml
```

**If you are using the offline model and dataset in local env**, please modify the model path and dataset path in `lora.yml`. Otherwise, keep them unchanged.

```yaml
# Please change to local path if model is offline, e.g., /path/to/model/Llama-2-7b-hf
base_model: NousResearch/Llama-2-7b-hf
datasets:
  # Please change to local path if dataset is offline, e.g., /path/to/dataset/alpaca_2k_test
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
```

Modify LoRA parameters, such as `lora_r` and `lora_alpha`, etc.

```yaml
adapter: lora
lora_model_dir:

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:
```

Launch LoRA training with the following command.

```bash
accelerate launch finetune.py lora.yml
```

In Axolotl v0.4.0, you can use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```bash
accelerate launch train.py lora.yml
```

#### 2.4 QLoRA finetune

Prepare `lora.yml` for QLoRA finetune. You can download a template from github.

```bash
wget https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/qlora.yml
```

**If you are using the offline model and dataset in local env**, please modify the model path and dataset path in `qlora.yml`. Otherwise, keep them unchanged.

```yaml
# Please change to local path if model is offline, e.g., /path/to/model/Llama-2-7b-hf
base_model: NousResearch/Llama-2-7b-hf
datasets:
  # Please change to local path if dataset is offline, e.g., /path/to/dataset/alpaca_2k_test
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
```

Modify QLoRA parameters, such as `lora_r` and `lora_alpha`, etc.

```yaml
adapter: qlora
lora_model_dir:

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:
```

Launch LoRA training with the following command.

```bash
accelerate launch finetune.py qlora.yml
```

In Axolotl v0.4.0, you can use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```bash
accelerate launch train.py qlora.yml
```

### 3. Finetune Llama-3-8B (Experimental)

Warning: this section will install axolotl main ([796a085](https://github.com/OpenAccess-AI-Collective/axolotl/tree/796a085b2f688f4a5efe249d95f53ff6833bf009)) for new features, e.g., Llama-3-8B.

#### 3.1 Install Axolotl main in conda

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

Config accelerate and oneAPIs, according to [Set Environment Variables](#22-set-environment-variables).

#### 3.2 Alpaca QLoRA

Based on [axolotl Llama-3 QLoRA example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora.yml).

Prepare `llama3-qlora.yml` for QLoRA finetune. You can download a template from github.

```bash
wget https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/LLM-Finetuning/axolotl/llama3-qlora.yml
```

**If you are using the offline model and dataset in local env**, please modify the model path and dataset path in `llama3-qlora.yml`. Otherwise, keep them unchanged.

```yaml
# Please change to local path if model is offline, e.g., /path/to/model/Meta-Llama-3-8B
base_model: meta-llama/Meta-Llama-3-8B
datasets:
  # Please change to local path if dataset is offline, e.g., /path/to/dataset/alpaca_2k_test
  - path: aaditya/alpaca_subset_1
    type: alpaca
```

Modify QLoRA parameters, such as `lora_r` and `lora_alpha`, etc.

```yaml
adapter: qlora
lora_model_dir:

sequence_len: 256
sample_packing: true
pad_to_sequence_len: true

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:
```

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

## Troubleshooting

### TypeError: PosixPath

Error message: `TypeError: argument of type 'PosixPath' is not iterable`

This issue is related to [axolotl #1544](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1544). It can be fixed by downgrading datasets to 2.15.0.

```bash
pip install datasets==2.15.0
```

### RuntimeError: out of device memory

Error message: `RuntimeError: Allocation is out of device memory on current platform.`

This issue is caused by running out of GPU memory. Please reduce `lora_r` or `micro_batch_size` in `qlora.yml` or `lora.yml`, or reduce data using in training.

### OSError: libmkl_intel_lp64.so.2

Error message: `OSError: libmkl_intel_lp64.so.2: cannot open shared object file: No such file or directory`

oneAPI environment is not correctly set. Please refer to [Set Environment Variables](#22-set-environment-variables).
