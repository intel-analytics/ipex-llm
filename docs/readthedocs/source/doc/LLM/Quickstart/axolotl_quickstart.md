# Finetune LLM with Axolotl on Intel GPU

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) is a popular tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures. You can now use [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `Axolotl` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of finetuning LLaMA2-7B on Intel Arc GPU below.

<video src="https://llm-assets.readthedocs.io/en/latest/_images/axolotl-qlora-linux-arc.mp4" width="100%" controls></video>

## Quickstart

### 0. Prerequisites

IPEX-LLM's support for [Axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0) is only available for Linux system. We recommend Ubuntu 20.04 or later (Ubuntu 22.04 is preferred).

Visit the [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html), follow [Install Intel GPU Driver](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-intel-gpu-driver) and [Install oneAPI](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-oneapi) to install GPU driver and Intel® oneAPI Base Toolkit 2024.0.

### 1. Install IPEX-LLM for Axolotl

Create a new conda env, and install `ipex-llm[xpu]`.

```cmd
conda create -n axolotl python=3.11
conda activate axolotl
# install ipex-llm
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

Install [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0) from git.

```cmd
# install axolotl v0.4.0
git clone https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0
cd axolotl
# replace requirements.txt
remove requirements.txt
wget -O requirements.txt https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/axolotl/requirements-xpu.txt
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

```cmd
huggingface-cli login
```

If you prefer offline models and datasets, please download [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b) and [alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test). Then, set `HF_HUB_OFFLINE=1` to avoid connecting to Huggingface.

```cmd
export HF_HUB_OFFLINE=1
```

#### 2.2 Set Environment Variables

```eval_rst
.. note::

   This is a required step on for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.
```

Configure oneAPI variables by running the following command:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         source /opt/intel/oneapi/setvars.sh

```

Configure accelerate to avoid training with CPU

```cmd
accelerate config
```

Please answer `NO` in option `Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:`.

After finishing accelerate config, check if `use_cpu` is disabled (i.e., `use_cpu: false`) in accelerate config file (`~/.cache/huggingface/accelerate/default_config.yaml`).

#### 2.3 LoRA finetune

Prepare `lora.yml` for Axolotl LoRA finetune. You can download a template from github.

```cmd
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

lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:
```

Launch LoRA training with the following command.

```cmd
accelerate launch finetune.py lora.yml
```

In Axolotl v0.4.0, you can use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```cmd
accelerate launch train.py lora.yml
```

#### 2.4 QLoRA finetune

Prepare `lora.yml` for QLoRA finetune. You can download a template from github.

```cmd
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

lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:
```

Launch LoRA training with the following command.

```cmd
accelerate launch finetune.py qlora.yml
```

In Axolotl v0.4.0, you can use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```cmd
accelerate launch train.py qlora.yml
```

## Troubleshooting

#### TypeError: PosixPath

Error message: `TypeError: argument of type 'PosixPath' is not iterable`

This issue is related to [axolotl #1544](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1544). It can be fixed by downgrading datasets to 2.15.0.

```cmd
pip install datasets==2.15.0
```

#### RuntimeError: out of device memory

Error message: `RuntimeError: Allocation is out of device memory on current platform.`

This issue is caused by running out of GPU memory. Please reduce `lora_r` or `micro_batch_size` in `qlora.yml` or `lora.yml`, or reduce data using in training.

#### OSError: libmkl_intel_lp64.so.2

Error message: `OSError: libmkl_intel_lp64.so.2: cannot open shared object file: No such file or directory`

oneAPI environment is not correctly set. Please refer to [Set Environment Variables](#set-environment-variables).
