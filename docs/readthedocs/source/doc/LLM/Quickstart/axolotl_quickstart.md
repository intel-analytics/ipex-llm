# Finetune LLM with Axolotl on Intel GPU without coding

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) is a popular tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures. You can now use [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `Axolotl` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of finetuning LLaMA2-7B on Intel Arc GPU below.

<video src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.mp4" width="100%" controls></video>

## Quickstart

IPEX-LLM's support for `Axolotl` now is available for Linux system. For Linux system, we recommend Ubuntu 20.04 or later (Ubuntu 22.04 is preferred).

### 1 Install IPEX-LLM for Axolotl

Visit the [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html), follow [Install Intel GPU Driver](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-intel-gpu-driver) and [Install oneAPI](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-oneapi) to install GPU driver and Intel® oneAPI Base Toolkit 2024.0.
 
Create new conda env, and install `ipex-llm[xpu]` and [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0).

```cmd
conda create -n axolotl python=3.11
conda activate axolotl
# install ipex-llm
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# install axolotl v0.4.0
git clone https://github.com/OpenAccess-AI-Collective/axolotl/tree/v0.4.0
cd axolotl
remove requirements.txt
wget -O requirements.txt https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/axolotl/requirements-xpu.txt
pip install -e .
# prepare axolotl entrypoints
wget https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/axolotl/finetune.py
wget https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/axolotl/train.py
```

**After the installation, you should have created a conda environment, named `axolotl` for instance, for running `Axolotl` commands with IPEX-LLM.**

### 2. Finetune Llama-2-7B with axolotl

| Model | Dataset | Finetune method |
|-------|-------|-------|
| Llama-2-7B | alpaca_2k_test | LoRA (Low-Rank Adaptation)  |
| Llama-2-7B | alpaca_2k_test | QLoRA (Quantized Low-Rank Adaptation) |


#### 2.1 Download Llama-2-7B and alpaca_2k_test

By default, Axolotl will automatically download models and datasets from Huggingface. Please ensure you have login to Huggingface.

```bash
huggingface-cli login
```

If you prefer to use offline model and datasets, please download [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b) and [alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test). Then, set `HF_HUB_OFFLINE=1` to avoid connecting to Huggingface.

```bash
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

```bash
accelerate config
```

Please answer `NO` in option `Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:`.

After finish accelerate config, check if `use_cpu` is disable (i.e., `use_cpu: false`) in accelerate config file (`~/.cache/huggingface/accelerate/default_config.yaml`).

#### 3.1 LoRA

Prepare `lora.yml` for finetuning. You can download a template from github.

```bash
wget https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/axolotl/lora.yml
```

Modify key parameters, such as `base_model` and `datasets.path`.

```yaml
# Please change to local path if model is offline
base_model: NousResearch/Llama-2-7b-hf
datasets:
  # Please change to local path if dataset is offline
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
```

Launch LoRA training with following command,
```bash
accelerate launch finetune.py lora.yml
```

In Axolotl v0.4.0, you can also use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```bash
accelerate launch train.py lora.yml
```

#### 3.2 QLoRA

Prepare `lora.yml` for finetuning. You can download a template from github.

```bash
wget https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/LLM-Finetuning/axolotl/qlora.yml
```

Modify key parameters, such as `base_model` and `datasets.path`.

```yaml
# Please change to local path if model is offline
base_model: NousResearch/Llama-2-7b-hf
datasets:
  # Please change to local path if dataset is offline
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
```

Launch LoRA training with following command,
```bash
accelerate launch finetune.py qlora.yml
```

In Axolotl v0.4.0, you can also use `train.py` instead of `-m axolotl.cli.train` or `finetune.py`.

```bash
accelerate launch train.py qlora.yml
```

### Troubleshooting

#### 

