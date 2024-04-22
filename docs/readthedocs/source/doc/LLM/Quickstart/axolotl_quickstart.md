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

#### 2.1 Download Llama-2-7B and alpaca_2k_test

Huggingface hug


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


export HF_HUB_OFFLINE

#### 3.1 LoRA

#### 3.2 QLoRA


### Troubleshooting

#### 

