# Run IPEX-LLM on Intel NPU

This guide demonstrates:

- How to install IPEX-LLM for Intel NPU on Intel Core™ Ultra Processors
- Python and C++ APIs for running IPEX-LLM on Intel NPU

> [!IMPORTANT]
> IPEX-LLM currently only supports Windows on Intel NPU.

## Table of Contents

- [Install Prerequisites](#install-prerequisites)
- [Install `ipex-llm` with NPU Support](#install-ipex-llm-with-npu-support)
- [Runtime Configurations](#runtime-configurations)
- [Python API](#python-api)
- [C++ API](#c-api)
- [llama.cpp Support](#experimental-llamacpp-support)
- [Accuracy Tuning](#accuracy-tuning)

## Install Prerequisites

### Update NPU Driver

> [!IMPORTANT]
> It is highly recommended to update your NPU driver to `32.0.100.3104`, which has been thoroughly verified.

To update driver for Intel NPU:

1. Download the NPU driver

   - Visit the [official Intel NPU driver page for Windows](https://www.intel.com/content/www/us/en/download/794734/838895/intel-npu-driver-windows.html) and download the driver zip file.
   - Extract the driver zip file

2. Install the driver

   - Open **Device Manager** and locate **Neural processors** -> **Intel(R) AI Boost** in the device list
   - Right-click on **Intel(R) AI Boost** and select **Update driver**
   - Choose **Browse my computer for drivers**, navigate to the folder where you extracted the driver zip file, and select **Next**
   - Wait for the installation finished

3. (Optional) Uninstall and update the driver

   Please skip this if you have successfully installed. This is required when the driver to be installed is lower than current version.

   - Open **Device Manager** and locate **Neural processors** -> **Intel(R) AI Boost** in the device list
   - Right-click on **Intel(R) AI Boost** and select **Uninstall driver**
   - Choose **Attempt to remove the driver for this device** and select **Uninstall**
   - Locate and click **Add Drivers** in the toolbar, choose the folder where you extracted the driver zip file, and select **Next**
   - Wait for the installation finished, locate and click **Scan for hardware changes** in the toolbar

A system reboot is necessary to apply the changes after the installation is complete.

### (Optional) Install Visual Studio 2022

> [!NOTE]
> To use IPEX-LLM **C++ API** on Intel NPU, you are required to install Visual Studio 2022 on your system. If you plan to use the **Python API**, skip this step.

Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) Community Edition and select "Desktop development with C++" workload:

<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_1.png"  width=80%/>
</div>

### Setup Python Environment

Visit [Miniforge installation page](https://conda-forge.org/download/), download the **Miniforge installer for Windows**, and follow the instructions to complete the installation.

<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_miniforge_download.png"  width=80%/>
</div>

After installation, open the **Miniforge Prompt**, create a new python environment `llm-npu`:
```cmd
conda create -n llm-npu python=3.11
```
Activate the newly created environment `llm-npu`:
```cmd
conda activate llm-npu
```

> [!TIP]
> `ipex-llm` for NPU supports Python 3.10 and 3.11.

### (Optional) Install CMake

> [!NOTE]
>  Cmake installation is for IPEX-LLM **C++ API** on Intel NPU. If you plan to use the **Python API**, skip this step.

With the `llm-npu` environment active, install CMake:

```cmd
conda activate llm-npu

pip install cmake
```

## Install `ipex-llm` with NPU Support

With the `llm-npu` environment active, use `pip` to install `ipex-llm` for NPU:

```cmd
conda activate llm-npu

pip install --pre --upgrade ipex-llm[npu]
```

## Runtime Configurations

For `ipex-llm` NPU support, please set the following environment variable with active `llm-npu` environment based on your device:

- For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxV (code name Lunar Lake)**:

  - For Intel Core™ Ultra 7 Processor 258V:

      No runtime configuration required.

  - For Intel Core™ Ultra 5 Processor 228V & 226V:
      ```cmd
      set IPEX_LLM_NPU_DISABLE_COMPILE_OPT=1
      ```

- For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxK or 2xxH (code name Arrow Lake)**:
   ```cmd
   set IPEX_LLM_NPU_ARL=1
   ```

- For **Intel Core™ Ultra Processors (Series 1) with processor number 1xxH (code name Meteor Lake)**:

   ```cmd
   set IPEX_LLM_NPU_MTL=1
   ```

## Python API

IPEX-LLM offers Hugging Face `transformers`-like Python API, enabling seamless running of Hugging Face transformers models on Intel NPU.

Refer to the following table for examples of verified models:
[](../../../python/llm/)
| Model | Model link | Example link | Verified Platforms |
|:--|:--|:--|:--|
| LLaMA 2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| LLaMA 3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| LLaMA 3.2 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| GLM-Edge | [THUDM/glm-edge-1.5b-chat](https://huggingface.co/THUDM/glm-edge-1.5b-chat), [THUDM/glm-edge-4b-chat](https://huggingface.co/THUDM/glm-edge-4b-chat)  | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| Qwen 2 | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct), [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| Qwen 2.5 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Lunar Lake |
|  | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| DeepSeek-R1 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| MiniCPM | [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16), [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Meteor Lake, Lunar Lake, Arrow Lake |
| Baichuan 2 | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/) | Lunar Lake |
| MiniCPM-Llama3-V-2_5 | [openbmb/MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal/) | Lunar Lake |
| MiniCPM-V-2_6 | [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal/) | Lunar Lake |
| Speech_Paraformer-Large | [iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/Multimodal/) | Lunar Lake |
| Bce-Embedding-Base-V1 | [maidalun1020/bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/Embedding/) | Lunar Lake |


> [!TIP]
> You could refer to [here](../../../python/llm/example/NPU/HF-Transformers-AutoModels) for full IPEX-LLM examples on Intel NPU.

### Save & Load Low-Bit Models

IPEX-LLM also provides Python API for saving/loading models with low-bit optimizations on Intel NPU, to avoid repeated loading & optimizing of the original models. Refer to the [Save-Load example](../../../python/llm/example/NPU/HF-Transformers-AutoModels/Save-Load) for usage in details.

## C++ API

IPEX-LLM also provides C++ API for running Hugging Face `transformers` models.

Refer to the following table for examples of verified models:

| Model | Model link | Example link | Verified Platforms |
|:--|:--|:--|:--|
| LLaMA 2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Meteor Lake, Lunar Lake, Arrow Lake |
| LLaMA 3 | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Meteor Lake, Lunar Lake, Arrow Lake |
| LLaMA 3.2 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Meteor Lake, Lunar Lake, Arrow Lake |
| Qwen 2 | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct), [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Meteor Lake, Lunar Lake, Arrow Lake |
| Qwen 2.5 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Lunar Lake |
|  | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Meteor Lake, Lunar Lake, Arrow Lake |
| DeepSeek-R1 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Meteor Lake, Lunar Lake, Arrow Lake |
| MiniCPM | [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16), [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) | [link](../../../python/llm/example/NPU/HF-Transformers-AutoModels/LLM/CPP_Examples) | Meteor Lake, Lunar Lake, Arrow Lake |

> [!TIP]
> You could refer to [here](../../../python/llm/example/NPU/HF-Transformers-AutoModels) for full IPEX-LLM examples on Intel NPU.

## (Experimental) llama.cpp Support

IPEX-LLM provides `llama.cpp` compatible API for running GGUF models on Intel NPU.

Refer to the following table for verified models:

| Model | Model link | Verified Platforms |
|:--|:--|:--|
| LLaMA 3.2 | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Meteor Lake, Lunar Lake, Arrow Lake |
| DeepSeek-R1 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | Meteor Lake, Lunar Lake, Arrow Lake |

> [!Important]
>
> You may use [llama.cpp portable zip](./llama_cpp_npu_portable_zip_quickstart.md) to directly run GGUF models on Intel NPU with ipex-llm (***without the need of manual installations***).

### Run GGUF model using CLI tool
#### Setup for running llama.cpp

First, you should create a directory to use `llama.cpp`, for instance, use following command to create a `llama-cpp-npu` directory and enter it.

```cmd
mkdir llama-cpp-npu
cd llama-cpp-npu
```

Then, please run the following command with **administrator privilege in Miniforge Prompt** to initialize `llama.cpp` for NPU:

```cmd
init-llama-cpp.bat
```

#### Model Download

Before running, you should download or copy community GGUF model to your current directory. For instance,  `DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf` of [DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/tree/main).

#### Run the quantized model

Please refer to [Runtime Configurations](#runtime-configurations) before running the following command in Miniforge Prompt.

```cmd
llama-cli-npu.exe -m DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf -n 32 --prompt "What is AI?"
```

And you could use `llama-cli-npu.exe -h` for more details about meaning of each parameter.

### Run GGUF model using llama.cpp C++ API

IPEX-LLM also supports `llama.cpp` C++ API for running GGUF models on Intel NPU. Refer to [Simple Example](../../../python/llm/example/NPU/llama.cpp/) for usage in details.

> **Note**:
>
> - **Warmup on first run**: When running specific GGUF models on NPU for the first time, you might notice delays up to several minutes before the first token is generated. This delay occurs because the blob compilation.

## Accuracy Tuning

IPEX-LLM provides several optimization methods for enhancing the accuracy of model outputs on Intel NPU. You can select and combine these techniques to achieve better outputs based on your specific use case.

### 1. `IPEX_LLM_NPU_QUANTIZATION_OPT` Env

You could set environment variable `IPEX_LLM_NPU_QUANTIZATION_OPT=1` before loading & optimizing the model with `from_pretrained` function from `ipex_llm.transformers.npu_model` Auto Model class to further enhance model accuracy of low-bit models.

### 2. Low-Bit Optimizations

IPEX-LLM on Intel NPU currently supports `sym_int4`/`asym_int4`/`sym_int8` low-bit optimizations. You could adjust the low-bit value to tune the accuracy. 

For example, you could try to set `load_in_low_bit='asym_int4'` instead of `load_in_low_bit='sym_int4'` when loading & optimizing the model with `from_pretrained` function from `ipex_llm.transformers.npu_model` Auto Model class, to switch from `sym_int4` low-bit optimizations to `asym_int4`.

### 3. Mixed Precision

When loading & optimizing the model with `from_pretrained` function of `ipex_llm.transformers.npu_model` Auto Model class, you could try to set parameter `mixed_precision=True` to enable mixed precision optimization when encountering output problems.

### 4. Group Size

IPEX-LLM low-bit optimizations support both channel-wise and group-wise quantization on Intel NPU. When loading & optimizing the model with `from_pretrained` function of Auto Model class from `ipex_llm.transformers.npu_model`, parameter `quantization_group_size` will control whether to use channel-wise or group-wise quantization.

If setting `quantization_group_size=0`, IPEX-LLM will use channel-wise quantization. If setting `quantization_group_size=128`, IPEX-LLM will use group-wise quantization with group size 128.

You could try to use group-wise quantization for better outputs.