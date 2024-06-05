# Run IPEX-LLM on Multiple Intel GPUs in pipeline parallel fashion

This example demonstrates how to run IPEX-LLM optimized low-bit model vertically partitioned on two [Intel GPUs](../README.md) for Linux users.

## Requirements
To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

> [!NOTE]
> To run IPEX-LLM on multiple Intel GPUs in pipeline parallel fashion, you will need to install **Intel® oneAPI Base Toolkit 2024.1**, which could be done through an offline installer:
> ```bash
> wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/fdc7a2bc-b7a8-47eb-8876-de6201297144/l_BaseKit_p_2024.1.0.596_offline.sh
> 
> sudo sh ./l_BaseKit_p_2024.1.0.596_offline.sh
> ```

## Example:

### 1. Installation

```bash
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30+xpu oneccl_bind_pt==2.1.300+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Configures OneAPI environment variables

```bash
source /opt/intel/oneapi/setvars.sh
```

> [!NOTE]
> Please make sure you configure the environment variables for **oneAPI 2024.1**.

### 3 Runtime Configurations

For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.

<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> [!NOTE]
> Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

### 4. Running examples
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT --gpu-num GPU_NUM
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--gpu-num GPU_NUM`: argument defining the number of GPU to use. It is default to be `2`.

#### Sample Output
##### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<s>[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]
-------------------- Output --------------------
[INST] <<SYS>>

<</SYS>>

What is AI? [/INST]  Artificial intelligence (AI) is the broader field of research and development aimed at creating machines that can perform tasks that typically require human intelligence,
```