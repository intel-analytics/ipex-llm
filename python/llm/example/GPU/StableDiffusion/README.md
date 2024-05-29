# Stable Diffusion
In this directory, you will find examples on how you could use run StableDiffusion models on [Intel GPUs](../README.md).

### 1. Installation
#### 1.1 Installation on Linux
We suggest using conda to manage environment. 
```bash
conda create -n diffusion python=3.11
conda activate diffusion
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install diffusers["torch"] transformers
pip install -U PEFT transformers
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment. 
```bash
conda create -n diffusion python=3.11 libuv
conda activate diffusion
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install diffusers["torch"] transformers
pip install -U PEFT transformers
```

### 2. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
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
</details>

<details>

<summary>For Intel iGPU</summary>

```bash
export SYCL_CACHE_PERSISTENT=1
export BIGDL_LLM_XMX_DISABLED=1
```

</details>

#### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

</details>

<details>

<summary>For Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>

> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.

### 4. Examples

#### 4.1 StableDiffusion XL Example
The example shows how to run StableDiffusion XL example on Intel GPU.
```bash
python ./sdxl.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the stable diffusion xl model (e.g. `stabilityai/stable-diffusion-xl-base-1.0`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'stabilityai/stable-diffusion-xl-base-1.0'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'A lovely dog on the table, detailed, 8k'`.
- `--save-path`: argument defining the path to save the generated figure. It is default to be `sdxl-gpu.png`.
- `--num-steps`: argument defining the number of inference steps. It is default to be `20`. 

#### 4.2 LCM-LoRA Example
The example shows how to performing inference with LCM-LoRA on Intel GPU.
```bash
python ./lora-lcm.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the stable diffusion xl model (e.g. `stabilityai/stable-diffusion-xl-base-1.0`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'stabilityai/stable-diffusion-xl-base-1.0'`.
- `--lora-weights-path`: argument defining the huggingface repo id for the LCM-LoRA model (e.g. `latent-consistency/lcm-lora-sdxl`) to be downloaded, or the path to huggingface checkpoint folder. It is default to be `'latent-consistency/lcm-lora-sdxl'`. 
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'A lovely dog on the table, detailed, 8k'`.
- `--save-path`: argument defining the path to save the generated figure. It is default to be `lcm-lora-sdxl-gpu.png`.
- `--num-steps`: argument defining the number of inference steps. It is default to be `4`.
