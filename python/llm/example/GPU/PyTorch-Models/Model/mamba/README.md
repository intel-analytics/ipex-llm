# Mamba
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate Mamba models. For illustration purposes, we utilize the [state-spaces/mamba-1.4b](https://huggingface.co/state-spaces/mamba-1.4b) and [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b) as reference Mamba models.

## Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Mamba model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install einops # package required by Mamba
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install einops # package required by Mamba
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
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

<details>

<summary>For Intel iGPU</summary>

```bash
export SYCL_CACHE_PERSISTENT=1
```

</details>

#### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU and Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>


> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
### 4. Running examples

```bash
python ./generate.py
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Mamba model (e.g `state-spaces/mamba-1.4b` and `state-spaces/mamba-2.8b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `state-spaces/mamba-1.4b`.
- `--tokenizer-repo-id-or-path`: argument defining the huggingface repo id for the tokenizer of Mamba model to be downloaded, or the path to the huggingface checkpoint folder.  It is default to be `EleutherAI/gpt-neox-20b`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 4.3 Sample Output
#### [state-spaces/mamba-1.4b](https://huggingface.co/state-spaces/mamba-1.4b)
```log
Inference time: xxxx s
-------------------- Output --------------------
What is AI?

Artificial Intelligence (AI) is a broad term that describes the use of artificial intelligence (AI) to create artificial intelligence (AI). AI is a
```

#### [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b)
```log
Inference time: xxxx s
-------------------- Output --------------------
What is AI?

Artificial Intelligence is a field of study that focuses on creating machines that can perform intelligent functions. These functions are performed by machines that are smarter than humans
```
