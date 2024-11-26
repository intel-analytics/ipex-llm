# Yuan2
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Yuan2 models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf) as a reference Yuan2 model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

In addition, you need to modify some files in Yuan2-2B-hf folder, since Flash attention dependency is for CUDA usage and currently cannot be installed on Intel CPUs. To manually turn it off, please refer to [this issue](https://github.com/IEIT-Yuan/Yuan-2.0/issues/92). We also provide two modified files([config.json](yuan2-2B-instruct/config.json) and [yuan_hf_model.py](yuan2-2B-instruct/yuan_hf_model.py)), which can be used to replace the original content in config.json and yuan_hf_model.py.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an Yuan2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install einops # additional package required for Yuan2 to conduct generation
pip install pandas # additional package required for Yuan2 to conduct generation
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install einops # additional package required for Yuan2 to conduct generation
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

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Yuan2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'IEITYuan/Yuan2-2B-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `100`.

#### Sample Output
#### [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf)
```log
Inference time: xxxx seconds
-------------------- Output --------------------

What is AI?
AI is a field of technology and technologies that is used to analyze and improve human behavior such as language processing, machine learning and artificial intelligence (AI).<eod>
```