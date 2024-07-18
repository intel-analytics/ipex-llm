# CoHere/command-r
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on cohere models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01) as a reference model.
> **Note**: Because the size of this cohere model is 35B, even running low_bit `sym_int4` still requires about 17.5GB. So currently it can only be run on MAX GPU, or run with [Pipeline-Parallel-Inference](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Pipeline-Parallel-Inference) on multiple Arc GPUs.
>
> Please select the appropriate size of the cohere model based on the capabilities of your machine.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a cohere model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install "transformers>=4.40.0"
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install "transformers>=4.40.0"
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

### 4. Running examples

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the cohere model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'CohereForAI/c4ai-command-r-v01'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
```log
Inference time: xxxxx s
-------------------- Prompt --------------------

<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>What is AI?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

-------------------- Output --------------------

<|START_OF_TURN_TOKEN|><|USER_TOKEN|>What is AI?<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
Artificial Intelligence Quora User,

Artificial Intelligence (AI) is the simulation of human intelligence in machines, typically by machines, that have become a very complex
```
