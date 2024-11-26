# Chatglm3-32k
In this directory, you will find examples on how you could apply IPEX-LLM INT4/FP8 optimizations on Chatglm3-32K models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) as reference Chatglm3-32K models.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Chatglm3 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4/FP8 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
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
#### 4.1 Using simple prompt
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT --low-bit LOW_BIT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Chatglm3 model (e.g. `THUDM/chatglm3-6b-32k`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/chatglm3-6b-32k'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--low-bit LOW_BIT`: argument defining which low bit optimization to use. Options are sym_int4 or fp8. It is default to be `sym_int4`.

#### 4.2 Using long context input prompt
You can set the `prompt` argument to be a `.txt` file path containing the long context prompt text. An example command using the 8k input size prompt we provide is given below:
```
python ./generate.py --repo-id-or-model-path togethercomputer/chatglm3-6b-32k --prompt 8k.txt
```
> Note: If you need to run longer input or use less memory, please set `IPEX_LLM_LOW_MEM=1`, which will enable memory optimization and may slightly affect the latency performance.
#### Sample Output
#### [THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|user|>
What is AI?
<|assistant|>
-------------------- Output --------------------
[gMASK]sop <|user|>
What is AI?
<|assistant|>
 AI stands for Artificial Intelligence. It refers to the ability of computers and other machines to perform tasks that typically require human intelligence, such as recognizing patterns, making
```
