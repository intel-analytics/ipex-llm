# MiniCPM3
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on MiniCPM3 models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B) (or [OpenBMB/MiniCPM3-4B](https://www.modelscope.cn/models/OpenBMB/MiniCPM3-4B) for ModelScope) as a reference MiniCPM3 model.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a MiniCPM3 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install jsonschema datamodel_code_generator

# [optional] only needed if you would like to use ModelScope as model hub
pip install modelscope==1.11.0
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install jsonschema datamodel_code_generator

# [optional] only needed if you would like to use ModelScope as model hub
pip install modelscope==1.11.0
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
# for Hugging Face model hub
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT

# for ModelScope model hub
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT --modelscope
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the **Hugging Face** (e.g. `openbmb/MiniCPM3-4B`) or **ModelScope** (e.g. `OpenBMB/MiniCPM3-4B`) repo id for the MiniCPM3 model to be downloaded, or the path to the checkpoint folder. It is default to be `'openbmb/MiniCPM3-4B'` for **Hugging Face** or `'OpenBMB/MiniCPM3-4B'` for **ModelScope**.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--modelscope`: using **ModelScope** as model hub instead of **Hugging Face**.

#### Sample Output
#### [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|im_start|>user
AI是什么?<|im_end|>
<|im_start|>assistant

-------------------- Output --------------------
<s><|im_start|> user
AI是什么?<|im_end|>
<|im_start|> assistant
AI，即人工智能（Artificial Intelligence），是指由人类创造的、能够模拟人类智能的相关理论和实践的一门新兴技术。它使计算机 或其他
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant

-------------------- Output --------------------
<s><|im_start|> user
What is AI?<|im_end|>
<|im_start|> assistant
AI, or Artificial Intelligence, is a field of computer science that emphasizes the creation of intelligent machines capable of performing tasks that typically require human intelligence. These tasks include
```
