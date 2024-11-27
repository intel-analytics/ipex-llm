# InternLM2
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on InternLM2 models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) as a reference InternLM model.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a InternLM2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==3.36.2
pip install huggingface_hub 
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install transformers==3.36.2
pip install huggingface_hub 
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
Setup local MODEL_PATH and run python code to download the right version of model from hugginface.
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id=repo_id, local_dir=MODEL_PATH, local_dir_use_symlinks=False, revision="v1.1.0")
```
Then run the example with the downloaded model
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the InternLM2 model (e.g. `internlm/internlm2-chat-7b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'internlm/internlm2-chat-7b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|User|>:AI是什么？
<|Bot|>:
-------------------- Output --------------------
<|User|>:AI是什么？
<|Bot|>:AI是人工智能的缩写，是计算机科学的一个分支，旨在使计算机能够像人类一样思考、学习和执行任务。AI技术包括机器学习、自然
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|User|>:What is AI?
<|Bot|>:
-------------------- Output --------------------
<|User|>:What is AI?
<|Bot|>:AI is the ability of machines to perform tasks that would normally require human intelligence, such as perception, reasoning, learning, and decision-making. AI is made possible
```
