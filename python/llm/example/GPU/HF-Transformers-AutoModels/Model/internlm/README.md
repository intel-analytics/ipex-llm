# InternLM
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on InternLM models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [internlm/internlm-chat-7b-8k](https://huggingface.co/internlm/internlm-chat-7b-8k) as a reference InternLM model.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a InternLM model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9 libuv
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Configures OneAPI environment variables
#### 2.1 Configurations for Linux
```bash
source /opt/intel/oneapi/setvars.sh
```

#### 2.2 Configurations for Windows
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
> Note: Please make sure you are using **CMD** (**Anaconda Prompt** if using conda) to run the command as PowerShell is not supported.
### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
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

<summary>For Intel Arc™ A300-Series or Pro A60</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For other Intel dGPU Series</summary>

There is no need to set further environment variables.

</details>

> Note: For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
### 4. Running examples

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the InternLM model (e.g. `internlm/internlm-chat-7b-8k`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'internlm/internlm-chat-7b-8k'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [internlm/internlm-chat-7b-8k](https://huggingface.co/internlm/internlm-chat-7b-8k)
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
