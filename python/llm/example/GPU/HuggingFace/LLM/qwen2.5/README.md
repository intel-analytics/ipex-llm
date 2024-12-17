# Qwen2.5
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Qwen2.5 models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) (or [Qwen/Qwen2.5-3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct), [Qwen/Qwen2.5-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) and [Qwen/Qwen2.5-14B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct) for ModelScope) as reference Qwen2.5 models.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Qwen2.5 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the **Hugging Face** or **ModelScope** repo id for the Qwen2.5 model (e.g. `Qwen/Qwen2.5-7B-Instruct`) to be downloaded, or the path to the checkpoint folder. It is default to be `'Qwen/Qwen2.5-7B-Instruct'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
- `--modelscope`: using **ModelScope** as model hub instead of **Hugging Face**.

#### Sample Output
##### [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
AI是什么？
-------------------- Output --------------------
AI是Artificial Intelligence的缩写，意为“人工智能”，是指由人制造出来的系统，能够进行类似于人类智慧的行为，如学习、推理
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
What is AI?
-------------------- Output --------------------
AI, or Artificial Intelligence, refers to the ability exhibited by machines to imitate human behavior and intelligence. This includes learning, problem-solving, perception, understanding language
```

##### [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
AI是什么？
-------------------- Output --------------------
AI是“人工智能”（Artificial Intelligence）的缩写。它是一门研究如何创建智能机器的学科，这些机器能够执行通常需要人类
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
What is AI?
-------------------- Output --------------------
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think, learn, and perform tasks that typically require human intelligence.
```

##### [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
AI是什么？
-------------------- Output --------------------
AI是“人工智能”的简称，是指由人结合科学原理设计，并通过工程实践创造的能够完成特定任务的软件或硬件系统。这些系统
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
What is AI?
-------------------- Output --------------------
Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that would typically require human intelligence. These tasks can include things like visual perception
```