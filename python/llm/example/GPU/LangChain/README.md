# Langchain Example

The examples in this folder shows how to use [LangChain](https://www.langchain.com/) with `ipex-llm` on Intel GPU.

> [!TIP]
> For more information, please refer to the upstream LangChain LLM documentation with IPEX-LLM [here](https://python.langchain.com/docs/integrations/llms/ipex_llm), and upstream LangChain embedding model documentation with IPEX-LLM [here](https://python.langchain.com/docs/integrations/text_embedding/ipex_llm_gpu/).

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#requirements) for more information.

## 1. Install

### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## 2. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

## 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
### 3.1 Configurations for Linux
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

### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU and Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>


> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.

## 4. Run examples with LangChain

### 4.1. Example: Streaming Chat

Install LangChain dependencies:

```bash
pip install -U langchain langchain-community
```

In the current directory, run the example with command:

```bash
python chat.py -m MODEL_PATH -q QUESTION
```
**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **required**, path to the model
- `-q QUESTION`: question to ask. Default is `What is AI?`.

### 4.2. Example: Retrival Augmented Generation (RAG)

The RAG example ([rag.py](./rag.py)) shows how to load the input text into vector database, and then use LangChain to build a retrival pipeline.

Install LangChain dependencies:

```bash
pip install -U langchain langchain-community langchain-chroma sentence-transformers==3.0.1
```

In the current directory, run the example with command:

```bash
python rag.py -m <path_to_llm_model> -e <path_to_embedding_model> [-q QUESTION] [-i INPUT_PATH]
```
**Additional Parameters for Configuration:**
- `-m LLM_MODEL_PATH`: **required**, path to the model.
- `-e EMBEDDING_MODEL_PATH`: **required**, path to the embedding model.
- `-q QUESTION`: question to ask. Default is `What is IPEX-LLM?`.
- `-i INPUT_PATH`: path to the input doc.


### 4.3. Example: Low Bit

The low_bit example ([low_bit.py](./low_bit.py)) showcases how to use use LangChain with low_bit optimized model.LangChain
By `save_low_bit` we save the weights of low_bit model into the target folder.
> [!NOTE]
> `save_low_bit` only saves the weights of the model. 
> Users could copy the tokenizer model into the target folder or specify `tokenizer_id` during initialization. 

Install LangChain dependencies:

```bash
pip install -U langchain langchain-community
```

In the current directory, run the example with command:

```bash
python low_bit.py -m <path_to_model> -t <path_to_target> [-q <your question>]
```
**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **Required**, the path to the model
- `-t TARGET_PATH`: **Required**, the path to save the low_bit model
- `-q QUESTION`: question to ask. Default is `What is AI?`.
