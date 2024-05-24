# Langchain examples

The examples in this folder shows how to use [LangChain](https://www.langchain.com/) with `ipex-llm` on Intel GPU.

### 1. Install ipex-llm
Follow the instructions in [GPU Install Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) to install ipex-llm

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

### 4. Run the examples

#### 4.1. Streaming Chat

Install dependencies:

```bash
pip install langchain==0.0.184
pip install -U pandas==2.0.3
```

Then execute:

```bash
python chat.py -m MODEL_PATH -q QUESTION
```
arguments info:
- `-m MODEL_PATH`: **required**, path to the model
- `-q QUESTION`: question to ask. Default is `What is AI?`.

#### 5.2. RAG (Retrival Augmented Generation)

Install dependencies:
```bash
pip install langchain==0.0.184
pip install -U chromadb==0.3.25
pip install -U pandas==2.0.3
```

Then execute:

```bash
python rag.py -m <path_to_model> [-q QUESTION] [-i INPUT_PATH]
```
arguments info:
- `-m MODEL_PATH`: **required**, path to the model.
- `-q QUESTION`: question to ask. Default is `What is IPEX?`.
- `-i INPUT_PATH`: path to the input doc.


#### 5.2. Low Bit

The low_bit example ([low_bit.py](./low_bit.py)) showcases how to use use langchain with low_bit optimized model.
By `save_low_bit` we save the weights of low_bit model into the target folder.
> Note: `save_low_bit` only saves the weights of the model. 
> Users could copy the tokenizer model into the target folder or specify `tokenizer_id` during initialization. 

Install dependencies:
```bash
pip install langchain==0.0.184
pip install -U pandas==2.0.3
```
Then execute:

```bash
python low_bit.py -m <path_to_model> -t <path_to_target> [-q <your question>]
```
**Runtime Arguments Explained**:
- `-m MODEL_PATH`: **Required**, the path to the model
- `-t TARGET_PATH`: **Required**, the path to save the low_bit model
- `-q QUESTION`: the question

#### 5.3 vLLM

The vLLM example ([vllm.py](./vllm.py)) showcases how to use langchain with ipex-llm integrated vLLM engine.

Install dependencies:
```bash
pip install "langchain<0.2"
```

Besides, you should also install IPEX-LLM integrated vLLM according instructions listed [here](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/vLLM_quickstart.html#install-vllm)

**Runtime Arguments Explained**:
- `-m MODEL_PATH`: **Required**, the path to the model
- `-q QUESTION`: the question
- `-t MAX_TOKENS`: max tokens to generate, default 128
- `-p TENSOR_PARALLEL_SIZE`: Use multiple cards for generation
- `-l LOAD_IN_LOW_BIT`: Low bit format for quantization

To use `-p TENSOR_PARALLEL_SIZE` option, you will need to use our docker image: `intelanalytics/ipex-llm-serving-xpu:latest`. For how to use the image, try check this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/DockerGuides/vllm_docker_quickstart.html#multi-card-serving).

Also, please set the following environment variables:

```bash
export CCL_WORKER_COUNT=2
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
```