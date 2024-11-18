# Langchain examples

The examples in this folder shows how to use [LangChain](https://www.langchain.com/) with `ipex-llm` on Intel GPU.

### 1. Install ipex-llm
Follow the instructions in [GPU Install Guide](../../../../../docs/mddocs/Overview/install_gpu.md) to install ipex-llm

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

### 4. Using langchain upstream to run examples

## Streaming Chat Example

### 1. Install Prerequisites

To benefit from IPEX-LLM on Intel GPUs, there are several prerequisite steps for tools installation and environment preparation.

If you are a Windows user, visit the [Install IPEX-LLM on Windows with Intel GPU Guide](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md#install-prerequisites) to update GPU driver (optional) and install Conda.

If you are a Linux user, visit the [Install IPEX-LLM on Linux with Intel GPU](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md#install-prerequisites) to install GPU driver, Intel® oneAPI Base Toolkit 2024.0, and Conda.


### 2. Setting up Dependencies

Install langchain dependencies:

```bash
pip install -U langchain langchain-community
```

### 3. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 4.1 Configurations for Linux
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

#### 4.2 Configurations for Windows
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


### 5. Running the Streaming Chat Example

In the current directory, run the example with command:

```bash
python chat.py -m MODEL_PATH -q QUESTION
```
**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **required**, path to the model
- `-q QUESTION`: question to ask. Default is `What is AI?`.

## Retrival Augmented Generation (RAG) Example

### 1. Install Prerequisites

To benefit from IPEX-LLM on Intel GPUs, there are several prerequisite steps for tools installation and environment preparation.

If you are a Windows user, visit the [Install IPEX-LLM on Windows with Intel GPU Guide](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md#install-prerequisites) to update GPU driver (optional) and install Conda.

If you are a Linux user, visit the [Install IPEX-LLM on Linux with Intel GPU](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md#install-prerequisites) to install GPU driver, Intel® oneAPI Base Toolkit 2024.0, and Conda.


### 2. Setting up Dependencies

Install langchain dependencies:

```bash
pip install -U langchain langchain-community langchain-chroma sentence-transformers==3.0.1
```

### 3. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 4.1 Configurations for Linux
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

#### 4.2 Configurations for Windows
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

### 5. Running the Retrival Augmented Generation (RAG) Example

In the current directory, run the example with command:

```bash
python rag.py -m <path_to_llm_model> -e <path_to_embedding_model> [-q QUESTION] [-i INPUT_PATH]
```
**Additional Parameters for Configuration:**
- `-m LLM_MODEL_PATH`: **required**, path to the model.
- `-e EMBEDDING_MODEL_PATH`: **required**, path to the embedding model.
- `-q QUESTION`: question to ask. Default is `What is IPEX-LLM?`.
- `-i INPUT_PATH`: path to the input doc.


## Low Bit Example

The low_bit example ([low_bit.py](./low_bit.py)) showcases how to use use langchain with low_bit optimized model.
By `save_low_bit` we save the weights of low_bit model into the target folder.
> [!NOTE]
> `save_low_bit` only saves the weights of the model. 
> Users could copy the tokenizer model into the target folder or specify `tokenizer_id` during initialization. 


### 1. Install Prerequisites

To benefit from IPEX-LLM on Intel GPUs, there are several prerequisite steps for tools installation and environment preparation.

If you are a Windows user, visit the [Install IPEX-LLM on Windows with Intel GPU Guide](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md#install-prerequisites) to update GPU driver (optional) and install Conda.

If you are a Linux user, visit the [Install IPEX-LLM on Linux with Intel GPU](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md#install-prerequisites) to install GPU driver, Intel® oneAPI Base Toolkit 2024.0, and Conda.


### 2. Setting up Dependencies

Install langchain dependencies:

```bash
pip install -U langchain langchain-community
```

### 3. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 4.1 Configurations for Linux
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

#### 4.2 Configurations for Windows
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

### 5. Running the Low Bit Example

```bash
python low_bit.py -m <path_to_model> -t <path_to_target> [-q <your question>]
```
**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **Required**, the path to the model
- `-t TARGET_PATH`: **Required**, the path to save the low_bit model
- `-q QUESTION`: the question


## vLLM Example

The vLLM example ([vllm.py](./vllm.py)) showcases how to use langchain with ipex-llm integrated vLLM engine.

### 1. Install Prerequisites

To benefit from IPEX-LLM on Intel GPUs, there are several prerequisite steps for tools installation and environment preparation.

If you are a Windows user, visit the [Install IPEX-LLM on Windows with Intel GPU Guide](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_windows_gpu.md#install-prerequisites) to update GPU driver (optional) and install Conda.

If you are a Linux user, visit the [Install IPEX-LLM on Linux with Intel GPU](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md), and follow [Install Prerequisites](../../../../../docs/mddocs/Quickstart/install_linux_gpu.md#install-prerequisites) to install GPU driver, Intel® oneAPI Base Toolkit 2024.0, and Conda.


### 2. Setting up Dependencies

Install langchain dependencies:

```bash
pip install "langchain<0.2"
```

Besides, you should also install IPEX-LLM integrated vLLM according instructions listed [here](../../../../../docs/mddocs/Quickstart/vLLM_quickstart.md#2-install-vllm)

### 3. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 4.1 Configurations for Linux
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

#### 4.2 Configurations for Windows
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


### 5. Running the vLLM Example

**Additional Parameters for Configuration:**
- `-m MODEL_PATH`: **Required**, the path to the model
- `-q QUESTION`: the question
- `-t MAX_TOKENS`: max tokens to generate, default 128
- `-p TENSOR_PARALLEL_SIZE`: Use multiple cards for generation
- `-l LOAD_IN_LOW_BIT`: Low bit format for quantization

#### Single card

The following command shows an example on how to execute the example using one card:

```bash
python ./vllm.py -m YOUR_MODEL_PATH -q "What is AI?" -t 128 -p 1 -l sym_int4
```

#### Multi cards

To use `-p TENSOR_PARALLEL_SIZE` option, you will need to use our docker image: `intelanalytics/ipex-llm-serving-xpu:latest`. For how to use the image, try check this [guide](../../../../../docs/mddocs/DockerGuides/vllm_docker_quickstart.md#multi-card-serving).

The following command shows an example on how to execute the example using two cards:

```bash
export CCL_WORKER_COUNT=2
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
python ./vllm.py -m YOUR_MODEL_PATH -q "What is AI?" -t 128 -p 2 -l sym_int4
```