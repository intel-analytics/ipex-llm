# Run Llama 3 on Intel GPU using llama.cpp and ollama with IPEX-LLM

[Llama 3](https://llama.meta.com/llama3/) is the latest Large Language Models released by [Meta](https://llama.meta.com/) which provides state-of-the-art performance and excels at language nuances, contextual understanding, and complex tasks like translation and dialogue generation.

Now, you can easily run Llama 3 on Intel GPU using `llama.cpp` and `Ollama` with IPEX-LLM.

See the demo of running Llama-3-8B-Instruct on Intel Arc GPU using `Ollama` below.

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama-llama3-linux-arc.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-llama3-linux-arc.png"/></a></td>
  </tr>
  <tr>
    <td align="center">You could also click <a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama-llama3-linux-arc.mp4">here</a> to watch the demo video.</td>
  </tr>
</table>

## Table of Contents
- [Run Llama 3 using llama.cpp](./llama3_llamacpp_ollama_quickstart.md#1-run-llama-3-using-llamacpp)
- [Run Llama3 using Ollama](./llama3_llamacpp_ollama_quickstart.md#2-run-llama3-using-ollama)



## Quick Start
This quickstart guide walks you through how to run Llama 3 on Intel GPU using `llama.cpp` / `Ollama` with IPEX-LLM.

### 1. Run Llama 3 using llama.cpp

#### 1.1 Install IPEX-LLM for llama.cpp and Initialize

Visit [Run llama.cpp with IPEX-LLM on Intel GPU Guide](./llama_cpp_quickstart.md), and follow the instructions in section [Prerequisites](./llama_cpp_quickstart.md#0-prerequisites) to setup and section [Install IPEX-LLM for llama.cpp](./llama_cpp_quickstart.md#1-install-ipex-llm-for-llamacpp) to install the IPEX-LLM with llama.cpp binaries, then follow the instructions in section [Initialize llama.cpp with IPEX-LLM](./llama_cpp_quickstart.md#initialize-llamacpp-with-ipex-llm) to initialize.

**After above steps, you should have created a conda environment, named `llm-cpp` for instance and have llama.cpp binaries in your current directory.**

**Now you can use these executable files by standard llama.cpp usage.**

#### 1.2 Download Llama3

There already are some GGUF models of Llama3 in community, here we take [Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) for example.

Suppose you have downloaded a [Meta-Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) model from [Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) and put it under `<model_dir>`.

#### 1.3 Run Llama3 on Intel GPU using llama.cpp

#### Runtime Configuration

To use GPU acceleration, several environment variables are required or recommended before running `llama.cpp`.

- For **Linux users**:
  
  ```bash
  source /opt/intel/oneapi/setvars.sh
  export SYCL_CACHE_PERSISTENT=1
  # [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  # [optional] if you want to run on single GPU, use below command to limit GPU may improve performance
  export ONEAPI_DEVICE_SELECTOR=level_zero:0
  ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  rem under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```

> [!TIP]
> When your machine has multi GPUs and you want to run on one of them, you need to set `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]`, here `[gpu_id]` varies based on your requirement. For more details, you can refer to [this section](../Overview/KeyFeatures/multi_gpus_selection.md#2-oneapi-device-selector).

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).

##### Run llama3

Under your current directory, exceuting below command to do inference with Llama3:

- For **Linux users**:
  
  ```bash
  ./llama-cli -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun doing something" -c 1024 -t 8 -e -ngl 33 --color --no-mmap
  ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  llama-cli -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun doing something" -c 1024 -e -ngl 33 --color --no-mmap
  ```

Under your current directory, you can also execute below command to have interactive chat with Llama3:

- For **Linux users**:
  
  ```bash
  ./llama-cli -ngl 33 --interactive-first --color -e --in-prefix '<|start_header_id|>user<|end_header_id|>\n\n' --in-suffix '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' -r '<|eot_id|>' -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -c 1024
  ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  llama-cli -ngl 33 --interactive-first --color -e --in-prefix "<|start_header_id|>user<|end_header_id|>\n\n" --in-suffix "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" -r "<|eot_id|>" -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -c 1024
  ```

Below is a sample output on Intel Arc GPU:
<img src="https://llm-assets.readthedocs.io/en/latest/_images/llama3-cpp-arc-demo.png" width=100%; />

### 2. Run Llama3 using Ollama

#### 2.1 Install IPEX-LLM for Ollama and Initialize

Visit [Run Ollama with IPEX-LLM on Intel GPU](./ollama_quickstart.md), and follow the instructions in section [Install IPEX-LLM for llama.cpp](./llama_cpp_quickstart.md#1-install-ipex-llm-for-llamacpp) to install the IPEX-LLM with Ollama binary, then follow the instructions in section [Initialize Ollama](./ollama_quickstart.md#2-initialize-ollama) to initialize.

**After above steps, you should have created a conda environment, named `llm-cpp` for instance and have ollama binary file in your current directory.**

**Now you can use this executable file by standard Ollama usage.**

#### 2.2 Run Llama3 on Intel GPU using Ollama

[ollama/ollama](https://github.com/ollama/ollama) has alreadly added [Llama3](https://ollama.com/library/llama3) into its library, so it's really easy to run Llama3 using ollama now.

##### 2.2.1 Run Ollama Serve

Launch the Ollama service:

- For **Linux users**:
  
  ```bash
  export no_proxy=localhost,127.0.0.1
  export ZES_ENABLE_SYSMAN=1
  export OLLAMA_NUM_GPU=999
  source /opt/intel/oneapi/setvars.sh
  export SYCL_CACHE_PERSISTENT=1
  # [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  # [optional] if you want to run on single GPU, use below command to limit GPU may improve performance
  export ONEAPI_DEVICE_SELECTOR=level_zero:0

  ./ollama serve
  ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  set no_proxy=localhost,127.0.0.1
  set ZES_ENABLE_SYSMAN=1
  set OLLAMA_NUM_GPU=999
  set SYCL_CACHE_PERSISTENT=1
  rem under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

  ollama serve
  ```


> [!NOTE]
>
> To allow the service to accept connections from all IP addresses, use `OLLAMA_HOST=0.0.0.0 ./ollama serve` instead of just `./ollama serve`.

> [!TIP]
> When your machine has multi GPUs and you want to run on one of them, you need to set `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]`, here `[gpu_id]` varies based on your requirement. For more details, you can refer to [this section](../Overview/KeyFeatures/multi_gpus_selection.md#2-oneapi-device-selector).

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).

##### 2.2.2 Using Ollama Run Llama3

Keep the Ollama service on and open another terminal and run llama3 with `ollama run`:

- For **Linux users**:
  
  ```bash
  export no_proxy=localhost,127.0.0.1
  ./ollama run llama3:8b-instruct-q4_K_M
  ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  set no_proxy=localhost,127.0.0.1
  ollama run llama3:8b-instruct-q4_K_M
  ```

> [!NOTE]
>
> Here we just take `llama3:8b-instruct-q4_K_M` for example, you can replace it with any other Llama3 model you want.

Below is a sample output on Intel Arc GPU :
<img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-llama3-arc-demo.png" width=100%; />
