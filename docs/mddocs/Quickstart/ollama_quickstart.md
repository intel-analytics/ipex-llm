# Run Ollama with IPEX-LLM on Intel GPU
<p>
  <b>< English</b> | <a href='./ollama_quickstart.zh-CN.md'>中文</a> >
</p>

[ollama/ollama](https://github.com/ollama/ollama) is popular framework designed to build and run language models on a local machine; you can now use the C++ interface of [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `ollama` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of running LLaMA2-7B on Intel Arc GPU below.

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.png"/></a></td>
  </tr>
  <tr>
    <td align="center">You could also click <a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.mp4">here</a> to watch the demo video.</td>
  </tr>
</table>

> [!NOTE]
> `ipex-llm[cpp]==2.2.0b20240826` is consistent with [v0.1.39](https://github.com/ollama/ollama/releases/tag/v0.1.39) of ollama.
>
> Our current version is consistent with [v0.3.6](https://github.com/ollama/ollama/releases/tag/v0.3.6) of ollama.

> [!NOTE]
> Starting from `ipex-llm[cpp]==2.2.0b20240912`, oneAPI dependency of `ipex-llm[cpp]` on Windows will switch from `2024.0.0` to `2024.2.1` .
> 
> For this update, it's necessary to create a new conda environment to install the latest version on Windows. If you directly upgrade to `ipex-llm[cpp]>=2.2.0b20240912` in the previous cpp conda environment, you may encounter the error `Can't find sycl7.dll`.

## Table of Contents
- [Install IPEX-LLM for Ollama](./ollama_quickstart.md#1-install-ipex-llm-for-ollama)
- [Initialize Ollama](./ollama_quickstart.md#2-initialize-ollama)
- [Run Ollama Serve](./ollama_quickstart.md#3-run-ollama-serve)
- [Pull Model](./ollama_quickstart.md#4-pull-model)
- [Using Ollama](./ollama_quickstart.md#5-using-ollama)

## Quickstart

### 1. Install IPEX-LLM for Ollama

IPEX-LLM's support for `ollama` now is available for Linux system and Windows system.

Visit [Run llama.cpp with IPEX-LLM on Intel GPU Guide](./llama_cpp_quickstart.md), and follow the instructions in section [Prerequisites](./llama_cpp_quickstart.md#0-prerequisites) to setup and section [Install IPEX-LLM cpp](./llama_cpp_quickstart.md#1-install-ipex-llm-for-llamacpp) to install the IPEX-LLM with Ollama binaries.

**After the installation, you should have created a conda environment, named `llm-cpp` for instance, for running `ollama` commands with IPEX-LLM.**

### 2. Initialize Ollama

Activate the `llm-cpp` conda environment and initialize Ollama by executing the commands below. A symbolic link to `ollama` will appear in your current directory.

- For **Linux users**:
  
  ```bash
  conda activate llm-cpp
  init-ollama
  ```

- For **Windows users**:

  Please run the following command with **administrator privilege in Miniforge Prompt**.

  ```cmd
  conda activate llm-cpp
  init-ollama.bat
  ```

> [!NOTE]
> If you have installed higher version `ipex-llm[cpp]` and want to upgrade your ollama binary file, don't forget to remove old binary files first and initialize again with `init-ollama` or `init-ollama.bat`.

**Now you can use this executable file by standard ollama's usage.**

### 3. Run Ollama Serve

You may launch the Ollama service as below:

- For **Linux users**:

  ```bash
  export OLLAMA_NUM_GPU=999
  export no_proxy=localhost,127.0.0.1
  export ZES_ENABLE_SYSMAN=1
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
  set OLLAMA_NUM_GPU=999
  set no_proxy=localhost,127.0.0.1
  set ZES_ENABLE_SYSMAN=1
  set SYCL_CACHE_PERSISTENT=1
  rem under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

  ollama serve
  ```

> [!NOTE]
> Please set environment variable `OLLAMA_NUM_GPU` to `999` to make sure all layers of your model are running on Intel GPU, otherwise, some layers may run on CPU.

> [!NOTE]
> To allow the service to accept connections from all IP addresses, use `OLLAMA_HOST=0.0.0.0 ./ollama serve` instead of just `./ollama serve`.

> [!TIP]
> When your machine has multi GPUs and you want to run on one of them, you need to set `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]`, here `[gpu_id]` varies based on your requirement. For more details, you can refer to [this section](../Overview/KeyFeatures/multi_gpus_selection.md#2-oneapi-device-selector).

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).


The console will display messages similar to the following:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_serve.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_serve.png" width=100%; />
</a>


### 4. Pull Model
Keep the Ollama service on and open another terminal and run `./ollama pull <model_name>` in Linux (`ollama.exe pull <model_name>` in Windows) to automatically pull a model. e.g. `dolphin-phi:latest`:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_pull.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_pull.png" width=100%; />
</a>


### 5. Using Ollama

#### Using Curl 

Using `curl` is the easiest way to verify the API service and model. Execute the following commands in a terminal. **Replace the <model_name> with your pulled 
model**, e.g. `dolphin-phi`.

- For **Linux users**:
  
   ```bash
   curl http://localhost:11434/api/generate -d '
   { 
      "model": "<model_name>", 
      "prompt": "Why is the sky blue?", 
      "stream": false
   }'
   ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

   ```cmd
   curl http://localhost:11434/api/generate -d "
   {
      \"model\": \"<model_name>\",
      \"prompt\": \"Why is the sky blue?\",
      \"stream\": false
   }"
   ```

#### Using Ollama Run GGUF models

Ollama supports importing GGUF models in the Modelfile, for example, suppose you have downloaded a `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main), then you can create a file named `Modelfile`:

```bash
FROM ./mistral-7b-instruct-v0.1.Q4_K_M.gguf
TEMPLATE [INST] {{ .Prompt }} [/INST]
PARAMETER num_predict 64
```

Then you can create the model in Ollama by `ollama create example -f Modelfile` and use `ollama run` to run the model directly on console.

- For **Linux users**:
  
  ```bash
  export no_proxy=localhost,127.0.0.1
  ./ollama create example -f Modelfile
  ./ollama run example
  ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  set no_proxy=localhost,127.0.0.1
  ollama create example -f Modelfile
  ollama run example
  ```

An example process of interacting with model with `ollama run example` looks like the following:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" width=100%; />
</a>

### Troubleshooting
#### 1. Unable to run the initialization script
If you are unable to run `init-ollama.bat`, please make sure you have installed `ipex-llm[cpp]` in your conda environment. If you have installed it, please check if you have activated the correct conda environment. Also, if you are using Windows, please make sure you have run the script with administrator privilege in prompt terminal.

#### 2. Why model is always loaded again after several minutes
Ollama will unload model from gpu memory in every 5 minutes as default. For latest version of ollama, you could set `OLLAMA_KEEP_ALIVE=-1` to keep the model loaded in memory. Reference issue: https://github.com/intel-analytics/ipex-llm/issues/11608

#### 3. `exit status 0xc0000135` error when executing  `ollama serve`
When executing `ollama serve`, if you meet `llama runner process has terminated: exit status 0xc0000135` on Windows or you meet `ollama_llama_server: error while loading shared libraries: libmkl_core.so.2: cannot open shared object file` on Linux, this is most likely caused by the lack of sycl dependency. Please check:

1. if you have installed conda and if you are in the right conda environment which has pip installed oneapi dependencies on Windows
2. if you have executed `source /opt/intel/oneapi/setvars.sh` on Linux

#### 4. Program hang during initial model loading stage
When launching `ollama serve` for the first time on Windows, it may get stuck during the model loading phase. If you notice that the program is hanging for a long time during the first run, you can manually input a space or other characters on the server side to ensure the program is running.

#### 5. How to distinguish the community version of Ollama from the ipex-llm version of Ollama
In the server log of community version of Ollama, you may see `source=payload_common.go:139 msg="Dynamic LLM libraries [rocm_v60000 cpu_avx2 cuda_v11 cpu cpu_avx]"`.
But in the server log of ipex-llm version of Ollama, you should only see `source=payload.go:44 msg="Dynamic LLM libraries [cpu cpu_avx cpu_avx2]"`.

#### 6. Ollama hang when multiple different questions is asked or context is long
If you find ollama hang when multiple different questions is asked or context is long, and you see `update_slots : failed to free spaces in the KV cache` in the server log, this could be because that sometimes the LLM context is larger than the default `n_ctx` value, you may increase the `n_ctx` and try it again.

#### 7. `signal: bus error (core dumped)` error
If you meet this error, please check your Linux kernel version first. You may encounter this issue on higher kernel versions (like kernel 6.15). You can also refer to [this issue](https://github.com/intel-analytics/ipex-llm/issues/10955) to see if it helps.

#### 8. Out of Memory
If you have a limited GPU memory, use `set OLLAMA_NUM_PARALLEL=1` on Windows or `export OLLAMA_NUM_PARALLEL=1` on Linux before `ollama serve` to reduce GPU usage. The default `OLLAMA_NUM_PARALLEL` in ollama upstream is set to 4.
