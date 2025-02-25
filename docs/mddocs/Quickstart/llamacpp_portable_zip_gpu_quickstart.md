# Run Llama.cpp Portable Zip on Intel GPU with IPEX-LLM
<p>
  <b>< English</b> | <a href='./llamacpp_portable_zip_gpu_quickstart.zh-CN.md'>中文</a> >
</p>

This guide demonstrates how to use [llama.cpp portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run llama.cpp on Intel GPU with `ipex-llm` (without the need of manual installations).

> [!NOTE]
> Llama.cpp portable zip has been verified on:
> - Intel Core Ultra processors
> - Intel Core 11th - 14th gen processors
> - Intel Arc A-Series GPU
> - Intel Arc B-Series GPU
> Currently, IPEX-LLM only provides llama.cpp portable zip on Windows. 

## Table of Contents
- [Windows Quickstart](#windows-quickstart)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download and Unzip](#step-1-download-and-unzip)
  - [Step 3: Runtime Configuration](#step-2-runtime-configuration)
  - [Step 3: Run GGUF models](#step-3-run-llama.cpp)
- [Tips & Troubleshooting](#tips--troubleshooting)
  - [Select specific GPU to run Llama.cpp when multiple ones are available](#select-specific-gpu-to-run-ollama-when-multiple-ones-are-available)
- [More details](ollama_quickstart.md)

## Windows Quickstart

### Prerequisites

Check your GPU driver version, and update it if needed:

- For Intel Core Ultra processors (Series 2) or Intel Arc B-Series GPU, we recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- For other Intel iGPU/dGPU, we recommend using GPU driver version [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

### Step 1: Download and Unzip

Download IPEX-LLM Llama.cpp portable zip for Windows users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

### Step 2: Runtime Configuration

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- To use GPU acceleration, several environment variables are required or recommended before running `llama.cpp`.
```cmd
set SYCL_CACHE_PERSISTENT=1
rem under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
- For multi-GPUs user, go to Tips for how to select specific GPU.

### Step 3: Running community GGUF models with IPEX-LLM

Here we provide a simple example to show how to run a community GGUF model with IPEX-LLM.

- Model Download
Before running, you should download or copy community GGUF model to your current directory. For instance,  `mistral-7b-instruct-v0.1.Q4_K_M.gguf` of [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main).

#### Run the quantized model
  ```bash
  ./llama-cli -m mistral-7b-instruct-v0.1.Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -c 1024 -t 8 -e -ngl 99 --color
  ```

<div align="center">
  <img src=""  width=80%/>
</div>


## Tips & Troubleshooting

### Select specific GPU to run llama.cpp when multiple ones are available

If your machine has multiple Intel GPUs, llama.cpp will by default runs on all of them.

To specify which Intel GPU you would like llama.cpp to use, you could set environment variable `ONEAPI_DEVICE_SELECTOR` **before starting llama.cpp command**, as follows:

- Identify the id (e.g. 0, 1, etc.) for your multiple GPUs. You could find them in the logs of llama.cpp, e.g.:

  <div align="center">
    <img src=""  width=80%/>
  </div>

- For **Windows** users:

  - Open "Command Prompt", and navigate to the extracted folder by `cd /d PATH\TO\EXTRACTED\FOLDER`
  - In the "Command Prompt", set `ONEAPI_DEVICE_SELECTOR` to define the Intel GPU you want to use, e.g. `set ONEAPI_DEVICE_SELECTOR=level_zero:0`, in which `0` should be changed to your desired GPU id

