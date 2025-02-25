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

## Table of Contents
- [Windows Quickstart](#windows-quickstart)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download and Unzip](#step-1-download-and-unzip)
  - [Step 3: Runtime Configuration](#step-2-start-ollama-serve)
  - [Step 3: Run GGUF models](#step-3-run-ollama)
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
- Run `start-ollama.bat` in the "Command Prompt. A window will then pop up as shown below:

### Step 3: Run llama.cpp

You could then use llama.cpp to run LLMs on Intel GPUs through running `ollama run deepseek-r1:7b` in the same "Command Prompt" (not the pop-up window). You may use any other model.

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

