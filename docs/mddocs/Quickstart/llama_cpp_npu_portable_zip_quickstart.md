# Run llama.cpp Portable Zip on Intel NPU with IPEX-LLM
<p>
  <b>< English</b> | <a href='./llama_cpp_npu_portable_zip_quickstart.zh-CN.md'>中文</a> >
</p>

IPEX-LLM provides llama.cpp support for running GGUF models on Intel NPU. This guide demonstrates how to use [llama.cpp NPU portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run on Intel NPU (without the need of manual installations).

> [!IMPORTANT]
> 
> - IPEX-LLM currently only supports Windows on Intel NPU.
> - Only `meta-llama/Llama-3.2-3B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` are supported.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Download and Unzip](#step-1-download-and-unzip)
- [Step 2: Setup](#step-2-setup)
- [Step 3: Run GGUF Model](#step-3-run-gguf-model)
- [More details](npu_quickstart.md)


## Prerequisites

Check your NPU driver version, and update it if needed:

- Please use NPU driver version [32.0.100.3104](https://www.intel.com/content/www/us/en/download/794734/838895/intel-npu-driver-windows.html).
- And you could refer to [here](npu_quickstart.md#update-npu-driver) for details about NPU driver update.

## Step 1: Download and Unzip

Download IPEX-LLM llama.cpp NPU portable zip for Windows users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

## Step 2: Setup

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- Runtime configuration based on your device:
  - For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxV (code name Lunar Lake)**:

    - For Intel Core™ Ultra 7 Processor 258V:
        No runtime configuration required.

    - For Intel Core™ Ultra 5 Processor 228V & 226V:
        ```cmd
        set IPEX_LLM_NPU_DISABLE_COMPILE_OPT=1
        ```

  - For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxK or 2xxH (code name Arrow Lake)**:
    ```cmd
    set IPEX_LLM_NPU_ARL=1
    ```

  - For **Intel Core™ Ultra Processors (Series 1) with processor number 1xxH (code name Meteor Lake)**:
    ```cmd
    set IPEX_LLM_NPU_MTL=1
    ```

## Step 3: Run GGUF Model

You could then use cli tool to run GGUF models on Intel NPU through running `llama-cli-npu.exe` in the "Command Prompt" as following:

```cmd
llama-cli-npu.exe -m DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf -n 32 --prompt "What is AI?"
```
