# Run FlashMoE Portable Zip on Intel GPU

<p>
  <b>< English</b> | <a href='./flashmoe_portable_zip_gpu_quickstart.zh-CN.md'>中文</a> >
</p>

This guide demonstrates how to use [FlashMoE portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run DeepSeek V3/R1 on Intel GPU with `ipex-llm` (without the need of manual installations).

> [!NOTE]
> FlashMoE portable zip has been verified on:
> - Intel Core Ultra processors
> - Intel Core 11th - 14th gen processors
> - Intel Arc A-Series GPU
> - Intel Arc B-Series GPU
> Currently, IPEX-LLM only provides FlashMoE portable zip on Windows. 

## Table of Contents
- [Windows Quickstart](#windows-quickstart)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download and Unzip](#step-1-download-and-unzip)
  - [Step 3: Runtime Configuration](#step-2-runtime-configuration)
  - [Step 3: Run GGUF models](#step-3-run-gguf-models)
- [Tips & Troubleshooting](#tips--troubleshooting)
  - [Error: Detected different sycl devices](#error-detected-different-sycl-devices)
  - [Multi-GPUs usage](#multi-gpus-usage)
  - [Performance Environment](#performance-environment)

## Windows Quickstart

### Prerequisites

Check your GPU driver version, and update it if needed:

- For Intel Core Ultra processors (Series 2) or Intel Arc B-Series GPU, we recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- For other Intel iGPU/dGPU, we recommend using GPU driver version [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

### Step 1: Download and Unzip

Download IPEX-LLM FlashMoE portable zip for Windows users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

### Step 2: Runtime Configuration

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- To use GPU acceleration, several environment variables are required or recommended before running `FlashMoE`.
  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  ```
- For multi-GPUs user, go to [Tips](#multi-gpus-usage) for how to select specific GPU.

### Step 3: Run GGUF models

Here we provide a simple example to show how to run a community GGUF model with IPEX-LLM.  

#### Model Download
Before running, you should download or copy community GGUF model to your current directory. For instance,  `DeepSeek-R1-Q4_K_M.gguf` of [unsloth/DeepSeek-R1-Q4_K_M](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M).

#### Run GGUF model
  ```cmd
  flashmoe.exe -m D:\llm-models\gguf\DeepSeek-R1-Q4_K_M.gguf -p "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Question:The product of the ages of three teenagers is 4590. How old is the oldest? a. 18 b. 19 c. 15 d. 17 Assistant: <think>" -n 2048  -t
8 -e -ngl 99 --color -c 2500 --temp 0
  ```
Part of outputs:
```
```

## Tips & Troubleshooting

### Error: Detected different sycl devices

You will meet error log like below:
```
Found 3 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Arc A770 Graphics|  12.55|    512|    1024|   32| 16225M|     1.6.31907.700000|
| 1| [level_zero:gpu:1]|                Intel Arc A770 Graphics|  12.55|    512|    1024|   32| 16225M|     1.6.31907.700000|
| 2| [level_zero:gpu:2]|                 Intel UHD Graphics 770|   12.2|     32|     512|   32| 63218M|     1.6.31907.700000|
Error: Detected different sycl devices, the performance will limit to the slowest device. 
If you want to disable this checking and use all of them, please set environment SYCL_DEVICE_CHECK=0, and try again.
If you just want to use one of the devices, please set environment like ONEAPI_DEVICE_SELECTOR=level_zero:0 or ONEAPI_DEVICE_SELECTOR=level_zero:1 to choose your devices.
If you want to use two or more deivces, please set environment like ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"
See https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Overview/KeyFeatures/multi_gpus_selection.md for details. Exiting.
```
Because the GPUs are not the same, the jobs will be allocated according to device's memory. Upon example, the iGPU(Intel UHD Graphics 770) will get 2/3 of the computing tasks. The performance will be quit bad.  
So disable the iGPU will can get the best performance. Visit [Multi-GPUs usage](#multi-gpus-usage) for details.  
If you still want to disable this check, you can run `set SYCL_DEVICE_CHECK=0`.  

### Multi-GPUs usage

If your machine has multiple Intel GPUs, flashmoe will by default runs on all of them. If you are not clear about your hardware configuration, you can get the configuration when you run a GGUF model. Like:
```
Found 3 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Arc A770 Graphics|  12.55|    512|    1024|   32| 16225M|     1.6.31907.700000|
| 1| [level_zero:gpu:1]|                Intel Arc A770 Graphics|  12.55|    512|    1024|   32| 16225M|     1.6.31907.700000|
```

To specify which Intel GPU you would like FlashMoE to use, you could set environment variable `ONEAPI_DEVICE_SELECTOR` **before starting FlashMoE command**, as follows:  

- For **Windows** users:
  ```cmd
  set ONEAPI_DEVICE_SELECTOR=level_zero:0 (If you want to run on one GPU, FlashMoE will use the first GPU.) 
  set ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1" (If you want to run on two GPUs, FlashMoE will use the first and second GPUs.)
  ```
 
### Performance Environment
#### SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS
To enable SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS, you can run  `set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`.   
> [!NOTE]
> The environment variable SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).  
