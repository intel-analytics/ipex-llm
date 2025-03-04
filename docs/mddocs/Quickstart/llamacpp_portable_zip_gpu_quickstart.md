# Run llama.cpp Portable Zip on Intel GPU with IPEX-LLM
<p>
   < <b>English</b> | <a href='./llamacpp_portable_zip_gpu_quickstart.zh-CN.md'>中文</a> >
</p>

This guide demonstrates how to use [llama.cpp portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run llama.cpp on Intel GPU with `ipex-llm` (without the need of manual installations).

> [!NOTE]
> llama.cpp portable zip has been verified on:
> - Intel Core Ultra processors
> - Intel Core 11th - 14th gen processors
> - Intel Arc A-Series GPU
> - Intel Arc B-Series GPU

## Table of Contents
- [Windows Quickstart](#windows-quickstart)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download and Unzip](#step-1-download-and-unzip)
  - [Step 3: Runtime Configuration](#step-2-runtime-configuration)
  - [Step 3: Run GGUF models](#step-3-run-gguf-models)
- [Linux Quickstart](#linux-quickstart)
  - [Prerequisites](#prerequisites-1)
  - [Step 1: Download and Extract](#step-1-download-and-extract)
  - [Step 2: Runtime Configuration](#step-2-runtime-configuration-1)
  - [Step 3: Run GGUF models](#step-3-run-gguf-models-1)
  - [(New) FlashMoE for Moe Models (e.g., DeeSeek V3/R1) using llama.cpp](#flashmoe-for-deeseek-v3r1)
- [Tips & Troubleshooting](#tips--troubleshooting)
  - [Error: Detected different sycl devices](#error-detected-different-sycl-devices)
  - [Multi-GPUs usage](#multi-gpus-usage)
  - [Performance Environment](#performance-environment)
- [More Details](llama_cpp_quickstart.md)

## Windows Quickstart

### Prerequisites

Check your GPU driver version, and update it if needed:

- For Intel Core Ultra processors (Series 2) or Intel Arc B-Series GPU, we recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- For other Intel iGPU/dGPU, we recommend using GPU driver version [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

### Step 1: Download and Unzip

Download IPEX-LLM llama.cpp portable zip for Windows users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

### Step 2: Runtime Configuration

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- To use GPU acceleration, several environment variables are required or recommended before running `llama.cpp`.
  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  ```
- For multi-GPUs user, go to [Tips](#multi-gpus-usage) for how to select specific GPU.

### Step 3: Run GGUF models

Here we provide a simple example to show how to run a community GGUF model with IPEX-LLM.  

#### Model Download
Before running, you should download or copy community GGUF model to your current directory. For instance,  `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` of [bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf).

#### Run GGUF model

```cmd
llama-cli.exe -m D:\llm-models\gguf\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf -p "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Question:The product of the ages of three teenagers is 4590. How old is the oldest? a. 18 b. 19 c. 15 d. 17 Assistant: <think>" -n 2048  -t 8 -e -ngl 99 --color -c 2500 --temp 0
```

Part of outputs:

```
Found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                     Intel Arc Graphics|  12.71|    128|    1024|   32| 13578M|            1.3.27504|
llama_kv_cache_init:      SYCL0 KV buffer size =   138.25 MiB
llama_new_context_with_model: KV self size  =  138.25 MiB, K (f16):   69.12 MiB, V (f16):   69.12 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      SYCL0 compute buffer size =  1501.00 MiB
llama_new_context_with_model:  SYCL_Host compute buffer size =    58.97 MiB
llama_new_context_with_model: graph nodes  = 874
llama_new_context_with_model: graph splits = 2
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 8

system_info: n_threads = 8 (n_threads_batch = 8) / 22 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

sampler seed: 341519086
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = -1
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, temp = 0.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist

generate: n_ctx = 2528, n_batch = 4096, n_predict = 2048, n_keep = 1

<think>
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
</think>

<answer>XXXX</answer> [end of text]


llama_perf_sampler_print:    sampling time =     xxx.xx ms /  1386 runs   (    x.xx ms per token, xxxxx.xx tokens per second)
llama_perf_context_print:        load time =   xxxxx.xx ms
llama_perf_context_print: prompt eval time =     xxx.xx ms /   129 tokens (    x.xx ms per token,   xxx.xx tokens per second)
llama_perf_context_print:        eval time =   xxxxx.xx ms /  1256 runs   (   xx.xx ms per token,    xx.xx tokens per second)
llama_perf_context_print:       total time =   xxxxx.xx ms /  1385 tokens
```

## Linux Quickstart

### Prerequisites

Check your GPU driver version, and update it if needed; we recommend following [Intel client GPU driver installation guide](https://dgpu-docs.intel.com/driver/client/overview.html) to install your GPU driver.

### Step 1: Download and Extract

Download IPEX-LLM llama.cpp portable tgz for Linux users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the tgz file to a folder.

### Step 2: Runtime Configuration

- Open a "Terminal", and enter the extracted folder through `cd /PATH/TO/EXTRACTED/FOLDER`
- To use GPU acceleration, several environment variables are required or recommended before running `llama.cpp`.
  ```bash
  export SYCL_CACHE_PERSISTENT=1
  ```
- For multi-GPUs user, go to [Tips](#multi-gpus-usage) for how to select specific GPU.

### Step 3: Run GGUF models

Here we provide a simple example to show how to run a community GGUF model with IPEX-LLM.  

#### Model Download
Before running, you should download or copy community GGUF model to your current directory. For instance,  `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` of [bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf).

#### Run GGUF model

```bash
llama-cli -m D:\llm-models\gguf\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf -p "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Question:The product of the ages of three teenagers is 4590. How old is the oldest? a. 18 b. 19 c. 15 d. 17 Assistant: <think>" -n 2048  -t 8 -e -ngl 99 --color -c 2500 --temp 0
```

Part of outputs:

```bash
Found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                     Intel Arc Graphics|  12.71|    128|    1024|   32| 13578M|            1.3.27504|
llama_kv_cache_init:      SYCL0 KV buffer size =   138.25 MiB
llama_new_context_with_model: KV self size  =  138.25 MiB, K (f16):   69.12 MiB, V (f16):   69.12 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      SYCL0 compute buffer size =  1501.00 MiB
llama_new_context_with_model:  SYCL_Host compute buffer size =    58.97 MiB
llama_new_context_with_model: graph nodes  = 874
llama_new_context_with_model: graph splits = 2
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 8

system_info: n_threads = 8 (n_threads_batch = 8) / 22 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

sampler seed: 341519086
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = -1
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, temp = 0.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist

generate: n_ctx = 2528, n_batch = 4096, n_predict = 2048, n_keep = 1

<think>
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
</think>

<answer>XXXX</answer> [end of text]
```

### FlashMoE for DeeSeek V3/R1

FlashMoE is a command-line tool built on llama.cpp, optimized for mixture-of-experts (MoE) models such as DeepSeek V3/R1. Now, it's available for Linux platforms.

Tested MoE GGUF Models (other MoE GGUF models are also supported):
- [DeepSeek-V3-Q4_K_M](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M)
- [DeepSeek-V3-Q6_K](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q6_K)
- [DeepSeek-R1-Q4_K_M.gguf](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M)
- [DeepSeek-R1-Q6_K](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q6_K)

#### Run DeepSeek V3/R1 with FlashMoE

Requirements: 
- 380GB CPU Memory
- 1-8 ARC A770
- 500GB Disk

Note: 
- Larger models and other precisions may require more resources.
- For 1 ARC A770 platform, please reduce context length (e.g., 1024) to avoid OOM. Add this option `-c 1024` at the end of below command.

Before running, you should download or copy community GGUF model to your current directory. For instance,  `DeepSeek-R1-Q4_K_M.gguf` of [DeepSeek-R1-Q4_K_M.gguf](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M).

Run `DeepSeek-R1-Q4_K_M.gguf`

```bash
flash-moe -m DeepSeek-R1-Q4_K_M-00001-of-00009.gguf --prompt "What's AI?"
```

Part of outputs

```bash
llama_kv_cache_init:      SYCL0 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL1 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL2 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL3 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL4 KV buffer size =  1120.00 MiB
llama_kv_cache_init:      SYCL5 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL6 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL7 KV buffer size =   960.00 MiB
llama_new_context_with_model: KV self size  = 9760.00 MiB, K (i8): 5856.00 MiB, V (i8): 3904.00 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      SYCL0 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL1 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL2 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL3 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL4 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL5 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL6 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL7 compute buffer size =  3264.00 MiB
llama_new_context_with_model:  SYCL_Host compute buffer size =  1332.05 MiB
llama_new_context_with_model: graph nodes  = 5184 (with bs=4096), 4720 (with bs=1)
llama_new_context_with_model: graph splits = 125
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 48

system_info: n_threads = 48 (n_threads_batch = 48) / 192 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

sampler seed: 2052631435
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = -1
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist
generate: n_ctx = 4096, n_batch = 4096, n_predict = -1, n_keep = 1

<think>
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
</think>

<answer>XXXX</answer> [end of text]
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

If your machine has multiple Intel GPUs, llama.cpp will by default runs on all of them. If you are not clear about your hardware configuration, you can get the configuration when you run a GGUF model. Like:
```
Found 3 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Arc A770 Graphics|  12.55|    512|    1024|   32| 16225M|     1.6.31907.700000|
| 1| [level_zero:gpu:1]|                Intel Arc A770 Graphics|  12.55|    512|    1024|   32| 16225M|     1.6.31907.700000|
```

To specify which Intel GPU you would like llama.cpp to use, you could set environment variable `ONEAPI_DEVICE_SELECTOR` **before starting llama.cpp command**, as follows:  

- For **Windows** users:
  ```cmd
  set ONEAPI_DEVICE_SELECTOR=level_zero:0 (If you want to run on one GPU, llama.cpp will use the first GPU.) 
  set ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1" (If you want to run on two GPUs, llama.cpp will use the first and second GPUs.)
  ```
 
### Performance Environment
#### SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS
To enable SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS, you can run  `set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`.   
> [!NOTE]
> The environment variable SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).  
