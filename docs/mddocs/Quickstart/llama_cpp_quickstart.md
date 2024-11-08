# Run llama.cpp with IPEX-LLM on Intel GPU 
<p>
  <b>< English</b> | <a href='./llama_cpp_quickstart.zh-CN.md'>中文</a> >
</p>

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) prvoides fast LLM inference in pure C++ across a variety of hardware; you can now use the C++ interface of [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `llama.cpp` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of running LLaMA2-7B on Intel Arc GPU below.

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/llama-cpp-arc.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-cpp-arc.png"/></a></td>
  </tr>
  <tr>
    <td align="center">You could also click <a href="https://llm-assets.readthedocs.io/en/latest/_images/llama-cpp-arc.mp4">here</a> to watch the demo video.</td>
  </tr>
</table>

> [!NOTE]
> `ipex-llm[cpp]==2.2.0b20240826` is consistent with [62bfef5](https://github.com/ggerganov/llama.cpp/commit/62bfef5194d5582486d62da3db59bf44981b7912) of llama.cpp.
>
> Our latest version is consistent with [a1631e5](https://github.com/ggerganov/llama.cpp/commit/a1631e53f6763e17da522ba219b030d8932900bd) of llama.cpp.

> [!NOTE]
> Starting from `ipex-llm[cpp]==2.2.0b20240912`, oneAPI dependency of `ipex-llm[cpp]` on Windows will switch from `2024.0.0` to `2024.2.1` .
> 
> For this update, it's necessary to create a new conda environment to install the latest version on Windows. If you directly upgrade to `ipex-llm[cpp]>=2.2.0b20240912` in the previous cpp conda environment, you may encounter the error `Can't find sycl7.dll`.

## Table of Contents
- [Prerequisites](./llama_cpp_quickstart.md#0-prerequisites)
- [Install IPEX-LLM for llama.cpp](./llama_cpp_quickstart.md#1-install-ipex-llm-for-llamacpp)
- [Setup for running llama.cpp](./llama_cpp_quickstart.md#2-setup-for-running-llamacpp)
- [Example: Running community GGUF models with IPEX-LLM](./llama_cpp_quickstart.md#3-example-running-community-gguf-models-with-ipex-llm)
- [Troubleshooting](./llama_cpp_quickstart.md#troubleshooting)

## Quick Start
This quickstart guide walks you through installing and running `llama.cpp` with `ipex-llm`.

### 0 Prerequisites
IPEX-LLM's support for `llama.cpp` now is available for Linux system and Windows system.

#### Linux
For Linux system, we recommend Ubuntu 20.04 or later (Ubuntu 22.04 is preferred).

Visit the [Install IPEX-LLM on Linux with Intel GPU](./install_linux_gpu.md), follow [Install Intel GPU Driver](./install_linux_gpu.md#install-gpu-driver) and [Install oneAPI](./install_linux_gpu.md#install-oneapi) to install GPU driver and Intel® oneAPI Base Toolkit 2024.0.

#### Windows (Optional)

Please make sure your GPU driver version is equal or newer than `31.0.101.5522`. If it is not, follow the instructions in [this section](./install_windows_gpu.md#optional-update-gpu-driver) to update your GPU driver; otherwise, you might encounter gibberish output. 

### 1. Install IPEX-LLM for llama.cpp

To use `llama.cpp` with IPEX-LLM, first ensure that `ipex-llm[cpp]` is installed.

- For **Linux users**:
  
  ```bash
  conda create -n llm-cpp python=3.11
  conda activate llm-cpp
  pip install --pre --upgrade ipex-llm[cpp]
  ```

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  conda create -n llm-cpp python=3.11
  conda activate llm-cpp
  pip install --pre --upgrade ipex-llm[cpp]
  ```

**After the installation, you should have created a conda environment, named `llm-cpp` for instance, for running `llama.cpp` commands with IPEX-LLM.**

### 2. Setup for running llama.cpp

First you should create a directory to use `llama.cpp`, for instance, use following command to create a `llama-cpp` directory and enter it.
```cmd
mkdir llama-cpp
cd llama-cpp
```

#### Initialize llama.cpp with IPEX-LLM

Then you can use following command to initialize `llama.cpp` with IPEX-LLM:

- For **Linux users**:
  
  ```bash
  init-llama-cpp
  ```

  After `init-llama-cpp`, you should see many soft links of `llama.cpp`'s executable files and a `convert.py` in current directory.

  ![init_llama_cpp_demo_image](https://llm-assets.readthedocs.io/en/latest/_images/init_llama_cpp_demo_image.png)

- For **Windows users**:

  Please run the following command with **administrator privilege in Miniforge Prompt**.

  ```cmd
  init-llama-cpp.bat
  ```

  After `init-llama-cpp.bat`, you should see many soft links of `llama.cpp`'s executable files and a `convert.py` in current directory.

  ![init_llama_cpp_demo_image_windows](https://llm-assets.readthedocs.io/en/latest/_images/init_llama_cpp_demo_image_windows.png)

> [!TIP]
> `init-llama-cpp` will create soft links of llama.cpp's executable files to current directory, if you want to use these executable files in other places, don't forget to run above commands again.

> [!NOTE]
> If you have installed higher version `ipex-llm[cpp]` and want to upgrade your binary file, don't forget to remove old binary files first and initialize again with `init-llama-cpp` or `init-llama-cpp.bat`.

**Now you can use these executable files by standard llama.cpp's usage.**

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

### 3. Example: Running community GGUF models with IPEX-LLM

Here we provide a simple example to show how to run a community GGUF model with IPEX-LLM.

#### Model Download
Before running, you should download or copy community GGUF model to your current directory. For instance,  `mistral-7b-instruct-v0.1.Q4_K_M.gguf` of [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main).

#### Run the quantized model

- For **Linux users**:
  
  ```bash
  ./llama-cli -m mistral-7b-instruct-v0.1.Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -c 1024 -t 8 -e -ngl 99 --color
  ```

  > **Note**:
  >
  > For more details about meaning of each parameter, you can use `./llama-cli -h`.

- For **Windows users**:

  Please run the following command in Miniforge Prompt.

  ```cmd
  llama-cli -m mistral-7b-instruct-v0.1.Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -c 1024 -t 8 -e -ngl 99 --color
  ```

  > **Note**:
  >
  > For more details about meaning of each parameter, you can use `./llama-cli -h`.

#### Sample Output
```
Log start
main: build = 1 (6f4ec98)
main: built with MSVC 19.39.33519.0 for
main: seed  = 1724921424
llama_model_loader: loaded meta data with 20 key-value pairs and 291 tensors from D:\gguf-models\mistral-7b-instruct-v0.1.Q4_K_M.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = mistralai_mistral-7b-instruct-v0.1
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                       llama.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  11:                          general.file_type u32              = 15
llama_model_loader: - kv  12:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  19:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens cache size = 3
llm_load_vocab: token to piece cache size = 0.1637 MB
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 7.24 B
llm_load_print_meta: model size       = 4.07 GiB (4.83 BPW)
llm_load_print_meta: general.name     = mistralai_mistral-7b-instruct-v0.1
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_print_meta: max token length = 48
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:      SYCL0 buffer size =  4095.05 MiB
llm_load_tensors:        CPU buffer size =    70.31 MiB
..............................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |                     |
|  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                     Intel Arc Graphics|    1.3|    112|    1024|   32| 13578M|            1.3.27504|
llama_kv_cache_init:      SYCL0 KV buffer size =    64.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =     0.12 MiB
llama_new_context_with_model:      SYCL0 compute buffer size =    81.00 MiB
llama_new_context_with_model:  SYCL_Host compute buffer size =     9.01 MiB
llama_new_context_with_model: graph nodes  = 902
llama_new_context_with_model: graph splits = 2

system_info: n_threads = 8 / 18 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 512, n_batch = 2048, n_predict = 32, n_keep = 1


 Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun exploring the world. She lived in a small village where there weren't many opportunities for adventures, but that didn't stop her. She would often read
llama_print_timings:        load time =    xxxx ms
llama_print_timings:      sample time =     x.xx ms /    32 runs   (   xx.xx ms per token,  xx.xx tokens per second)
llama_print_timings: prompt eval time =    xx.xx ms /    31 tokens (   xx.xx ms per token,  xx.xx tokens per second)
llama_print_timings:        eval time =    xx.xx ms /    31 runs   (   xx.xx ms per token,  xx.xx tokens per second)
llama_print_timings:       total time =    xx.xx ms /    62 tokens
Log end

```

### Troubleshooting

#### 1. Unable to run the initialization script
If you are unable to run `init-llama-cpp.bat`, please make sure you have installed `ipex-llm[cpp]` in your conda environment. If you have installed it, please check if you have activated the correct conda environment. Also, if you are using Windows, please make sure you have run the script with administrator privilege in prompt terminal.

#### 2. `DeviceList is empty. -30 (PI_ERROR_INVALID_VALUE)` error
On Linux, this error happens when devices starting with `[ext_oneapi_level_zero]` are not found. Please make sure you have installed level-zero, and have sourced `/opt/intel/oneapi/setvars.sh` before running the command.

#### 3. `Prompt is too long` error
If you encounter `main: prompt is too long (xxx tokens, max xxx)`, please increase the `-c` parameter to set a larger size of context.

#### 4. `gemm: cannot allocate memory on host` error / `could not create an engine` error
If you meet `oneapi::mkl::oneapi::mkl::blas::gemm: cannot allocate memory on host` error, or `could not create an engine` on Linux, this is probably caused by pip installed OneAPI dependencies. You should prevent installing like `pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0`, and instead use `apt` to install on Linux. Please refer to [this guide](./install_linux_gpu.md) for more details.

#### 5. Fail to quantize model
If you encounter `main: failed to quantize model from xxx`, please make sure you have created related output directory.

#### 6. Program hang during model loading
If your program hang after `llm_load_tensors:  SYCL_Host buffer size =    xx.xx MiB`, you can add `--no-mmap` in your command.

#### 7. How to set `-ngl` parameter
`-ngl` means the number of layers to store in VRAM. If your VRAM is enough, we recommend putting all layers on GPU, you can just set `-ngl` to a large number like 999 to achieve this goal.

If `-ngl` is set to 0, it means that the entire model will run on CPU. If `-ngl` is set to greater than 0 and less than model layers, then it's mixed GPU + CPU scenario.

#### 8. How to specificy GPU
If your machine has multi GPUs, `llama.cpp` will default use all GPUs which may slow down your inference for model which can run on single GPU. You can add `-sm none` in your command to use one GPU only.

Also, you can use `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]` to select device before excuting your command, more details can refer to [here](../Overview/KeyFeatures/multi_gpus_selection.md#2-oneapi-device-selector).

#### 9. Program crash with Chinese prompt
If you run the llama.cpp program on Windows and find that your program crashes or outputs abnormally when accepting Chinese prompts, you can open `Region->Administrative->Change System locale..`, check `Beta: Use Unicode UTF-8 for worldwide language support` option and then restart your computer.

For detailed instructions on how to do this, see [this issue](https://github.com/intel-analytics/ipex-llm/issues/10989#issuecomment-2105600469).

#### 10. sycl7.dll not found error
If you meet `System Error: sycl7.dll not found` on Windows or you meet similar error on Linux, please check:

1. if you have installed conda and if you are in the right conda environment which has pip installed oneapi dependencies on Windows
2. if you have executed `source /opt/intel/oneapi/setvars.sh` on Linux

#### 11. Check driver first when you meet garbage output
If you meet garbage output, please check if your GPU driver version is >= [31.0.101.5522](https://www.intel.cn/content/www/cn/zh/download/785597/823163/intel-arc-iris-xe-graphics-windows.html). If not, please follow the instructions in [this section](./install_linux_gpu.md#install-gpu-driver) to update your GPU driver.

#### 12. Why my program can't find sycl device
If you meet `GGML_ASSERT: C:/Users/Administrator/actions-runner/cpp-release/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml-sycl.cpp:18283: main_gpu_id<g_all_sycl_device_count` error or similar error, and you find nothing is output when using `ls-sycl-device`, this is because llama.cpp cannot find the sycl device. On some laptops, the installation of the ARC driver may lead to a forced installation of `OpenCL, OpenGL, and Vulkan Compatibility Pack` by Microsoft, which inadvertently blocks the system from locating sycl devices. This issue can be resolved by manually uninstalling it in Microsoft store.

#### 13. Core dump when having both integrated and dedicated graphics
If you have both integrated and dedicated graphics displayed in your llama.cpp's device log and don't specify which device to use, it will cause a core dump. In such case, you may need to specify `export ONEAPI_DEVICE_SELECTOR=level_zero:0` before running `llama-cli`.

#### 14. `Native API failed` error
On latest version of `ipex-llm`, you might come across `native API failed` error with certain models without the `-c` parameter. Simply adding `-c xx` would resolve this problem.

#### 15. `signal: bus error (core dumped)` error
If you meet this error, please check your Linux kernel version first. You may encounter this issue on higher kernel versions (like kernel 6.15). You can also refer to [this issue](https://github.com/intel-analytics/ipex-llm/issues/10955) to see if it helps.

#### 16. `backend buffer base cannot be NULL` error
If you meet `ggml-backend.c:96: GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL") failed`, simply adding `-c xx` parameter during inference, for example `-c 1024` would resolve this problem.