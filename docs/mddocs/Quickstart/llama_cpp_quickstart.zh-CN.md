# 使用 IPEX-LLM 在 Intel GPU 上运行 llama.cpp
<p>
  < <a href='./llama_cpp_quickstart.md'>English</a> | <b>中文</b> >
</p>

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) 是一个使用纯C++实现的、支持多种硬件平台的高效大语言模型推理库。现在，借助 [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) 的 C++ 接口作为其加速后端，你可以在 Intel **GPU**  *(如配有集成显卡，以及 Arc，Flex 和 Max 等独立显卡的本地 PC)* 上，轻松部署并运行 `llama.cpp` 。

以下是在 Intel Arc GPU 上运行 LLaMA2-7B 的 DEMO 演示。

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/llama-cpp-arc.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/llama-cpp-arc.png"/></a></td>
  </tr>
  <tr>
    <td align="center">你也可以点击<a href="https://llm-assets.readthedocs.io/en/latest/_images/llama-cpp-arc.mp4">这里</a>观看 DEMO 视频。</td>
  </tr>
</table>

> [!NOTE]
> `ipex-llm[cpp]==2.2.0b20240826` 版本与官方 llama.cpp 版本 [62bfef5](https://github.com/ggerganov/llama.cpp/commit/62bfef5194d5582486d62da3db59bf44981b7912) 一致。
>
> `ipex-llm[cpp]` 的最新版本与官方 llama.cpp 版本 [a1631e5](https://github.com/ggerganov/llama.cpp/commit/a1631e53f6763e17da522ba219b030d8932900bd) 一致。

> [!NOTE]
> 从 `ipex-llm[cpp]==2.2.0b20240912` 版本开始，Windows 上 `ipex-llm[cpp]` 依赖的 oneAPI 版本已从 `2024.0.0` 更新到 `2024.2.1`。
> 
> 如果要将 `ipex-llm[cpp]` 升级到 `2.2.0b20240912` 或更高版本，在Windows环境下，你需要新建一个干净的 conda 环境来安装新版本。如果直接在旧的 conda 环境中卸载旧版本并升级，可能会遇到 `找不到 sycl7.dll` 的错误。

## 目录
- [系统环境准备](./llama_cpp_quickstart.zh-CN.md#0-系统环境准备)
- [安装 IPEX-LLM](./llama_cpp_quickstart.zh-CN.md#1-为-llamacpp-安装-IPEX-LLM)
- [llama.cpp 运行设置](./llama_cpp_quickstart.zh-CN.md#2-llamacpp-运行设置)
- [示例: 使用 IPEX-LLM 运行社区 GGUF 模型](./llama_cpp_quickstart.zh-CN.md#3-示例-使用-ipex-llm-运行社区-GGUF-模型)
- [故障排除](./llama_cpp_quickstart.zh-CN.md#故障排除)

## 快速入门
本快速入门指南将引导你完成安装和使用 `ipex-llm` 运行 `llama.cpp`。

### 0 系统环境准备
IPEX-LLM 现在已支持在 Linux 和 Windows 系统上运行 `llama.cpp`。

#### Linux
对于 Linux 系统，我们推荐使用 Ubuntu 20.04 或更高版本 (优先推荐 Ubuntu 22.04)。

请仔细参阅网页[在配有 Intel GPU 的 Linux 系统下安装 IPEX-LLM](./install_linux_gpu.md), 首先按照 [Intel GPU 驱动程序安装](./install_linux_gpu.md#install-gpu-driver)步骤安装 Intel GPU 驱动程序，然后参考 [oneAPI 安装](./install_linux_gpu.md#install-oneapi)步骤安装 Intel® oneAPI Base Toolkit 2024.0。

#### Windows (可选)

请确保你的 GPU 驱动程序版本不低于 `31.0.101.5522`。 如果版本较低，请参考 [GPU 驱动更新指南](./install_windows_gpu.md#optional-update-gpu-driver)进行升级，否则可能会遇到输出乱码的问题。 

### 1. 为 llama.cpp 安装 IPEX-LLM

要使用 IPEX-LLM 加速的 `llama.cpp`，需要安装 `ipex-llm[cpp]`。请根据你的操作系统选择以下对应的安装步骤进行操作。

- **Linux 用户**:
  
  ```bash
  conda create -n llm-cpp python=3.11
  conda activate llm-cpp
  pip install --pre --upgrade ipex-llm[cpp]
  ```

- **Windows 用户**:

  请在 Miniforge Prompt 中运行以下命令。

  ```cmd
  conda create -n llm-cpp python=3.11
  conda activate llm-cpp
  pip install --pre --upgrade ipex-llm[cpp]
  ```

**完成上述步骤后，你应该已经创建了一个名为 `llm-cpp` 的新 conda 环境。你也可以修改上述命令来更改环境名称。该 conda 环境将用于在 Intel GPU 上使用 IPEX-LLM 运行 llama.cpp。**

### 2. llama.cpp 运行设置

首先，你需要创建一个用于存放 `llama.cpp` 的可执行文件并运行它的目录, 例如, 用如下命令创建一个名为 `llama-cpp` 目录，并进入该目录。

```cmd
mkdir llama-cpp
cd llama-cpp
```

#### 使用 IPEX-LLM 初始化 llama.cpp

然后，在当前目录下，运行下列命令进行初始化。

- **Linux 用户**:
  
  ```bash
  init-llama-cpp
  ```

  在 `intel-llama.cpp` 执行完成之后，你应该在当前目录中看到许多 `llama.cpp` 的可执行文件的软链接和一个 `convert.py` 文件。

  ![init_llama_cpp_demo_image](https://llm-assets.readthedocs.io/en/latest/_images/init_llama_cpp_demo_image.png)

- **Windows 用户**:

  请**在 Miniforge Prompt 中使用管理员权限**运行以下命令。

  ```cmd
  init-llama-cpp.bat
  ```

  在 `init-llama-cpp.bat` 执行完成之后，你应该在当前目录中看到许多 `llama.cpp` 的可执行文件的软链接和一个 `convert.py` 文件。

  ![init_llama_cpp_demo_image_windows](https://llm-assets.readthedocs.io/en/latest/_images/init_llama_cpp_demo_image_windows.png)

> [!TIP]
> `init-llama-cpp` 将会在当前目录中创建指向 llama.cpp 可执行文件的软链接，如果你想在其他地方使用这些可执行文件, 别忘了再次运行上述命令。

> [!NOTE]
> 如果你已经安装了更高版本的 `ipex-llm[cpp]`，并希望同时升级这些可执行文件文件和  `.py` 文件，请先删除目录下的旧文件，然后使用 `init-llama-cpp`（Linux）或 `init-llama-cpp.bat`（Windows）重新初始化。

**现在，（在这个目录下）你可以按照 llama.cpp 的官方用法来执行 llama.cpp 的命令了。**

#### 运行时配置

要更高效地使用 Intel GPU 加速运行 `llama.cpp`，建议设置如下环境变量。

- **Linux 用户**:
  
  ```bash
  source /opt/intel/oneapi/setvars.sh
  export SYCL_CACHE_PERSISTENT=1
  # [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  # [optional] if you want to run on single GPU, use below command to limit GPU may improve performance
  export ONEAPI_DEVICE_SELECTOR=level_zero:0
  ```

- **Windows 用户**:

  请在 Miniforge Prompt 中运行下列命令。

  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  rem under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```

> [!TIP]
> 如果你的设备配备了多个 GPU， 而你只想在其中一个 GPU 上运行 llama.cpp，就需要设置环境变量 `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]`, 其中 `[gpu_id]` 是指定运行 `llama.cpp` 的 GPU 设备 ID。相关详情请参阅[多 GPU 选择指南](../Overview/KeyFeatures/multi_gpus_selection.md#2-oneapi-device-selector)。

> [!NOTE]
> 环境变量 `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` 用于控制是否使用*即时命令列表*将任务提交到 GPU。启动此变量通常可以提高性能，但也有例外情况。因此，建议你在启用和禁用该环境变量的情况下进行测试，以找到最佳的性能设置。更多相关细节请参考[此处文档](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html)。

### 3. 示例: 使用 IPEX-LLM 运行社区 GGUF 模型

这里我们提供一个简单的示例来展示如何使用 IPEX-LLM 运行社区 GGUF 模型。

#### 模型下载
运行之前, 你应该下载或复制社区的 GGUF 模型到你当前的目录。例如，[Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main) 的 `mistral-7b-instruct-v0.1.Q4_K_M.gguf`。

#### 运行量化模型

- **Linux 用户**:
  
  ```bash
  ./llama-cli -m mistral-7b-instruct-v0.1.Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -c 1024 -t 8 -e -ngl 99 --color
  ```

  > **Note**:
  >
  > 可以使用 `./llama-cli -h` 查看每个参数的详细含义。

- **Windows 用户**:

  请在 Miniforge Prompt 中运行以下命令。

  ```cmd
  llama-cli -m mistral-7b-instruct-v0.1.Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -c 1024 -t 8 -e -ngl 99 --color
  ```

  > **Note**:
  >
  > 可以使用 `llama-cli -h` 查看每个参数的详细含义。

#### 示例输出
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

### 故障排除

#### 1. 无法运行初始化脚本
如果你无法运行 `init-llama-cpp.bat`, 请确保在你的 conda 环境中已经安装了 `ipex-llm[cpp]`。如果你已安装, 请检查是否已激活正确的 conda 环境。此外，如果你使用的是 Windows，请确保是在提示终端中以管理员权限运行该脚本。

#### 2. `DeviceList is empty. -30 (PI_ERROR_INVALID_VALUE)` 错误
在 Linux 中, 当找不到以 `[ext_oneapi_level_zero]` 开头的设备时，会出现此错误。请确保你已经安装 level-zero，并在运行命令之前执行了 `/opt/intel/oneapi/setvars.sh`。

#### 3. `Prompt is too long` 错误
如果出现类似 `main: prompt is too long (xxx tokens, max xxx)` 的错误，请将 `-c` 参数设置为更大的数值，来支持更长的上下文内容。

#### 4. `gemm: cannot allocate memory on host` 错误 / `could not create an engine` 错误
如果在 Linux 上遇到 `oneapi::mkl::oneapi::mkl::blas::gemm: cannot allocate memory on host` 或 `could not create an engine` 错误，可能是因为你使用 pip 安装了 oneAPI 依赖项（例如 `pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0`）。建议换用 `apt` 来安装 oneAPI 依赖项以避免此问题。更多详情信息请参考[此处指南](./install_linux_gpu.md)。

#### 5. 无法量化模型
如果你遇到 `main: failed to quantize model from xxx`，请确保已经创建相关的输出目录。

#### 6. 模型加载时程序挂起
如果 `llm_load_tensors:  SYCL_Host buffer size =    xx.xx MiB`之后程序挂起，你可以在命令中添加 `--no-mmap`。

#### 7. 如何设置 `-ngl` 参数
`-ngl` 参数表示在显存中存储的网络层数量。如果你的显存足够，建议将所有层都放在 GPU 上，你可以将 `-ngl` 设置为一个较大的数值（例如 999）来实现这一目标。

如果 `-ngl` 设置为大于0且小于模型总层数的数值, 将采用 GPU + CPU 混合运行的模式。

#### 8. 如何指定 GPU
如果你的机器配备了多个 GPU，`llama.cpp` 默认会使用所有 GPU，这可能会导致原本可以在单个 GPU 上运行的模型推理变慢。你可以在命令中添加 `-sm none` 来仅使用一个 GPU。

此外，你也可以在执行命令前使用 `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]` 来指定要使用的 GPU 设备，更多详情信息请参阅[此处指南](../Overview/KeyFeatures/multi_gpus_selection.md#2-oneapi-device-selector)。

#### 9. 使用中文提示词时发生程序崩溃
如果你在 Windows 上运行 llama.cpp 程序时，发现程序在接受中文提示时崩溃或者输出异常，可以进入“区域设置 -> 管理 -> 更改系统区域设置”，勾选“Beta: 使用 Unicode UTF-8 提供全球语言支持”选项，然后重启计算机。

有关如何执行此操作的详细说明，请参阅[此问题](https://github.com/intel-analytics/ipex-llm/issues/10989#issuecomment-2105600469)。

#### 10. sycl7.dll 未找到错误
如果你在 Linux 或者 Windows 上遇到类似 `System Error: sycl7.dll not found` 的错误, 请根据操作系统进行下列检查:

1. Windows：是否已经安装了 conda 并激活了正确的 conda 环境，环境中是否已经使用 pip 安装了 oneAPI 依赖项
2. Linux：是否已经在运行 llama.cpp 命令前执行了 `source /opt/intel/oneapi/setvars.sh`。执行此 source 命令只在当前会话有效。

#### 11. 遇到输出乱码请先检查驱动
如果你遇到输出乱码，请检查 GPU 驱动版本是否 >= [31.0.101.5522](https://www.intel.cn/content/www/cn/zh/download/785597/823163/intel-arc-iris-xe-graphics-windows.html)。如果不是，请参照[这里](./install_linux_gpu.md#install-gpu-driver) 的说明更新你的 GPU 驱动。

#### 12. 为什么我的程序找不到 sycl 设备
如果你遇到 `GGML_ASSERT: C:/Users/Administrator/actions-runner/cpp-release/_work/llm.cpp/llm.cpp/llama-cpp-bigdl/ggml-sycl.cpp:18283: main_gpu_id<g_all_sycl_device_count` 错误或者类似错误，并且发现使用 `ls-sycl-device` 时没有任何输出，这是因为 llama.cpp 无法找到 sycl 设备。在某些笔记本电脑上，安装 ARC 驱动程序可能会导致被 Microsoft 强制安装 `OpenCL, OpenGL, and Vulkan Compatibility Pack`，这会无意中阻止系统定位 sycl 设备。这个问题可以通过在微软应用商店中手动卸载这个软件包来解决。

#### 13. 当系统中同时存在集成显卡和独立显卡时发生 Core Dump
如果你的 llama.cpp 设备日志中显示同时检测到了集成显卡和独立显卡，但未明确指定要使用哪一个，这可能会导致程序崩溃并出现 Core Dump。在这种情况下，你需要在运行 `llama-cli` 之前指定使用哪个 GPU 设备，例如 `export ONEAPI_DEVICE_SELECTOR=level_zero:0`。

#### 14. `Native API failed` 错误
在最新版本的 `ipex-llm` 中，对于未使用 `-c` 参数运行某些模型，可能会遇到 `native API failed` 的错误。只需添加 `-c xx` 参数即可解决此问题。

#### 15. `signal: bus error (core dumped)` 错误
如果你遇到此错误，请先检查你的 Linux 内核版本。较高版本的内核（例如 6.15）可能会导致此问题。你也可以参考[此问题](https://github.com/intel-analytics/ipex-llm/issues/10955)来查看是否有帮助。

#### 16. `backend buffer base cannot be NULL` 错误
如果你遇到`ggml-backend.c:96: GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL") failed`错误，可以在推理时传入参数`-c xx`，如`-c 1024`即可解决。