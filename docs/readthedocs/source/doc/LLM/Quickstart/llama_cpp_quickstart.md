# Run llama.cpp with IPEX-LLM on Intel GPU 

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) prvoides fast LLM inference in in pure C++ across a variety of hardware; you can now use the C++ interface of [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `llama.cpp` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of running LLaMA2-7B on Intel Arc GPU below.

<video src="https://llm-assets.readthedocs.io/en/latest/_images/llama-cpp-arc.mp4" width="100%" controls></video>

## Quick Start
This quickstart guide walks you through installing and running `llama.cpp` with `ipex-llm`.

### 0 Prerequisites
IPEX-LLM's support for `llama.cpp` now is available for Linux system and Windows system.

#### Linux
For Linux system, we recommend Ubuntu 20.04 or later (Ubuntu 22.04 is preferred).

Visit the [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html), follow [Install Intel GPU Driver](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-intel-gpu-driver) and [Install oneAPI](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-oneapi) to install GPU driver and Intel® oneAPI Base Toolkit 2024.0.

#### Windows
Visit the [Install IPEX-LLM on Windows with Intel GPU Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html), and follow [Install Prerequisites](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-prerequisites) to install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) Community Edition, latest [GPU driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html) and Intel® oneAPI Base Toolkit 2024.0.

**Note**: IPEX-LLM backend only supports the more recent GPU drivers. Please make sure your GPU driver version is equal or newer than `31.0.101.5333`, otherwise you might find gibberish output.

### 1 Install IPEX-LLM for llama.cpp

To use `llama.cpp` with IPEX-LLM, first ensure that `ipex-llm[cpp]` is installed.
```cmd
conda create -n llm-cpp python=3.11
conda activate llm-cpp
pip install --pre --upgrade ipex-llm[cpp]
```

**After the installation, you should have created a conda environment, named `llm-cpp` for instance, for running `llama.cpp` commands with IPEX-LLM.**

### 2 Setup for running llama.cpp

First you should create a directory to use `llama.cpp`, for instance, use following command to create a `llama-cpp` directory and enter it.
```cmd
mkdir llama-cpp
cd llama-cpp
```

#### Initialize llama.cpp with IPEX-LLM

Then you can use following command to initialize `llama.cpp` with IPEX-LLM:
```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash
      
         init-llama-cpp

      After ``init-llama-cpp``, you should see many soft links of ``llama.cpp``'s executable files and a ``convert.py`` in current directory.

      .. image:: https://llm-assets.readthedocs.io/en/latest/_images/init_llama_cpp_demo_image.png

   .. tab:: Windows

      Please run the following command with **administrator privilege in Anaconda Prompt**.

      .. code-block:: bash
      
         init-llama-cpp.bat

      After ``init-llama-cpp.bat``, you should see many soft links of ``llama.cpp``'s executable files and a ``convert.py`` in current directory.

      .. image:: https://llm-assets.readthedocs.io/en/latest/_images/init_llama_cpp_demo_image_windows.png

```

```eval_rst
.. note::

   ``init-llama-cpp`` will create soft links of llama.cpp's executable files to current directory, if you want to use these executable files in other places, don't forget to run above commands again.
```

**Now you can use these executable files by standard llama.cpp's usage.**

### 3 Example: Running community GGUF models with IPEX-LLM

Here we provide a simple example to show how to run a community GGUF model with IPEX-LLM.

#### Set Environment Variables

Configure oneAPI variables by running the following command:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         source /opt/intel/oneapi/setvars.sh

   .. tab:: Windows

      .. note::

      This is a required step for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

      .. code-block:: bash

         call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

```

#### Model Download
Before running, you should download or copy community GGUF model to your current directory. For instance,  `mistral-7b-instruct-v0.1.Q4_K_M.gguf` of [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main).

#### Run the quantized model

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash
      
         ./main -m mistral-7b-instruct-v0.1.Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -t 8 -e -ngl 33 --color
      
      .. note::

      For more details about meaning of each parameter, you can use ``./main -h``.

   .. tab:: Windows

      .. code-block:: bash

         main.exe -m mistral-7b-instruct-v0.1.Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -t 8 -e -ngl 33 --color

      .. note::

      For more details about meaning of each parameter, you can use ``main.exe -h``.
```

#### Sample Output
```
Log start
main: build = 1 (38bcbd4)
main: built with Intel(R) oneAPI DPC++/C++ Compiler 2024.0.0 (2024.0.0.20231017) for x86_64-unknown-linux-gnu
main: seed  = 1710359960
ggml_init_sycl: GGML_SYCL_DEBUG: 0
ggml_init_sycl: GGML_SYCL_F16: no
found 8 SYCL devices:
|ID| Name                                        |compute capability|Max compute units|Max work group|Max sub group|Global mem size|
|--|---------------------------------------------|------------------|-----------------|--------------|-------------|---------------|
| 0|               Intel(R) Arc(TM) A770 Graphics|               1.3|              512|          1024|           32|    16225243136|
| 1|               Intel(R) FPGA Emulation Device|               1.2|               32|      67108864|           64|    67181625344|
| 2|         13th Gen Intel(R) Core(TM) i9-13900K|               3.0|               32|          8192|           64|    67181625344|
| 3|               Intel(R) Arc(TM) A770 Graphics|               3.0|              512|          1024|           32|    16225243136|
| 4|               Intel(R) Arc(TM) A770 Graphics|               3.0|              512|          1024|           32|    16225243136|
| 5|                    Intel(R) UHD Graphics 770|               3.0|               32|           512|           32|    53745299456|
| 6|               Intel(R) Arc(TM) A770 Graphics|               1.3|              512|          1024|           32|    16225243136|
| 7|                    Intel(R) UHD Graphics 770|               1.3|               32|           512|           32|    53745299456|
detect 2 SYCL GPUs: [0,6] with Max compute units:512
llama_model_loader: loaded meta data with 20 key-value pairs and 291 tensors from ~/mistral-7b-instruct-v0.1.Q4_K_M.gguf (version GGUF V2)
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
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attm      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 7.24 B
llm_load_print_meta: model size       = 4.07 GiB (4.83 BPW) 
llm_load_print_meta: general.name     = mistralai_mistral-7b-instruct-v0.1
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
llm_load_tensors: ggml ctx size =    0.33 MiB
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:      SYCL0 buffer size =  2113.28 MiB
llm_load_tensors:      SYCL6 buffer size =  1981.77 MiB
llm_load_tensors:  SYCL_Host buffer size =    70.31 MiB
...............................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      SYCL0 KV buffer size =    34.00 MiB
llama_kv_cache_init:      SYCL6 KV buffer size =    30.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:  SYCL_Host input buffer size   =    10.01 MiB
llama_new_context_with_model:      SYCL0 compute buffer size =    73.00 MiB
llama_new_context_with_model:      SYCL6 compute buffer size =    73.00 MiB
llama_new_context_with_model:  SYCL_Host compute buffer size =     8.00 MiB
llama_new_context_with_model: graph splits (measure): 3
system_info: n_threads = 8 / 32 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | 
sampling: 
        repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
generate: n_ctx = 512, n_batch = 512, n_predict = 32, n_keep = 1
 Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun exploring the world around her. Her parents were kind and let her do what she wanted, as long as she stayed safe.
One day, the little
llama_print_timings:        load time =   10096.78 ms
llama_print_timings:      sample time =     x.xx ms /    32 runs   (   xx.xx ms per token,  xx.xx tokens per second)
llama_print_timings: prompt eval time =    xx.xx ms /    31 tokens (   xx.xx ms per token,  xx.xx tokens per second)
llama_print_timings:        eval time =    xx.xx ms /    31 runs   (   xx.xx ms per token,  xx.xx tokens per second)
llama_print_timings:       total time =    xx.xx ms /    62 tokens
Log end
```

### Troubleshooting

#### Fail to quantize model
If you encounter `main: failed to quantize model from xxx`, please make sure you have created related output directory.

#### Program hang during model loading
If your program hang after `llm_load_tensors:  SYCL_Host buffer size =    xx.xx MiB`, you can add `--no-mmap` in your command.

#### How to set `-ngl` parameter
`-ngl` means the number of layers to store in VRAM. If your VRAM is enough, we recommend putting all layers on GPU, you can just set `-ngl` to a large number like 999 to achieve this goal.
