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

Download IPEX-LLM Llama.cpp portable zip for Windows users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

### Step 2: Runtime Configuration

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- To use GPU acceleration, several environment variables are required or recommended before running `llama.cpp`.
```cmd
set SYCL_CACHE_PERSISTENT=1
```
- For multi-GPUs user, go to [Tips](#select-specific-gpu-to-run-llamacpp-when-multiple-ones-are-available) for how to select specific GPU.

### Step 3: Run GGUF models

Here we provide a simple example to show how to run a community GGUF model with IPEX-LLM.  

#### Model Download
Before running, you should download or copy community GGUF model to your current directory. For instance,  `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` of [bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf).

#### Run GGUF model
  ```cmd
  llama-cli.exe -m D:\llm-models\gguf\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf -p "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Question:The product of the ages of three teenagers is 4590. How old is the oldest? a. 18 b. 19 c. 15 d. 17 Assistant: <think>" -n 2048  -t
8 -e -ngl 99 --color -c 2500 --temp 0
  ```
Outputs:
  ```
main: llama backend init
main: load the model and apply lora adapter, if any
llama_load_model_from_file: using device SYCL0 (Intel(R) Arc(TM) Graphics) - 12949 MiB free
llama_model_loader: loaded meta data with 30 key-value pairs and 339 tensors from D:\llm-models\gguf\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 Distill Qwen 7B
llama_model_loader: - kv   3:                           general.basename str              = DeepSeek-R1-Distill-Qwen
llama_model_loader: - kv   4:                         general.size_label str              = 7B
llama_model_loader: - kv   5:                          qwen2.block_count u32              = 28
llama_model_loader: - kv   6:                       qwen2.context_length u32              = 131072
llama_model_loader: - kv   7:                     qwen2.embedding_length u32              = 3584
llama_model_loader: - kv   8:                  qwen2.feed_forward_length u32              = 18944
llama_model_loader: - kv   9:                 qwen2.attention.head_count u32              = 28
llama_model_loader: - kv  10:              qwen2.attention.head_count_kv u32              = 4
llama_model_loader: - kv  11:                       qwen2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  12:     qwen2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  14:                         tokenizer.ggml.pre str              = deepseek-r1-qwen
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,152064]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  16:                  tokenizer.ggml.token_type arr[i32,152064]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  17:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 151646
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 151643
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  21:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  22:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  23:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                          general.file_type u32              = 15
llama_model_loader: - kv  26:                      quantize.imatrix.file str              = /models_out/DeepSeek-R1-Distill-Qwen-...
llama_model_loader: - kv  27:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  28:             quantize.imatrix.entries_count i32              = 196
llama_model_loader: - kv  29:              quantize.imatrix.chunks_count i32              = 128
llama_model_loader: - type  f32:  141 tensors
llama_model_loader: - type q4_K:  169 tensors
llama_model_loader: - type q6_K:   29 tensors
llm_load_vocab: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
llm_load_vocab: special tokens cache size = 22
llm_load_vocab: token to piece cache size = 0.9310 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen2
llm_load_print_meta: vocab type       = BPE
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
llm_load_print_meta: n_vocab          = 152064
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 3584
llm_load_print_meta: n_layer          = 28
llm_load_print_meta: n_head           = 28
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 7
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18944
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 7.62 B
llm_load_print_meta: model size       = 4.36 GiB (4.91 BPW)
llm_load_print_meta: general.name     = DeepSeek R1 Distill Qwen 7B
llm_load_print_meta: BOS token        = 151646 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 151643 '<｜end▁of▁sentence｜>'
llm_load_print_meta: EOT token        = 151643 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 151643 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: FIM PRE token    = 151659 '<|fim_prefix|>'
llm_load_print_meta: FIM SUF token    = 151661 '<|fim_suffix|>'
llm_load_print_meta: FIM MID token    = 151660 '<|fim_middle|>'
llm_load_print_meta: FIM PAD token    = 151662 '<|fim_pad|>'
llm_load_print_meta: FIM REP token    = 151663 '<|repo_name|>'
llm_load_print_meta: FIM SEP token    = 151664 '<|file_sep|>'
llm_load_print_meta: EOG token        = 151643 '<｜end▁of▁sentence｜>'
llm_load_print_meta: EOG token        = 151662 '<|fim_pad|>'
llm_load_print_meta: EOG token        = 151663 '<|repo_name|>'
llm_load_print_meta: EOG token        = 151664 '<|file_sep|>'
llm_load_print_meta: max token length = 256
get_memory_info: [warning] ext_intel_free_memory is not supported (export/set ZES_ENABLE_SYSMAN=1 to support), use total memory as free memory
llm_load_tensors: offloading 28 repeating layers to GPU
llm_load_tensors: offloading output layer to GPU
llm_load_tensors: offloaded 29/29 layers to GPU
llm_load_tensors:        SYCL0 model buffer size =  4168.09 MiB
llm_load_tensors:   CPU_Mapped model buffer size =   292.36 MiB
..................................................................................
llama_new_context_with_model: n_seq_max     = 1
llama_new_context_with_model: n_ctx         = 2528
llama_new_context_with_model: n_ctx_per_seq = 2528
llama_new_context_with_model: n_batch       = 2528
llama_new_context_with_model: n_ubatch      = 2528
llama_new_context_with_model: flash_attn    = 0
llama_new_context_with_model: freq_base     = 10000.0
llama_new_context_with_model: freq_scale    = 1
llama_new_context_with_model: n_ctx_per_seq (2528) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
[SYCL] call ggml_check_sycl
ggml_check_sycl: GGML_SYCL_DEBUG: 0
ggml_check_sycl: GGML_SYCL_F16: no
Found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |
    |
|  |                   |                                       |       |compute|Max work|sub  |mem    |
    |
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

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Question:The product of the ages of three teenagers is 4590. How old is the oldest? a. 18 b. 19 c. 15 d. 17 Assistant: <think>

Okay, so I have this problem where the product of the ages of three teenagers is 4590, and I need to figure out how old the oldest one is. The options are 18, 19, 15, or 17. Hmm, let's break this down step by step.

First, I know that teenagers are typically between 13 and 19 years old. So, the ages we're dealing with are all in that range. The product of their ages is 4590, which is a pretty big number. I need to find three numbers between 13 and 19 that multiply together to give 4590.

Maybe I should start by factoring 4590 to see what numbers I'm working with. Let's see, 4590. I can divide by 10 first because it's easy. 4590 divided by 10 is 459. So, 4590 = 10 × 459. Now, 459 is a bit tricky. Let's factor that further. 459 divided by 3 is 153. So, 459 = 3 × 153. Continuing, 153 divided by 3 is 51, so 153 = 3 × 51. And 51 divided by 3 is 17, so 51 = 3 × 17. Putting it all together, 4590 = 10 × 3 × 3 × 3 × 17.

Wait, but 10 is not a teenager's age. Teenagers are between 13 and 19, so 10 is too young. Maybe I need to combine some of these factors to get ages within the teenager range. Let's see, 10 can be broken down into 2 × 5. So, 4590 = 2 × 5 × 3 × 3 × 3 × 17.

Now, I need to group these prime factors into three numbers between 13 and 19. Let's list out the prime factors: 2, 3, 3, 3, 5, 17. Hmm, 17 is a prime number, so that has to be one of the ages. So, one of the teenagers is 17 years old.

Now, I need to find two more ages from the remaining factors: 2, 3, 3, 3, 5. Let's see, 2 × 3 = 6, which is too young. 3 × 3 = 9, also too young. 3 × 5 = 15, which is within the teenager range. So, one age is 15.

Now, the remaining factors are 2, 3, and 5. Let's multiply them together: 2 × 3 × 5 = 30. Wait, 30 is too old for a teenager. That's a problem. Maybe I need to combine them differently. Let's see, 2 × 5 = 10, which is too young. 3 × 5 = 15, which we've already used. Hmm, maybe I need to adjust my grouping.

Wait, perhaps I can combine 2 and 3 to make 6, but that's too young. Alternatively, maybe I can combine 3 and 3 to make 9, but that's still too young. Hmm, this is tricky. Maybe I need to consider that one of the ages could be 18 or 19, which are also teenagers.

Let me try a different approach. Let's list all possible combinations of three numbers between 13 and 19 that multiply to 4590. Starting with 17, since it's a prime factor, it has to be one of them. So, 17 is one age. Now, I need two more ages from the remaining factors: 2, 3, 3, 3, 5.

If I take 15 (which is 3 × 5), then the remaining factors are 2 and 3. Multiplying those gives 6, which is too young. So that doesn't work. Alternatively, if I take 18 (which is 2 × 3 × 3), then the remaining factors are 3 and 5. Multiplying those gives 15. So, the ages would be 17, 18, and 15. That adds up correctly: 17 × 18 × 15 = 4590.

Wait, let me double-check that multiplication. 17 × 18 is 306, and 306 × 15 is indeed 4590. So, the ages are 15, 17, and
 18. Therefore, the oldest is 18.

But wait, the options include 19 as well. Did I miss a combination where one of the ages is 19? Let's see. If I try to include 19, I need to see if 4590 is divisible by 19. Let's divide 4590 by 19. 19 × 241 is 4579, which is close but not exactly 4590. So, 19 doesn't divide evenly into 4590. Therefore, 19 can't be one of the ages.

So, the only possible ages are 15, 17, and 18, making the oldest 18. Therefore, the answer should be 18.
</think>

The oldest teenager is 18 years old.

<answer>18</answer> [end of text]


llama_perf_sampler_print:    sampling time =     xxx.xx ms /  1386 runs   (    x.xx ms per token, xxxxx.xx tokens per second)
llama_perf_context_print:        load time =   xxxxx.xx ms
llama_perf_context_print: prompt eval time =     xxx.xx ms /   129 tokens (    x.xx ms per token,   xxx.xx tokens per second)
llama_perf_context_print:        eval time =   xxxxx.xx ms /  1256 runs   (   xx.xx ms per token,    xx.xx tokens per second)
llama_perf_context_print:       total time =   xxxxx.xx ms /  1385 tokens
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
So disable the iGPU will can get the best performance. Visit [Multi-GPUs usage](#multi-gpus-usage) for defaults.  
If you still want to disable this check, you can run `set SYCL_DEVICE_CHECK=0`(Windows user) or `export SYCL_DEVICE_CHECK=0`(Linux user).  

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
To enable SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS, you can run  
`set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`(Windows user)   
or  
`export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`(Linux user)
> [!NOTE]
> The environment variable SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS determines the usage of immediate command lists for task submission to the GPU. While this mode typically enhances performance, exceptions may occur. Please consider experimenting with and without this environment variable for best performance. For more details, you can refer to [this article](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html).  


