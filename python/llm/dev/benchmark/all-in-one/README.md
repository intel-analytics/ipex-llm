# All in One Benchmark Test

All in one benchmark test allows users to test all the benchmarks and record them in a result CSV. Users can provide models and related information in `config.yaml`.

Before running, make sure to have [ipex-llm](../../../../../README.md) installed.

## Dependencies

```bash
pip install omegaconf
pip install pandas
```

Install gperftools to use libtcmalloc.so for MAX GPU to get better performance:

```bash
conda install -c conda-forge -y gperftools=2.10
```

## Config

Config YAML file has following format

```yaml
repo_id:
  # - 'THUDM/chatglm2-6b'
  - 'meta-llama/Llama-2-7b-chat-hf'
  # - 'liuhaotian/llava-v1.5-7b' # requires a LLAVA_REPO_DIR env variables pointing to the llava dir; added only for gpu win related test_api now
local_model_hub: 'path to your local model hub'
warm_up: 1 # must set >=2 when run "pipeline_parallel_gpu" test_api
num_trials: 3
num_beams: 1 # default to greedy search
low_bit: 'sym_int4' # default to use 'sym_int4' (i.e. symmetric int4)
batch_size: 1 # default to 1
in_out_pairs:
  - '32-32'
  - '1024-128'
test_api:
  - "transformer_int4_fp16_gpu"             # on Intel GPU, transformer-like API, (qtype=int4), (dtype=fp16)
  # - "transformer_int4_fp16_gpu_win"       # on Intel GPU for Windows, transformer-like API, (qtype=int4), (dtype=fp16)
  # - "transformer_int4_gpu"                # on Intel GPU, transformer-like API, (qtype=int4), (dtype=fp32)
  # - "transformer_int4_gpu_win"            # on Intel GPU for Windows, transformer-like API, (qtype=int4), (dtype=fp32)
  # - "transformer_int4_loadlowbit_gpu_win" # on Intel GPU for Windows, transformer-like API, (qtype=int4), use load_low_bit API. Please make sure you have used the save.py to save the converted low bit model
  # - "bigdl_fp16_gpu"                      # on Intel GPU, use ipex-llm transformers API, (dtype=fp16), (qtype=fp16)
  # - "optimize_model_gpu"                  # on Intel GPU, can optimize any pytorch models include transformer model
  # - "deepspeed_optimize_model_gpu"        # on Intel GPU, deepspeed autotp inference
  # - "pipeline_parallel_gpu"               # on Intel GPU, pipeline parallel inference
  # - "speculative_gpu"                     # on Intel GPU, inference with self-speculative decoding
  # - "transformer_int4"                    # on Intel CPU, transformer-like API, (qtype=int4)
  # - "native_int4"                         # on Intel CPU
  # - "optimize_model"                      # on Intel CPU, can optimize any pytorch models include transformer model
  # - "pytorch_autocast_bf16"               # on Intel CPU
  # - "transformer_autocast_bf16"           # on Intel CPU
  # - "bigdl_ipex_bf16"                     # on Intel CPU, (qtype=bf16)
  # - "bigdl_ipex_int4"                     # on Intel CPU, (qtype=int4)
  # - "bigdl_ipex_int8"                     # on Intel CPU, (qtype=int8)
  # - "speculative_cpu"                     # on Intel CPU, inference with self-speculative decoding
  # - "deepspeed_transformer_int4_cpu"      # on Intel CPU, deepspeed autotp inference
  # - "transformer_int4_fp16_lookahead_gpu" # on Intel GPU, transformer-like API, with lookahead, (qtype=int4), (dtype=fp16)
cpu_embedding: False # whether put embedding to CPU
streaming: False # whether output in streaming way (only available now for gpu win related test_api)
use_fp16_torch_dtype: True # whether use fp16 for non-linear layer (only available now for "pipeline_parallel_gpu" test_api)
lookahead: 3
max_matching_ngram_size: 2
task: 'continuation' # when test_api is "transformer_int4_fp16_lookahead_gpu", task could be 'QA', 'continuation' or 'summarize' 

```

## (Optional) Save model in low bit
If you choose the `transformer_int4_loadlowbit_gpu_win` test API, you will need to save the model in low bit first.

Run `python save.py` will save all models declared in `repo_id` list into low bit models under `local_model_hub` folder.

## Run

run `python run.py`, this will output results to `results.csv`.

For SPR performance, run `bash run-spr.sh`.

> **Note**
>
> The value of `OMP_NUM_THREADS` should be the same as the cpu cores specified by `numactl -C`.

> **Note**
>
> Please install torch nightly version to avoid `Illegal instruction (core dumped)` issue, you can follow the following command to install: `pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu`

For ARC performance, run `bash run-arc.sh`.

For MAX GPU performance, run `bash run-max-gpu.sh`.
