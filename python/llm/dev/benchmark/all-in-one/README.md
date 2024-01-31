# All in One Benchmark Test

All in one benchmark test allows users to test all the benchmarks and record them in a result CSV. Users can provide models and related information in `config.yaml`.

Before running, make sure to have [bigdl-llm](../../../README.md).

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
  - 'THUDM/chatglm-6b'
  - 'THUDM/chatglm2-6b'
  - 'meta-llama/Llama-2-7b-chat-hf'
  # - 'liuhaotian/llava-v1.5-7b' # requires a LLAVA_REPO_DIR env variables pointing to the llava dir; added only for gpu win related test_api now
local_model_hub: 'path to your local model hub'
warm_up: 1
num_trials: 3
num_beams: 1 # default to greedy search
low_bit: 'sym_int4' # default to use 'sym_int4' (i.e. symmetric int4)
batch_size: 1 # default to 1
in_out_pairs:
  - '32-32'
  - '1024-128'
test_api:
  - "transformer_int4"
  - "native_int4"
  - "optimize_model"
  - "pytorch_autocast_bf16"
  # - "transformer_autocast_bf16"
  # - "ipex_fp16_gpu" # on Intel GPU
  # - "transformer_int4_gpu"  # on Intel GPU
  # - "optimize_model_gpu"  # on Intel GPU
  # - "deepspeed_transformer_int4_cpu" # on Intel SPR Server
  # - "transformer_int4_gpu_win" # on Intel GPU for Windows (catch GPU peak memory)
cpu_embedding: False # whether put embedding to CPU (only avaiable now for gpu win related test_api)
plot_memory_vs_time_graph: False 
# Whether plot memory usage graph (Linux Only)
# The 'top' tool is used to obtain memory usage in Linux. However, due to different Linux distributions and versions of the top tool, the output format may vary. Therefore, you need to manually check if the memory usage obtained  correctly corresponds to the right location in the original output. After running the code, instructions will appear; you just need to follow the instructions.
```

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

All the outputs would be in ` ./output/ `.