# All in One Benchmark Test
All in one benchmark test allows users to test all the benchmarks and record them in a result CSV. Users can provide models and related information in `config.yaml`.

Before running, make sure to have [bigdl-llm](../../../README.md) and [bigdl-nano](../../../../nano/README.md) installed.

## Config
Config YAML file has following format
```yaml
repo_id:
  - 'THUDM/chatglm-6b'
  - 'THUDM/chatglm2-6b'
  - 'meta-llama/Llama-2-7b-chat-hf'
local_model_hub: 'path to your local model hub'
warm_up: 1
num_trials: 3
in_out_pairs:
  - '32-32'
  - '1024-128'
test_api:
  - "transformer_int4"
  - "native_int4"
  - "optimize_model"
  - "pytorch_autocast_bf16"
  # - "transformer_int4_gpu"  # on arc
  # - "optimize_model_gpu"  # on arc
```

## Run
run `python run.py`, this will output results to `results.csv`.

For SPR performance, run `bash run-spr.sh`.
> **Note**
>
> In `run-spr.sh`, we set optimal environment varaible by `source bigdl-nano-init -c`, `-c` stands for disabling jemalloc. Enabling jemalloc may lead to latency increasement after multiple trials.
>
> The value of `OMP_NUM_THREADS` should be the same as the cpu cores specified by `numactl -C`.

For ARC performance, run `bash run-arc.sh`.
