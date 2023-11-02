# All in One Benchmark Test
All in one benchmark test allows users to test all the benchmarks and record them in a result CSV. Users can provide models and related information in `config.yaml`.

Before running, make sure to have [bigdl-llm](../../../README.md).

## Dependencies
```bash
pip install omegaconf
pip install pandas
```

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
num_beams: 1 # default to greedy search
in_out_pairs:
  - '32-32'
  - '1024-128'
test_api:
  - "transformer_int4"
  - "native_int4"
  - "optimize_model"
  - "pytorch_autocast_bf16"
  # - "ipex_fp16_gpu" # on Intel GPU
  # - "transformer_int4_gpu"  # on Intel GPU
  # - "optimize_model_gpu"  # on Intel GPU
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
