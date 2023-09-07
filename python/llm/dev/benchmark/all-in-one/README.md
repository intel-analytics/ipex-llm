# All in One Benchmark Test
All in one benchmark test allows users to test all the benchmarks and record them in a result CSV. Users can provide models and related information in `config.yaml`.

Before running, make sure to have [bigdl-llm](../../../README.md) installed.

## Config
Config YAML file has following format
```yaml
repo_id:
  - 'THUDM/chatglm2-6b'
  - 'meta-llama/Llama-2-7b-chat-hf'
local_model_hub: 'path to your local model hub'
warm_up: 1
num_trials: 3
test_case: 'transformer_int4' # or 'optimize_model', currently works when run on GPU
```

## Run on CPU
run `python run.py`, this will output results to `results.csv`.

For SPR performance, run `bash run-spr.sh`.

## Run on GPU
Configures OneAPI environment variables:
```bash
source /opt/intel/oneapi/setvars.sh
```

For optimal performance on Arc, it is recommended to set several environment variables.
```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

Run `python run_gpu.py`, this will output results to `results.csv`.
