# All in One Benchmark Test
All in one benchmark test allows users to test all the benchmarks and record them in a result CSV. Users can provide models and related information in `config.yaml`.

Before running, make sure to have [bigdl-llm](../../../README.md) installed.

## Config
Config YAML file has following format
```yaml
model_name: model_path
# following is an example, with model name llama2
llama2: /path/to/llama2
```

## Run
run `python run.py`, this will output results to `results.csv`.