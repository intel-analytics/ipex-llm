# LongBench Benchmark Test

LongBench Benchmark allows users to test LongBench benchmark and record them in some json files. Users can provide models and related information in `config.yaml` and `config` directory.

Before running, make sure to have [ipex-llm](../../../../../README.md) installed.

## Dependencies

```bash
pip install omegaconf
pip install datasets
pip install jieba
pip install fuzzywuzzy
pip install rouge
```

## config

### `config.yaml`

Config YAML file has following format

```yaml
# The name of the models you want to test
model_name:
  # - "mistral-7B-instruct-v0.2"
  - "llama2-7b-chat-4k"
  # - "chatglm4-9b"
  # - "qwen2-7b-instruct"

# whether or not to test the full-kv score
full_kv: True
# whether or not to open optimize_model
optimize_model: True
# dtype of the model
dtype: 'fp16'
# low bit of the model
low_bit: 'sym_int4'
# whether or not to use the 'e' version of the datasets
e: False

# the compress kv configs you want to test
compress_kv:
  - "ablation_c512_w32_k7_maxpool"
  - "ablation_c1024_w32_k7_maxpool"

# the datasets you want to test
datasets:
  - "multi_news"
  - "qasper"
  - "hotpotqa"
  - "trec"
  - "passage_count"
  - "lcc"
  # - "multifieldqa_zh"
  # - "dureader"
  # - "vcsum"
  # - "lsht"
  # - "passage_retrieval_zh"

```

### The `config` dir

Some json files is saved in the `config` dir. It can be divided into three kinds: about models, about datasets, and about compress-kv.

#### About Models

- `model2path.json`: This file saves the path to the models.

-  `model2maxlen.json`: This file saves the max length of the prompts of each model.

#### About datasets

- `dataset2maxlen.json`: The max length of the outputs of the models of each dataset.

- `dataset2prompt.json`: The format of prompts of each dataset.

#### About compress-kv

The rest json files are about compress-kv. 

## Run

There are two python files for users' call.

- `pred.py`: This script will give the output of the models configged in the `config.yaml`

- `eval.py`: This script calculates the score of each case.

> [!Note]
>
> To test the models and get the score in a row, please run `test_and_eval.sh`