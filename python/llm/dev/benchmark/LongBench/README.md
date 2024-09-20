# LongBench Benchmark Test

LongBench is the first benchmark for bilingual, multitask, and comprehensive assessment of long context understanding capabilities of large language models. This benchmark implementation is adapted from [THUDM/LongBench](https://github.com/THUDM/LongBench) and [SnapKV/experiments/LongBench](https://github.com/FasterDecoding/SnapKV/tree/main/experiments/LongBench).


## Environment Preparation

Before running, make sure to have [ipex-llm](../../../../../README.md) installed.

```bash
pip install omegaconf
pip install datasets
pip install jieba
pip install fuzzywuzzy
pip install rouge
```

### Load Data

You can download and load the LongBench data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/THUDM/LongBench)):

```python

from datasets import load_dataset

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
    data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')

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

# whether test the full-kv
full_kv: True
# Whether apply model optimization
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

The rest JSON files are compress-kv test configurations.

## Run

There are two python files for users' call.

1. Configure the `config.yaml` and run `pred.py` and you can obtain the output of the model under `pred/` folder corresponding to the model name.

2. Run the evaluation code `eval.py`, you can get the evaluation results on all datasets in `result.json`.

> [!Note]
>
> To test the models and get the score in a row, please run `test_and_eval.sh`

## Citation

```bibtex
@article{bai2023longbench,
  title={LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding},
  author={Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  journal={arXiv preprint arXiv:2308.14508},
  year={2023}
}

```