# Inference Pipeline for GPT-NeoX Family Model in INT4 data type

In this example, we show a pipeline to conduct inference on a converted low-precision (int4) large language model in GPT-NeoX family, using `bigdl-llm`.

## Prepare Environment
```bash
conda create -n bigdl-llm python=3.9
conda activate bigdl-llm

pip install bigdl-llm[all]
```

## Run Example
usage: gptneox.py [-h] [--repo-id-or-model-path REPO_ID_OR_MODEL_PATH] [--thread-num THREAD_NUM] [--prompt PROMPT]
```bash
python ./gptneox.py --thread-num THREAD_NUM
```
