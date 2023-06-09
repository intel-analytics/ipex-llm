# Inference Pipeline for GPT-NeoX Family Models in INT4 Data Type

In this example, we show a pipeline to conduct inference on a converted low-precision (int4) large language model in GPT-NeoX family, using `bigdl-llm`.

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all]
```

## Run Example
```bash
python ./gptneox.py --thread-num THREAD_NUM
```
arguments info:
- `--thread-num THREAD_NUM`: required argument defining the number of threads to use for inference. It is default to be `2`.
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: optional argument defining the huggingface repo id from which the GPT-NeoX family model is downloaded, or the path to the huggingface checkpoint folder for GPT-NeoX family model. It is default to be `'togethercomputer/RedPajama-INCITE-7B-Chat'`
- `--promp PROMPT`: optional argument defining the prompt to be infered. It is default to be `'Q: tell me something about Intel. A:'`.
