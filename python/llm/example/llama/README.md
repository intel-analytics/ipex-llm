# Inference Pipeline for Llama Family Models in INT4 Data Type

In this example, we show a pipeline to conduct inference on a converted low-precision (int4) large language model in Llama family, using `bigdl-llm`.

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all]
```

## Run Example
```bash
python ./llama.py --thread-num THREAD_NUM
```
arguments info:
- `--thread-num THREAD_NUM`: required argument defining the number of threads to use for inference. It is default to be `2`.
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: optional argument defining the huggingface repo id from which the Llama family model is downloaded, or the path to the huggingface checkpoint folder for Llama family model. It is default to be `'decapoda-research/llama-7b-hf'`
- `--promp PROMPT`: optional argument defining the prompt to be infered. It is default to be `'Q: tell me something about intel. A:'`.
