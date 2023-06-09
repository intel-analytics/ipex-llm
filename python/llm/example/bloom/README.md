# Inference Pipeline for BLOOM Family Models in INT4 Data Type

In this example, we show a pipeline to conduct inference on a converted low-precision (int4) large language model in BLOOM family, using `bigdl-llm`.

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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: optional argument defining the huggingface repo id from which the BLOOM family model is downloaded, or the path to the huggingface checkpoint folder for BLOOM family model. It is default to be `'bigscience/bloomz-7b1'`
- `--promp PROMPT`: optional argument defining the prompt to be infered. It is default to be `'Q: What is AI? A:'`.

## Sample Output for Inference
```log
inference:    mem per token = 24471324 bytes
inference:      sample time =     xxxx ms
inference: evel prompt time =     xxxx ms / 5 tokens / xxxx ms per token
inference:     predict time =     xxxx ms / 2 tokens / xxxx ms per token
inference:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-bb268afb-e088-4729-91fa-8746ea4fa706', 'object': 'text_completion', 'created': 1686294707, 'model': '/disk5/yuwen/bloom/bigdl_llm_bloom_q4_0.bin', 'choices': [{'text': 'Q: What is AI? A: artificial intelligence</s>', 'index': 0, 'logprobs': None, 'finish_reason': None}], 'usage': {'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None}}
```