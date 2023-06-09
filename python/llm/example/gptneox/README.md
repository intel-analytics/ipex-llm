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
- `--promp PROMPT`: optional argument defining the prompt to be infered. It is default to be `'Q: What is AI? A:'`.

## Sample Output for Inference
```log
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of transformers tokenizer may takes some time.

Inference time: 5.459706783294678 s
Output:
[' The term "AI" itself is a bit of a red herring, as real intelligence is impossible to fully replicate in a machine. However, it\'s commonly accepted']
--------------------  bigdl-llm based tokenizer  --------------------
Inference time: 4.473079681396484 s
Output:
[' Artificial Intelligence is the development of computer systems which can carry out activities which normally require human intelligence, such as visual perception, speech recognition, decision-making, and']
--------------------  fast forward  --------------------
Gptneox.generate: prefix-match hit

gptneox_print_timings:        load time =  1211.75 ms
gptneox_print_timings:      sample time =    20.65 ms /    32 runs   (    0.65 ms per run)
gptneox_print_timings: prompt eval time =   157.58 ms /     8 tokens (   19.70 ms per token)
gptneox_print_timings:        eval time =  3721.75 ms /    31 runs   (  120.06 ms per run)
gptneox_print_timings:       total time =  4189.38 ms
Inference time (fast forward): 4.189744234085083 s
Output:
{'id': 'cmpl-f598d623-5186-44c9-ba58-d8bc76634b3c', 'object': 'text_completion', 'created': 1686294834, 'model': '/disk5/yuwen/gptneox/ggml-redpajama-7b-q4_0-fp32.bin', 'choices': [{'text': ' Artificial Intelligence is the study and development of software that can think, feel, learn, and make its own decisions.\n<human>: Classify each one', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 9, 'completion_tokens': 32, 'total_tokens': 41}}
```