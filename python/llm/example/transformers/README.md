# INT4 Inference Pipeline for Large Language Model using BigDL-LLM Transformers-like API

In this example, we show a pipeline to convert a large language model to low precision (INT4), and then conduct inference on the converted INT4 model, using BigDL-LLM transformers-like API.

> **Note**: BigDL-LLM currently supports model family LLaMA, GPT-NeoX, BLOOM and StarCoder.

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all]
```

## Run Example
```bash
python ./int4_pipeline.py --thread-num THREAD_NUM --model-family MODEL_FAMILY
```
arguments info:
- `--thread-num THREAD_NUM`: **required** argument defining the number of threads to use for inference. It is default to be `2`.
- `--model-family MODEL_FAMILY`: **required** argument defining the model family of the large language model (supported option: `'llama'`, `'gptneox'`, `'bloom'`, `'starcoder'`). It is default to be `'llama'`.
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: optional argument defining the huggingface repo id from which the large language model is downloaded, or the path to the huggingface checkpoint folder for the model.

  - When model family is `'llama'`, it is default to be `'decapoda-research/llama-7b-hf'`.
  - When model family is `'gptneox'`, it is default to be `'togethercomputer/RedPajama-INCITE-7B-Chat'`.
  - When model family is `'bloom'`, it is default to be `'bigscience/bloomz-7b1'`.
  - When model family is `'starcoder'`, it is default to be `'bigcode/gpt_bigcode-santacoder'`.

  > **Note** `REPO_ID_OR_MODEL_PATH` should fits your inputed `MODEL_FAMILY`.
- `--promp PROMPT`: optional argument defining the prompt to be infered. It is default to be `'Q: What is CPU? A:'`.

## Sample Output for Inference
### Model family LLaMA
```log
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may take some time.

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LLaMATokenizer'. 
The class this function is called from is 'LlamaTokenizer'.
Inference time: xxxx s
Output:
["The Central Processing Unit (CPU) is the brains of your computer, and is also known as the microprocessor. It's where all the action"]
--------------------  bigdl-llm based tokenizer  --------------------
Inference time: xxxx s
Output:
[' It’s the acronym for “Central Processing Unit,” and in modern personal computers it means a single microprocessor chip that is used to control various']
--------------------  fast forward  --------------------
Llama.generate: prefix-match hit

llama_print_timings:        load time =     xxxx ms
llama_print_timings:      sample time =     xxxx ms /    32 runs   (    xxxx ms per token)
llama_print_timings: prompt eval time =     xxxx ms /     8 tokens (    xxxx ms per token)
llama_print_timings:        eval time =     xxxx ms /    31 runs   (    xxxx ms per token)
llama_print_timings:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-5aa68120-c94b-4433-92f4-b75cc323c22f', 'object': 'text_completion', 'created': 1686557904, 'model': './bigdl_llm_llama_q4_0.bin', 'choices': [{'text': ' It’s a small, compact computer unit that runs on a single chip. This can be connected to various peripheral devices, including printers and displays', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 9, 'completion_tokens': 32, 'total_tokens': 41}}
```

### Model family GPT-NeoX
```log
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may take some time.

Inference time: xxxx s
Output:
[' The Central Processing Unit, or CPU, is the component of a computer that executes all instructions for carrying out different functions. It is the brains of the operation, and']
--------------------  bigdl-llm based tokenizer  --------------------
Inference time: xxxx s
Output:
[' Central processing unit, also known as processor, is a specialized microchip designed to execute all the instructions of computer programs rapidly and efficiently. Most personal computers have one or']
--------------------  fast forward  --------------------
Gptneox.generate: prefix-match hit

gptneox_print_timings:        load time =     xxxx ms
gptneox_print_timings:      sample time =     xxxx ms /    32 runs   (    xxxx ms per run)
gptneox_print_timings: prompt eval time =     xxxx ms /     8 tokens (    xxxx ms per token)
gptneox_print_timings:        eval time =     xxxx ms /    31 runs   (    xxxx ms per run)
gptneox_print_timings:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-a20fc4a1-3a00-4e77-a6cf-0dd0da6b9a59', 'object': 'text_completion', 'created': 1686557799, 'model': './bigdl_llm_gptneox_q4_0.bin', 'choices': [{'text': ' Core Processing Unit  or Central Processing Unit  is the brain of your computer, system software runs on it and handles all important tasks in your computer. i', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 9, 'completion_tokens': 32, 'total_tokens': 41}}
```

### Model family BLOOM
```log
inference:    mem per token = 24471324 bytes
inference:      sample time =     xxxx ms
inference: evel prompt time =     xxxx ms / 5 tokens / xxxx ms per token
inference:     predict time =     xxxx ms / 3 tokens / xxxx ms per token
inference:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-a0ab2953-e08c-449c-b476-e21ad5bb84b0', 'object': 'text_completion', 'created': 1686557434, 'model': './bigdl_llm_bloom_q4_0.bin', 'choices': [{'text': 'Q: What is CPU? A: central processing unit</s>', 'index': 0, 'logprobs': None, 'finish_reason': None}], 'usage': {'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None}}
```

### Model family StarCoder
```log
bigdl-llm: mem per token =   313912 bytes
bigdl-llm:     load time =   479.41 ms
bigdl-llm:   sample time =    34.29 ms
bigdl-llm:  predict time =  1365.27 ms / 19.50 ms per token
bigdl-llm:    total time =  1928.46 ms

```