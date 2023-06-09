# Inference Pipeline for LLaMA Family Models in INT4 Data Type

In this example, we show a pipeline to conduct inference on a converted low-precision (int4) large language model in LLaMA family, using `bigdl-llm`.

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
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: optional argument defining the huggingface repo id from which the LLaMA family model is downloaded, or the path to the huggingface checkpoint folder for LLaMA family model. It is default to be `'decapoda-research/llama-7b-hf'`
- `--promp PROMPT`: optional argument defining the prompt to be infered. It is default to be `'Q: What is AI? A:'`.

## Sample Output for Inference
```log
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may takes some time.

The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'LLaMATokenizer'. 
The class this function is called from is 'LlamaTokenizer'.
Inference time: 5.0778892040252686 s
Output:
['It’s the ability of computers to perform tasks that usually require human intelligence.\n WORLD WAR II: 75 YEARS LAT']
--------------------  bigdl-llm based tokenizer  --------------------
Inference time: 4.1576244831085205 s
Output:
[" It's everything\nEthics and artificial intelligence have been a hot topic this year, as researchers and the public wrestle with the implications of this"]
--------------------  fast forward  --------------------
Llama.generate: prefix-match hit

llama_print_timings:        load time =  2501.46 ms
llama_print_timings:      sample time =    13.03 ms /    32 runs   (    0.41 ms per token)
llama_print_timings: prompt eval time =   141.13 ms /     9 tokens (   15.68 ms per token)
llama_print_timings:        eval time =  3646.95 ms /    31 runs   (  117.64 ms per token)
llama_print_timings:       total time =  3900.21 ms
Inference time (fast forward): 3.900430917739868 s
Output:
{'id': 'cmpl-f3c5482a-b84e-4363-a85c-89cf7d23ff51', 'object': 'text_completion', 'created': 1686294953, 'model': '/disk5/yuwen/llama/ggml-llama-7b-q4_0-fp32.bin', 'choices': [{'text': ' It’s the latest hot topic in tech. From virtual assistants to driverless cars, machine learning to big data analytics. We hear it a', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 32, 'total_tokens': 42}}
```