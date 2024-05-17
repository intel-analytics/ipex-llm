# IPEX-LLM Native INT4 Inference Pipeline for Large Language Model

In this example, we show a pipeline to convert a large language model to IPEX-LLM native INT4 format, and then run inference on the converted INT4 model.

> **Note**: IPEX-LLM native INT4 format currently supports model family **LLaMA** (such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.), **LLaMA 2** (such as Llama-2-7B-chat, Llama-2-13B-chat), **GPT-NeoX** (such as RedPajama), **BLOOM** (such as Phoenix) and **StarCoder**.

## Prepare Environment
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
```

## Run Example
```bash
python ./native_int4_pipeline.py --thread-num THREAD_NUM --model-family MODEL_FAMILY --repo-id-or-model-path MODEL_PATH
```
arguments info:
- `--thread-num THREAD_NUM`: **required** argument defining the number of threads to use for inference. It is default to be `2`.
- `--model-family MODEL_FAMILY`: **required** argument defining the model family of the large language model (supported option: `'llama'`, `'llama2'`, `'gptneox'`, `'bloom'`, `'starcoder'`). It is default to be `'llama'`.
- `--repo-id-or-model-path MODEL_PATH`: **required** argument defining the path to the huggingface checkpoint folder for the model.

  > **Note** `MODEL_PATH` should fits your inputed `MODEL_FAMILY`.
- `--promp PROMPT`: optional argument defining the prompt to be infered. It is default to be `'Once upon a time, there existed a little girl who liked to have adventures. '`.
- `--tmp-path TMP_PATH`: optional argument defining the path to store intermediate model during the conversion process. It is default to be `'/tmp'`.

## Sample Output for Inference
### Model family LLaMA
#### [lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3)
```log
--------------------  ipex-llm based tokenizer  --------------------
Inference time: xxxx s
Output:
['\n She was always exploring new places and meeting new people.  One day, she stumbled upon a mysterious door in the woods that led her to']
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may take some time.

Inference time: xxxx s
Output:
['\nShe had read so many stories about brave heroes and their magical journeys that she decided to set out on her own adventure. \n']
--------------------  fast forward  --------------------

ipex-llm timings:        load time =    xxxx ms
ipex-llm timings:      sample time =    xxxx ms /    32 runs   (    xxxx ms per token)
ipex-llm timings: prompt eval time =    xxxx ms /     1 tokens (    xxxx ms per token)
ipex-llm timings:        eval time =    xxxx ms /    32 runs   (    xxxx ms per token)
ipex-llm timings:       total time =    xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-e5811030-cc60-462b-9857-13d43e3a1896', 'object': 'text_completion', 'created': 1690450682, 'model': './ipex_llm_llama_q4_0.bin', 'choices': [{'text': '\nShe was a curious and brave child, always eager to explore the world around her. She loved nothing more than setting off into the woods or down to the', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 19, 'completion_tokens': 32, 'total_tokens': 51}}
```

### Model family LLaMA 2
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
--------------------  ipex-llm based tokenizer  --------------------
Inference time: xxxx s
Output:
[' She lived in a small village surrounded by vast fields of golden wheat and blue skies.  One day, she decided to go on an adventure to']
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may take some time.

Llama.generate: prefix-match hit
Inference time: xxxx s
Output:
['She was so curious and eager to explore the world around her that she would often find herself in unexpected situations. \nOne day, while wandering through the']
--------------------  fast forward  --------------------
Llama.generate: prefix-match hit

ipex-llm timings:        load time =     xxxx ms
ipex-llm timings:      sample time =     xxxx ms /    32 runs   (    xxxx ms per token)
ipex-llm timings: prompt eval time =     xxxx ms /     1 tokens (    xxxx ms per token)
ipex-llm timings:        eval time =     xxxx ms /    32 runs   (    xxxx ms per token)
ipex-llm timings:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-556b831b-749f-4b06-801e-c920620cb8f5', 'object': 'text_completion', 'created': 1690449478, 'model': './ipex_llm_llama_q4_0.bin', 'choices': [{'text': ' She lived in a small village at the edge of a big forest, surrounded by tall trees and sparkling streams.  One day, while wandering around the', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 19, 'completion_tokens': 32, 'total_tokens': 51}}
```

### Model family GPT-NeoX
#### [togethercomputer/RedPajama-INCITE-7B-Chat](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat)
```log
--------------------  ipex-llm based tokenizer  --------------------
Inference time: xxxx s
Output:
['\nThis was no surprise since her mom and dad both loved adventure too. But what really stood out about this little girl is that she loved the stories! Her']
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may take some time.

Inference time: xxxx s
Output:
['\nFirst she got lost in the woods and it took some really tough searching by her parents to find her. But they did! Then one day when she was']
--------------------  fast forward  --------------------

gptneox_print_timings:        load time =     xxxx ms
gptneox_print_timings:      sample time =     xxxx ms /    32 runs   (    xxxx ms per run)
gptneox_print_timings: prompt eval time =     xxxx ms /    18 tokens (    xxxx ms per token)
gptneox_print_timings:        eval time =     xxxx ms /    31 runs   (    xxxx ms per run)
gptneox_print_timings:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-8b17585d-635a-43af-94a0-bd9c19ffc5a8', 'object': 'text_completion', 'created': 1690451587, 'model': './ipex_llm_gptneox_q4_0.bin', 'choices': [{'text': '\nOn one fine day her mother brought home an old shoe box full of toys and gave it to her daughter as she was not able to make the toy house', 'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 18, 'completion_tokens': 32, 'total_tokens': 50}}
```

### Model family BLOOM
#### [FreedomIntelligence/phoenix-inst-chat-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b)
```log
--------------------  ipex-llm based tokenizer  --------------------
Inference time: xxxx s
Output:
[' She was always eager to explore new places and meet new people. One day, she decided to embark on an epic journey across the land of the giants']
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may take some time.

Inference time: xxxx s
Output:
[' She loved exploring the world and trying new things. One day, she decided to embark on an epic journey across the land of the giants. The little']
--------------------  fast forward  --------------------


inference:    mem per token =     xxxx bytes
inference:      sample time =     xxxx ms
inference: evel prompt time =     xxxx ms / 12 tokens / xxxx ms per token
inference:     predict time =     xxxx ms / 31 tokens / xxxx ms per token
inference:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-e7039a29-dc80-4729-a446-301573a5315f', 'object': 'text_completion', 'created': 1690449783, 'model': './ipex_llm_bloom_q4_0.bin', 'choices': [{'text': ' She had the spirit of exploration, and her adventurous nature drove her to seek out new things every day. Little did she know that her adventures would take an', 'index': 0, 'logprobs': None, 'finish_reason': None}], 'usage': {'prompt_tokens': 17, 'completion_tokens': 32, 'total_tokens': 49}}
```

### Model family StarCoder
#### [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
```log
--------------------  ipex-llm based tokenizer  --------------------
Inference time: xxxx s
Output:
['\nOne day, she went on an adventure with a dragon. \nThe dragon was very angry, and he wanted to eat her.']
--------------------  HuggingFace transformers tokenizer  --------------------
Please note that the loading of HuggingFace transformers tokenizer may take some time.

Inference time: xxxx s
Output:
[' She was called "Alice".  She was very clever, and she loved to play with puzzles.  One day, she was playing with']
--------------------  fast forward  --------------------


ipex-llm:    mem per token =     xxxx bytes
ipex-llm:      sample time =     xxxx ms
ipex-llm: evel prompt time =     xxxx ms / 11 tokens / xxxx ms per token
ipex-llm:     predict time =     xxxx ms / 31 tokens / xxxx ms per token
ipex-llm:       total time =     xxxx ms
Inference time (fast forward): xxxx s
Output:
{'id': 'cmpl-d0266eb2-5e18-4fbc-bcc4-dec236f506f6', 'object': 'text_completion', 'created': 1690450075, 'model': './ipex_llm_starcoder_q4_0.bin', 'choices': [{'text': ' She loved to play with dolls and other stuff, but she loved the most to play with cats and other dogs.  She loved to', 'index': 0, 'logprobs': None, 'finish_reason': None}], 'usage': {'prompt_tokens': 21, 'completion_tokens': 32, 'total_tokens': 53}}
```
