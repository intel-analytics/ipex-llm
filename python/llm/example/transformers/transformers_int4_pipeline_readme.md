# BigDL-LLM Transformers INT4 Inference Pipeline for Large Language Model

In this example, we show a pipeline to apply BigDL-LLM INT4 optimizations to any Hugging Face Transformers model, and then run inference on the optimized INT4 model.

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all]
```

## Run Example
```bash
python ./transformers_int4_pipeline.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH
```
arguments info:
- `--repo-id-or-model-path MODEL_PATH`: argument defining the huggingface repo id for the large language model to be downloaded, or the path to the huggingface checkpoint folder.

  > **Note** In this example, `--repo-id-or-model-path MODEL_PATH` is limited be one of `['decapoda-research/llama-7b-hf', 'THUDM/chatglm-6b', 'fnlp/moss-moon-003-sft']` to better demonstrate English and Chinese support. And it is default to be `'decapoda-research/llama-7b-hf'`.

## Sample Output for Inference
### 'decapoda-research/llama-7b-hf' Model
```log
Prompt: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
Output: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She wanted to be a hero. She wanted to be a hero, but she didn't know how. She didn't know how to be a
Inference time: xxxx s
```

### 'THUDM/chatglm-6b' Model
```log
Prompt: 晚上睡不着应该怎么办
Output: 晚上睡不着应该怎么办 晚上睡不着可能会让人感到焦虑和不安,但以下是一些可能有用的建议:

1. 放松身体和思维:尝试进行深呼吸、渐进性
Inference time: xxxx s
```

### 'fnlp/moss-moon-003-sft' Model (16B)
Prompt: 五部值得推荐的科幻电影包括
Output: 五部值得推荐的科幻电影包括《银翼杀手》、《星际穿越》、《黑客帝国》、《异形》和《第五元素》。这些电影都有着独特的风格和故事情节，值得一看。银翼
Inference time: xxxx s