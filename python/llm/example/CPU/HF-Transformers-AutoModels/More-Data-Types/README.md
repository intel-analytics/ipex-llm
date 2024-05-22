# IPEX-LLM Transformers Low-Bit Inference Pipeline for Large Language Model

In this example, we show a pipeline to apply IPEX-LLM low-bit optimizations (including INT8/INT5/INT4) to any Hugging Face Transformers model, and then run inference on the optimized low-bit model.

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run Example
```bash
python ./transformers_low_bit_pipeline.py --repo-id-or-model-path decapoda-research/llama-7b-hf --low-bit sym_int5 --save-path ./llama-7b-sym_int5
```
arguments info:
- `--repo-id-or-model-path`: str value, argument defining the huggingface repo id for the large language model to be downloaded, or the path to the huggingface checkpoint folder, the value is 'decapoda-research/llama-7b-hf' by default.
- `--low-bit`: str value, options are sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8. (sym_int4 means symmetric int 4, asym_int4 means asymmetric int 4, etc.). Relevant low bit optimizations will be applied to the model.
- `--save-path`: str value, the path to save the low-bit model. Then you can load the low-bit directly.
- `--load-path`: optional str value. The path to load low-bit model.


## Sample Output for Inference
### 'decapoda-research/llama-7b-hf' Model
```log
Prompt: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
Output: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She wanted to be a princess, and she wanted to be a pirate. She wanted to be a superhero, and she wanted to be
Model and tokenizer are saved to ./llama-7b-sym_int5
```

### Load low-bit model
Command to run:
```bash
python ./transformers_low_bit_pipeline.py --load-path ./llama-7b-sym_int5
```
Output log:
```log
Prompt: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
Output: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She wanted to be a princess, and she wanted to be a pirate. She wanted to be a superhero, and she wanted to be
```

