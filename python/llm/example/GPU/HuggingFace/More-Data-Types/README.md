# IPEX-LLM Transformers Low-Bit Inference Pipeline (FP8, FP6, FP4, INT4 and more)

In this example, we show a pipeline to apply IPEX-LLM low-bit optimizations (including **FP8/INT8/FP6/FP4/INT4**) to any Hugging Face Transformers model, and then run inference on the optimized low-bit model.

## Prepare Environment
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## Run Example
```bash
python ./transformers_low_bit_pipeline.py --repo-id-or-model-path meta-llama/Llama-2-7b-chat-hf --low-bit fp4 --save-path ./llama-2-7b-fp4
```
arguments info:
- `--repo-id-or-model-path`: str value, argument defining the huggingface repo id for the large language model to be downloaded, or the path to the huggingface checkpoint folder, the value is `meta-llama/Llama-2-7b-chat-hf` by default.
- `--low-bit`: str value, options are fp8, fp6, sym_int8, fp4, sym_int4, mixed_fp8 or mixed_fp4. Relevant low bit optimizations will be applied to the model.
- `--save-path`: str value, the path to save the low-bit model. Then you can load the low-bit directly.
- `--load-path`: optional str value. The path to load low-bit model.


## Sample Output for Inference
### `meta-llama/Llama-2-7b-chat-hf` Model
```log
Prompt: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
Output: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. But her parents were always telling her to stay at home and be careful. They were worried about her safety and didn't want her to get hurt
Model and tokenizer are saved to ./llama-2-7b-fp4
```

### Load low-bit model
Command to run:
```bash
python ./transformers_low_bit_pipeline.py --load-path ./llama-2-7b-fp4
```
Output log:
```log
Prompt: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun
Output: Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. But her parents were always telling her to stay at home and be careful. They were worried about her safety and didn't want her to get hurt
```
