import os
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
import deepspeed
from bigdl.llm import optimize_model
import torch
import intel_extension_for_pytorch as ipex
import time
import argparse
from benchmark_util import BenchmarkWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--local_rank', type=str, default=0, help='this is automatically set when using deepspeed launcher')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("RANK", "1")) # RANK is automatically set by distributed backend

    # Native Huggingface transformers loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=True
    )

    # Parallelize model on deepspeed
    model = deepspeed.init_inference(
        model,
        mp_size = world_size,
        dtype=torch.float16,
        replace_method="auto"
    )

    # Apply BigDL-LLM INT4 optimizations on transformers
    model = optimize_model(model.module.to(f'cpu'), low_bit='sym_int4')

    model = model.to(f'cpu:{local_rank}')

    print(model)
    model = BenchmarkWrapper(model, do_print=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        # Batch tokenizing
        prompt = args.prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(f'cpu:{local_rank}')
        # ipex model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                use_cache=True)
        # start inference
        start = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)
        end = time.time()
        if local_rank == 0:
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            print('-'*20, 'Output', '-'*20)
            print(output_str)
            print(f'Inference time: {end - start} s')


