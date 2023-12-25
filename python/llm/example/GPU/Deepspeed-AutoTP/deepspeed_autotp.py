#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import torch
import transformers
import deepspeed

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return int(default)
 
local_rank = get_int_from_env(["LOCAL_RANK","PMI_RANK"], "0")
world_size = get_int_from_env(["WORLD_SIZE","PMI_SIZE"], "1")

from bigdl.llm import optimize_model

import torch
import intel_extension_for_pytorch as ipex
import time
import argparse

from transformers import AutoModelForCausalLM  # export AutoModelForCausalLM from transformers so that deepspeed use it
from transformers import LlamaTokenizer, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(args.repo_id_or_model_path,
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 use_cache=True)

    model = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.float16,
        replace_method="auto",
    )

    # move model to cpu and use bigdl-llm `optimize_model` to convert the
    # model into optimized low bit format
    # convert the rest of the model into float16 to reduce allreduce traffic
    model = optimize_model(model.module.to(f'cpu'), low_bit='sym_int4').to(torch.float16)

    # move model back to xpu
    model = model.to(f'xpu:{local_rank}')

    print(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        # prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
        prompt = args.prompt
        # input_str = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(f'xpu:{local_rank}')
        # ipex model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                use_cache=True)

        # start inference
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        if local_rank == 0:
            output = output.cpu()
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'Inference time: {end-st} s')
            print('-'*20, 'Prompt', '-'*20)
            print(prompt)
            print('-'*20, 'Output', '-'*20)
            print(output_str)
