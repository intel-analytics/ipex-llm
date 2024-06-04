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

# Some parts of this file is adapted from
# https://github.com/TimDettmers/bitsandbytes/blob/0.39.1/bitsandbytes/nn/modules.py
# which is licensed under the MIT license:
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
import deepspeed
from ipex_llm import optimize_model
import torch
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--local_rank', type=int, default=0, help='this is automatically set when using deepspeed launcher')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("RANK", "-1")) # RANK is automatically set by CCL distributed backend
    if local_rank == -1: # args.local_rank is automatically set by deepspeed subprocess command
        local_rank = args.local_rank

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

    # Apply IPEX-LLM INT4 optimizations on transformers
    model = optimize_model(model.module.to(f'cpu'), low_bit='sym_int4')

    model = model.to(f'cpu:{local_rank}')

    print(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        # Batch tokenizing
        prompt = args.prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(f'cpu:{local_rank}')
        # ipex-llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                use_cache=True)
        # start inference
        start = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)
        end = time.time()
        if local_rank == 0:
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            print('-'*20, 'Output', '-'*20)
            print(output_str)
            print(f'Inference time: {end - start} s')
