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

from typing import List, Optional, Tuple, Union
import warnings
import os

import torch
from torch import nn
import time
import argparse
import ipex_llm

from ipex_llm.transformers import convert_model_hybrid
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from transformers.cache_utils import Cache, DynamicCache


PROMPT_FORMAT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {prompt}.
Assistant: <think>
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="If \( a > 1 \), then the sum of the real solutions of \( \sqrt{a} - \sqrt{a + x} = x \) is equal to:",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--load-path', type=str, default=None,
                        help='The path to load the low-bit model.')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    load_path = args.load_path
    if load_path:
        model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(load_path,
                                              trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    load_in_4bit=True,
                                                    optimize_model=True,
                                                    trust_remote_code=True,
                                                    use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    #model = model.bfloat16()
    model = convert_model_hybrid(model)
    print(model)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)

        # start inference
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
