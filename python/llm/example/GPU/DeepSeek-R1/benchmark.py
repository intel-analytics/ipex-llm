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
import numpy as np

from ipex_llm.transformers import AutoModelForCausalLM, convert_model_hybrid
from ipex_llm.utils.benchmark_util_deepseek import BenchmarkWrapper

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
    parser.add_argument('--warm-up', type=int, default=1,
                        help='Num of warm-up trials.')
    parser.add_argument('--num-trials', type=int, default=1,
                        help='Num of trials to run.')

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
    
    model = model.bfloat16()
    model = convert_model_hybrid(model)
    print(model)

    model = BenchmarkWrapper(model)
    e2e_time_list = []
    prefill_time_list = []
    rest_cost_mean_list = []

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # ipex_llm model needs a warmup, then inference time can be accurate
        for i in range(args.warm_up):
            output = model.generate(input_ids,
                                    max_new_tokens=args.n_predict,
                                    min_new_tokens=args.n_predict)

        # start inference
        for i in range(args.num_trials):
            st = time.time()
            output = model.generate(input_ids,
                                    max_new_tokens=args.n_predict,
                                    min_new_tokens=args.n_predict)
            torch.xpu.synchronize()
            end = time.time()
            output = output.cpu()
            e2e_time_list.append(end - st)
            prefill_time_list.append(model.first_cost)
            rest_cost_mean_list.append(model.rest_cost_mean)

        print('-'*20, 'Performance', '-'*20)
        print(f"End-to-end time: {np.mean(e2e_time_list)} s")
        print(f"Prefill time: {np.mean(prefill_time_list)} s")
        print(f"Rest cost mean: {np.mean(rest_cost_mean_list) * 1000} ms")
