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

import torch
import os
import time
import argparse
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer INT4 example')
    parser.add_argument('--repo-id-or-model-path', type=str, default="decapoda-research/llama-7b-hf",
                        choices=['decapoda-research/llama-7b-hf', 'THUDM/chatglm-6b'],
                        help='The huggingface repo id for the large language model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    if model_path == 'decapoda-research/llama-7b-hf':
        # load_in_quant="q4_0" in bigdl.llm.transformers will convert
        # the relevant layers in the model into int4 format
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_quant="q4_0")
        tokenizer = LlamaTokenizer.from_pretrained(model_path)

        input_str = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

        with torch.inference_mode():
            st = time.time()
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            end = time.time()
        print('Prompt:', input_str)
        print('Output:', output_str)
        print(f'Inference time: {end-st} s')
    elif model_path == 'THUDM/chatglm-6b':
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_quant="q4_0")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        input_str = "晚上睡不着应该怎么办"

        with torch.inference_mode():
            st = time.time()
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            end = time.time()
        print('Prompt:', input_str)
        print('Output:', output_str)
        print(f'Inference time: {end-st} s')
