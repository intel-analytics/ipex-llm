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

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import time
import os
import argparse
from ipex_llm import optimize_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `chat()` API for InternLM-XComposer model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="internlm/internlm-xcomposer-vl-7b",
                        help='The huggingface repo id for the InternLM-XComposer model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Image path for the input image that the chat will focus on')
    parser.add_argument('--n-predict', type=int, default=512, help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image = args.image_path

    # Load model
    # For successful IPEX-LLM optimization on InternLM-XComposer, skip the 'qkv' module during optimization
    model = AutoModelForCausalLM.from_pretrained(model_path, device='cpu', load_in_4bit=True,
                                                 trust_remote_code=True, modules_to_not_convert=['qkv'])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.tokenizer = tokenizer

    history = None
    while True:
        try:
            user_input = input("User: ")
        except EOFError:
            user_input = ""
        if not user_input:
            print("exit...")
            break

        response, history = model.chat(text=user_input, image=image , history = history)
        print(f'Bot: {response}', end="")
        image = None

