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
import time
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from ipex_llm import optimize_model

# The instruction-tuned models use a chat template that must be adhered to for conversational use.
# see https://huggingface.co/google/codegemma-7b-it#chat-template.
chat = [
    { "role": "user", "content": "Write a hello world program" },
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for CodeGemma model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="google/codegemma-7b-it",
                        help='The huggingface repo id for the CodeGemma model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Write a hello world program",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # With only one line to enable IPEX-LLM optimization on model
    # To fix the issue that the output of codegemma-7b-it is abnormal, skip the 'lm_head' module during optimization
    model = optimize_model(model, modules_to_not_convert=["lm_head"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        chat[0]['content'] = args.prompt
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
