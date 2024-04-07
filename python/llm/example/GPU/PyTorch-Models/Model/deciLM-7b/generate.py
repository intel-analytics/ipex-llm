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

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/Deci/DeciLM-7B-instruct#prompt-template
SYSTEM_PROMPT_TEMPLATE ="""
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
{prompt}
### Assistant:
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for DeciLM-7B model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Deci/DeciLM-7B-instruct",
                        help='The huggingface repo id for the DeciLM-7B (e.g. `Deci/DeciLM-7B-instruct`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # With only one line to enable IPEX-LLM optimization on model
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = optimize_model(
        model, 
        cpu_embedding=True
    )
    model = model.to('xpu')
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = SYSTEM_PROMPT_TEMPLATE.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        print(output_str)

