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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for minicpm model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="openbmb/MiniCPM-2B-sft-bf16",
                        help='The huggingface repo id for the minicpm model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 use_cache=True)

    # With only one line to enable IPEX-LLM optimization on model
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the optimize_model function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = optimize_model(model)
    model = model.to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():

        # here the prompt formatting refers to: https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/blob/79fbb1db171e6d8bf77cdb0a94076a43003abd9e/modeling_minicpm.py#L1320
        chat = [
            { "role": "user", "content": args.prompt },
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')

        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        # start inference
        st = time.time()

        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=False)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
