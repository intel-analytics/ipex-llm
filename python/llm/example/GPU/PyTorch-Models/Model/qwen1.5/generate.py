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

from transformers import AutoTokenizer
from ipex_llm import optimize_model
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Qwen1.5-7B-Chat')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen1.5-7B-Chat",
                        help='The huggingface repo id for the Qwen1.5 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer') 
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    
    from transformers import AutoModelForCausalLM
    from ipex_llm import optimize_model
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype = torch.float16,
                                                 use_cache=True)
    model = optimize_model(model)
    model = model.to("xpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    prompt = args.prompt
    # Generate predicted tokens
    with torch.inference_mode():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
        model_inputs = tokenizer([text], return_tensors="pt").to("xpu")
        # warmup
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
            )
        
        st = time.time()
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=args.n_predict
            )
        torch.xpu.synchronize()
        end = time.time()
        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(response)
