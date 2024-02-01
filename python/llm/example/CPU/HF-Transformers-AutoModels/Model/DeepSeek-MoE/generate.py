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
import numpy as np
from transformers import AutoTokenizer, GenerationConfig

# you could tune the prompt based on your own model,
# here the prompt tuning refers to: https://huggingface.co/WisdomShell/CodeShell-7B

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for CodeShell model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="/mnt/disk1/models/deepseek-moe-16b-chat",
                        help='The huggingface repo id for the CodeShell model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=100,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # from bigdl.llm.transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True).eval()
    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    # optimize model
    from transformers import AutoModelForCausalLM
    from bigdl.llm import optimize_model
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model = optimize_model(model)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = args.prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens=args.n_predict)
        cost = []
        for _ in range(3):
            st = time.time()
            output = model.generate(input_ids, max_new_tokens=args.n_predict)
            end = time.time()
            cost.append(end-st)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            print('-'*20, 'Prompt', '-'*20)
            print(prompt)
            print('-'*20, 'Output', '-'*20)
            print(output_str)
        ave_time = np.average(cost)
        print(f'Inference time: {ave_time} s')
        
