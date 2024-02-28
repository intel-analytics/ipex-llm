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
import intel_extension_for_pytorch as ipex
import time
import argparse

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from bigdl.llm import optimize_model

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/RWKV/rwkv-4-world-7b

RWKV_PROMPT_FORMAT = "Question: {prompt}\n\nAnswer:"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for RWKV model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="RWKV/rwkv-4-world-7b",
                        help='The huggingface repo id for the RWKV model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="你叫什么名字？",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=40,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # First load the model in fp16 dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 trust_remote_code=True,
                                                 low_cpu_mem_usage=True, 
                                                 torch_dtype=torch.half)
    
    # Call the `_rescale_layers` method, prepare to convert to int4
    model.rwkv._rescale_layers()

    # With only one line to enable BigDL-LLM optimization on model
    model = optimize_model(model)
    model = model.to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = RWKV_PROMPT_FORMAT.format(prompt = args.prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to('xpu')

        # ipex model needs a warmup, then inference time can be accurate
        output = model.generate(inputs["input_ids"],  
                                max_new_tokens=args.n_predict)
        
        # start inference
        st = time.time()
        output = model.generate(inputs["input_ids"],  
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        output = output.cpu()
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
